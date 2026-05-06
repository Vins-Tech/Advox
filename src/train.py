# train.py
# AdVox V2 — Training script for Q-Former + Flan-T5 architecture.
#
# Training strategy:
#   - Epochs 1-2 : ViT encoder frozen, only Q-Former + heads train
#   - Epoch  3+  : Full model trains (encoder unfrozen)
#   - Flan-T5    : Always frozen, never updated
#   - AMP        : Mixed precision throughout for VRAM efficiency
#   - Loss       : Weighted sum of 4 task losses with masking
#
# Run from V2_AdVox_QFormer/ :
#   python src/train.py
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from utils   import load_config
from dataset import get_dataloaders
from model   import AdVox


# ── Loss Functions ─────────────────────────────────────────────────────────────

def topic_loss(log_probs: torch.Tensor,
               soft_labels: torch.Tensor) -> torch.Tensor:
    """
    KL Divergence loss for topic classification.

    Args:
        log_probs   : (B, 38) — log-softmax output from TopicHead
        soft_labels : (B, 38) — soft probability targets from annotators

    Returns:
        scalar loss
    """
    # F.kl_div expects log-probabilities as input, probabilities as target
    # reduction='batchmean' divides by batch size — standard for KL
    return F.kl_div(log_probs, soft_labels, reduction="batchmean")


def sentiment_loss(logits: torch.Tensor,
                   soft_labels: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Masked Binary Cross-Entropy loss for sentiment classification.

    Sentiment annotations only exist for sf0-sf3 and sf10.
    sf4-sf9 images have no sentiment labels — their loss is masked out.

    Args:
        logits      : (B, 30) — raw logits from SentimentHead (no sigmoid)
        soft_labels : (B, 30) — soft multi-hot targets
        mask        : (B,)    — True if image has sentiment annotation

    Returns:
        scalar loss (0.0 if no images in batch have sentiment labels)
    """
    # If no images in this batch have sentiment annotations, return 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    # Select only images that have sentiment annotations
    logits_masked = logits[mask]          # (N_valid, 30)
    labels_masked = soft_labels[mask]     # (N_valid, 30)

    return F.binary_cross_entropy_with_logits(
        logits_masked, labels_masked, reduction="mean"
    )


def compute_total_loss(out: dict, batch: dict,
                       weights: dict, device: torch.device) -> tuple:
    """
    Compute weighted multi-task loss.

    Loss = w_topic * L_topic
         + w_sentiment * L_sentiment  (masked)
         + w_action * L_action
         + w_reason * L_reason

    Args:
        out     : model forward output dict
        batch   : dataloader batch dict
        weights : loss_weights from config
        device  : current device

    Returns:
        total_loss  : scalar weighted sum
        loss_dict   : dict of individual loss values (for logging)
    """
    # Topic loss
    l_topic = topic_loss(
        out["topic_log_probs"],
        batch["topic_soft"].to(device)
    )

    # Sentiment loss (masked)
    l_sent = sentiment_loss(
        out["sentiment_logits"],
        batch["sent_soft"].to(device),
        batch["sent_mask"].to(device)
    )

    # Action + Reason losses come directly from T5 forward pass
    l_action = out["action_loss"]
    l_reason = out["reason_loss"]

    # Weighted sum
    total = (weights["topic"]     * l_topic  +
             weights["sentiment"] * l_sent   +
             weights["action"]    * l_action +
             weights["reason"]    * l_reason)

    loss_dict = {
        "topic":     l_topic.item(),
        "sentiment": l_sent.item(),
        "action":    l_action.item(),
        "reason":    l_reason.item(),
        "total":     total.item(),
    }

    return total, loss_dict


# ── Train One Epoch ────────────────────────────────────────────────────────────

def train_one_epoch(model: AdVox, loader, optimizer, scaler: GradScaler,
                    config: dict, device: torch.device,
                    epoch: int) -> dict:
    """
    Run one full training epoch.

    Returns:
        avg_losses : dict of average losses over the epoch
    """
    model.train()
    weights      = config["loss_weights"]
    grad_clip    = config["training"]["grad_clip"]
    log_interval = config["logging"]["log_every_n_steps"]

    # Accumulators
    running = {"topic": 0., "sentiment": 0., "action": 0.,
               "reason": 0., "total": 0.}
    n_batches = 0
    t_start   = time.time()

    for step, batch in enumerate(loader):

        images = batch["image"].to(device, non_blocking=True)
        a_lbl  = batch["action_labels"].to(device, non_blocking=True)
        r_lbl  = batch["reason_labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── Forward pass with AMP ──────────────────────────
        with autocast("cuda"):
            out = model(
                images,
                action_labels = a_lbl,
                reason_labels = r_lbl,
            )
            loss, loss_dict = compute_total_loss(out, batch, weights, device)

        # ── Backward pass ──────────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        # ── Accumulate losses ──────────────────────────────
        for k in running:
            running[k] += loss_dict[k]
        n_batches += 1

        # ── Logging ────────────────────────────────────────
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                  f"Loss {loss_dict['total']:.4f} "
                  f"(T:{loss_dict['topic']:.3f} "
                  f"S:{loss_dict['sentiment']:.3f} "
                  f"A:{loss_dict['action']:.3f} "
                  f"R:{loss_dict['reason']:.3f}) | "
                  f"Elapsed {elapsed:.0f}s")

    avg = {k: v / n_batches for k, v in running.items()}
    return avg


# ── Validation One Epoch ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: AdVox, loader, config: dict,
             device: torch.device) -> dict:
    """
    Run validation — forward pass only, no gradient updates.

    Returns:
        avg_losses : dict of average losses over the validation set
    """
    model.eval()
    weights = config["loss_weights"]

    running = {"topic": 0., "sentiment": 0., "action": 0.,
               "reason": 0., "total": 0.}
    n_batches = 0

    for batch in loader:

        images = batch["image"].to(device, non_blocking=True)
        a_lbl  = batch["action_labels"].to(device, non_blocking=True)
        r_lbl  = batch["reason_labels"].to(device, non_blocking=True)

        with autocast("cuda"):
            out = model(
                images,
                action_labels = a_lbl,
                reason_labels = r_lbl,
            )
            _, loss_dict = compute_total_loss(out, batch, weights, device)

        for k in running:
            running[k] += loss_dict[k]
        n_batches += 1

    avg = {k: v / n_batches for k, v in running.items()}
    return avg


# ── Checkpoint Helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model: AdVox, optimizer, scheduler, epoch: int,
                    val_loss: float, config: dict):
    ckpt_dir  = config["paths"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "advox_v2_best.pt")

    torch.save({
        "epoch":            epoch,
        "val_loss":         val_loss,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "config":           config,
    }, ckpt_path)
    print(f"  ✓ Checkpoint saved → {ckpt_path}  (val_loss={val_loss:.4f})")


def load_checkpoint(model: AdVox, optimizer, config: dict,
                    device: torch.device):
    """
    Load checkpoint if it exists.
    Returns (start_epoch, best_val_loss).
    """

    latest = os.path.join(config["paths"]["checkpoints_dir"], "advox_v2_latest.pt")
    best   = os.path.join(config["paths"]["checkpoints_dir"], "advox_v2_best.pt")
    ckpt_path = latest if os.path.exists(latest) else best

    if not os.path.exists(ckpt_path):
        print("No checkpoint found — training from scratch.")
        return 1, float("inf"), None

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["val_loss"]
    print(f"Resumed from checkpoint — epoch {ckpt['epoch']}, "
          f"val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss, ckpt.get("scheduler_state", None)

def log_inference_samples(model, config, device, epoch, log_path):
    """Run inference on testing/*.jpg images and append to log file."""
    from utils import get_transforms, load_topics_map, load_sentiments_map
    import glob

    testing_dir = "testing"
    image_paths = sorted(glob.glob(os.path.join(testing_dir, "*.jpg")))
    if not image_paths:
        return

    gen_cfg     = config["generation"]
    tf          = get_transforms(config["dataset"]["image_size"], mode="val")
    ann_dir     = config["paths"]["annotations_dir"]
    topics_map, _  = load_topics_map(ann_dir)
    sents_map,  _  = load_sentiments_map(ann_dir)

    model.eval()
    try:
        lines = [f"\n--- Epoch {epoch} Inference Samples ---\n"]

        with torch.no_grad():
            for img_path in image_paths:
                try:
                    from PIL import Image as PILImage
                    img   = PILImage.open(img_path).convert("RGB")
                    img_t = tf(img).unsqueeze(0).to(device)

                    out          = model(img_t)
                    topic_probs  = out["topic_log_probs"].exp()
                    topic_idx    = topic_probs.argmax(dim=-1).item()
                    topic_label  = topics_map.get(topic_idx, f"class_{topic_idx}")
                    topic_conf   = topic_probs[0, topic_idx].item()

                    sent_probs   = torch.sigmoid(out["sentiment_logits"])[0]
                    top3_sent    = sorted(
                        [(sents_map.get(i, f"class_{i}"), sent_probs[i].item())
                        for i in range(30)],
                        key=lambda x: x[1], reverse=True
                    )[:3]

                    action = model.generate_text(
                        img_t, task="action",
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        num_beams=gen_cfg["num_beams"],
                        min_new_tokens=gen_cfg["min_new_tokens"],
                    )[0]
                    reason = model.generate_text(
                        img_t, task="reason",
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        num_beams=gen_cfg["num_beams"],
                        min_new_tokens=gen_cfg["min_new_tokens"],
                    )[0]

                    lines.append(
                        f"[{os.path.basename(img_path)}]\n"
                        f"  Topic    : {topic_label} ({topic_conf*100:.1f}%)\n"
                        f"  Sentiments: {', '.join(f'{l}({p:.2f})' for l,p in top3_sent)}\n"
                        f"  Action   : {action}\n"
                        f"  Reason   : {reason}\n"
                    )
                except Exception as e:
                    lines.append(f"[{os.path.basename(img_path)}] ERROR: {e}\n")

        with open(log_path, "a") as f:
            f.writelines(lines)
            f.write("\n")

    finally:
        model.train()
    

# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(config: dict):
    """Full training loop."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"AdVox V2 Training")
    print(f"Device  : {device}")
    print(f"Epochs  : {config['training']['epochs']}")
    print(f"Batch   : {config['training']['batch_size']}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────
    print("Building dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}\n")

    # ── Model ─────────────────────────────────────────────
    print("Building model...")
    model = AdVox(config).to(device)

    # Freeze encoder at start
    model.freeze_encoder()

    print(f"\nTrainable params : {model.count_trainable_params():,}")
    print(f"Total params     : {model.count_total_params():,}\n")

    # ── Optimizer ─────────────────────────────────────────
    # Only pass trainable parameters to optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["training"]["learning_rate"],
        weight_decay = config["training"]["weight_decay"],
    )

    # Cosine LR scheduler with warmup
    total_epochs  = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor = 0.1,
                end_factor   = 1.0,
                total_iters  = warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = max(1, total_epochs - warmup_epochs),
            ),
        ],
        milestones = [warmup_epochs],
    )

    # AMP scaler
    scaler = GradScaler("cuda")

    # ── Resume from checkpoint if exists ──────────────────
    start_epoch, best_val_loss, sched_state = load_checkpoint(
        model, optimizer, config, device
    )
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)

    # ── Training Loop ─────────────────────────────────────
    logs_dir     = config["paths"]["logs_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    log_path     = os.path.join(logs_dir, "train_log.txt")

    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'='*60}\nNew training run\n{'='*60}\n")

    for epoch in range(start_epoch, total_epochs + 1):

        print(f"\nEpoch {epoch}/{total_epochs} "
              f"(LR={scheduler.get_last_lr()[0]:.6f})")
        print("-" * 50)

        # Train
        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, config, device, epoch
        )
        epoch_time = time.time() - t0

        # Validate
        print(f"  Running validation...")
        val_loss = validate(model, val_loader, config, device)

        # Scheduler step
        scheduler.step()

        # ── Print epoch summary ────────────────────────────
        print(f"\n  Epoch {epoch} Summary ({epoch_time:.0f}s)")
        print(f"  Train → total:{train_loss['total']:.4f}  "
              f"topic:{train_loss['topic']:.4f}  "
              f"sent:{train_loss['sentiment']:.4f}  "
              f"action:{train_loss['action']:.4f}  "
              f"reason:{train_loss['reason']:.4f}")
        print(f"  Val   → total:{val_loss['total']:.4f}  "
              f"topic:{val_loss['topic']:.4f}  "
              f"sent:{val_loss['sentiment']:.4f}  "
              f"action:{val_loss['action']:.4f}  "
              f"reason:{val_loss['reason']:.4f}")

        # ── Log to file ────────────────────────────────────
        log_line = (
            f"Epoch {epoch:03d} | "
            f"Train {train_loss['total']:.4f} "
            f"(T:{train_loss['topic']:.4f} "
            f"S:{train_loss['sentiment']:.4f} "
            f"A:{train_loss['action']:.4f} "
            f"R:{train_loss['reason']:.4f}) | "
            f"Val {val_loss['total']:.4f} "
            f"(T:{val_loss['topic']:.4f} "
            f"S:{val_loss['sentiment']:.4f} "
            f"A:{val_loss['action']:.4f} "
            f"R:{val_loss['reason']:.4f}) | "
            f"Time {epoch_time:.0f}s\n"
        )
        with open(log_path, "a") as log_file:
            log_file.write(log_line)

        # ── Save best checkpoint ───────────────────────────
        # Always save latest epoch
        latest_path = os.path.join(config["paths"]["checkpoints_dir"], "advox_v2_latest.pt")
        torch.save({
            "epoch":            epoch,
            "val_loss":         val_loss["total"],
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "config":           config,
        }, latest_path)
        print(f"  ✓ Latest checkpoint saved → epoch {epoch}")

        # Save best separately if improved
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, config)

        log_inference_samples(model, config, device, epoch, log_path)

    print(f"\n{'='*60}")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {config['paths']['checkpoints_dir']}/advox_v2_best.pt")
    print(f"{'='*60}\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run from V2_AdVox_QFormer/
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}")
        print("Run from V2_AdVox_QFormer/ directory.")
        sys.exit(1)

    config = load_config(config_path)
    train(config)