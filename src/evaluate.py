# evaluate.py
# AdVox V2 — Evaluation script.
# Computes Topic Accuracy, Sentiment MAE, BLEU-4 for Action and Reason.
#
# Run from V2_AdVox_QFormer/ :
#   python src/evaluate.py

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import torch
import torch.nn.functional as F
from torch.amp import autocast
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from utils   import load_config
from dataset import get_dataloaders
from model   import AdVox

# ── Topic Accuracy ─────────────────────────────────────────────────────────────

def evaluate_topic(log_probs: torch.Tensor,
                   soft_labels: torch.Tensor) -> tuple:
    """
    Compute topic accuracy — predicted class vs argmax of soft labels.

    Args:
        log_probs   : (B, 38) — log-softmax output
        soft_labels : (B, 38) — soft probability targets

    Returns:
        n_correct : int
        n_total   : int
    """
    preds   = log_probs.argmax(dim=-1)       # (B,)
    targets = soft_labels.argmax(dim=-1)     # (B,)
    n_correct = (preds == targets).sum().item()
    n_total   = preds.size(0)
    return n_correct, n_total


# ── Sentiment MAE ──────────────────────────────────────────────────────────────

def evaluate_sentiment(logits: torch.Tensor,
                       soft_labels: torch.Tensor,
                       mask: torch.Tensor) -> tuple:
    """
    Compute sentiment MAE on images that have sentiment annotations.

    Args:
        logits      : (B, 30) — raw logits
        soft_labels : (B, 30) — soft multi-hot targets
        mask        : (B,)    — True if image has sentiment annotation

    Returns:
        mae_sum   : float — sum of MAE over valid samples
        n_valid   : int   — number of valid samples
    """
    if mask.sum() == 0:
        return 0.0, 0

    preds   = torch.sigmoid(logits[mask])    # (N, 30)
    targets = soft_labels[mask]              # (N, 30)
    mae     = F.l1_loss(preds, targets, reduction="mean").item()
    return mae * mask.sum().item(), mask.sum().item()


# ── BLEU-4 ─────────────────────────────────────────────────────────────────────

def evaluate_bleu(hypotheses: list, references: list) -> float:
    """
    Compute corpus BLEU-4 score.

    Args:
        hypotheses : list of generated strings
        references : list of reference strings

    Returns:
        bleu4 : float in [0, 1]
    """
    # BLEU expects tokenized lists
    hyp_tokens = [h.lower().split() for h in hypotheses]
    ref_tokens = [[r.lower().split()] for r in references]   # list of list of list

    smoother = SmoothingFunction().method1
    return corpus_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother)


# ── Full Evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: AdVox, loader, config: dict,
             device: torch.device) -> dict:
    """
    Full evaluation over a dataloader.

    Computes:
        - Topic accuracy
        - Sentiment MAE
        - BLEU-4 Action
        - BLEU-4 Reason

    Returns:
        metrics : dict of all metric values
    """
    model.eval()
    gen_cfg = config["generation"]

    # Accumulators
    topic_correct = 0
    topic_total   = 0
    sent_mae_sum  = 0.0
    sent_total    = 0

    action_hyps = []
    action_refs = []
    reason_hyps = []
    reason_refs = []

    print("  Running evaluation...")

    for batch_idx, batch in enumerate(loader):

        images    = batch["image"].to(device, non_blocking=True)
        sent_mask = batch["sent_mask"].to(device, non_blocking=True)

        # ── Classification metrics (fast — no generation) ──
        with autocast("cuda"):
            out = model(images)   # no labels — inference mode

        # Topic accuracy
        c, t = evaluate_topic(
            out["topic_log_probs"],
            batch["topic_soft"].to(device)
        )
        topic_correct += c
        topic_total   += t

        # Sentiment MAE
        mae_s, n_s = evaluate_sentiment(
            out["sentiment_logits"],
            batch["sent_soft"].to(device),
            sent_mask
        )
        sent_mae_sum += mae_s
        sent_total   += n_s

        # ── Text generation (slower) ───────────────────────
        action_texts = model.generate_text(
            images, task="action",
            max_new_tokens = gen_cfg["max_new_tokens"],
            num_beams      = gen_cfg["num_beams"],
            min_new_tokens = gen_cfg["min_new_tokens"],
        )
        reason_texts = model.generate_text(
            images, task="reason",
            max_new_tokens = gen_cfg["max_new_tokens"],
            num_beams      = gen_cfg["num_beams"],
            min_new_tokens = gen_cfg["min_new_tokens"],
        )

        # Decode reference texts from token ids
        action_ref = model.tokenizer.batch_decode(
            batch["action_ids"], skip_special_tokens=True
        )
        reason_ref = model.tokenizer.batch_decode(
            batch["reason_ids"], skip_special_tokens=True
        )

        action_hyps.extend(action_texts)
        action_refs.extend(action_ref)
        reason_hyps.extend(reason_texts)
        reason_refs.extend(reason_ref)

        if (batch_idx + 1) % 20 == 0:
            print(f"    Evaluated {batch_idx + 1}/{len(loader)} batches...")

    # ── Compute final metrics ──────────────────────────────
    topic_acc  = topic_correct / topic_total if topic_total > 0 else 0.0
    sent_mae   = sent_mae_sum  / sent_total  if sent_total  > 0 else 0.0
    bleu_action = evaluate_bleu(action_hyps, action_refs)
    bleu_reason = evaluate_bleu(reason_hyps, reason_refs)

    metrics = {
        "topic_accuracy": topic_acc,
        "topic_correct":  topic_correct,
        "topic_total":    topic_total,
        "sentiment_mae":  sent_mae,
        "bleu4_action":   bleu_action,
        "bleu4_reason":   bleu_reason,
    }

    return metrics


# ── Print Metrics ──────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, epoch: int = None):
    """Pretty print evaluation metrics."""
    header = f"Evaluation Results" if epoch is None else f"Epoch {epoch} Evaluation"
    print(f"\n  {'─'*45}")
    print(f"  {header}")
    print(f"  {'─'*45}")
    print(f"  Topic Accuracy   : {metrics['topic_accuracy']*100:.2f}%  "
          f"({metrics['topic_correct']}/{metrics['topic_total']})")
    print(f"  Sentiment MAE    : {metrics['sentiment_mae']:.4f}")
    print(f"  BLEU-4 Action    : {metrics['bleu4_action']:.4f}")
    print(f"  BLEU-4 Reason    : {metrics['bleu4_reason']:.4f}")
    print(f"  {'─'*45}\n")

    # Target comparison
    print(f"  Targets (sanity check):")
    print(f"    Topic Accuracy  > 90%    {'✓' if metrics['topic_accuracy'] > 0.90 else '✗'}")
    print(f"    Sentiment MAE   < 0.06   {'✓' if metrics['sentiment_mae']  < 0.06 else '✗'}")
    print(f"    BLEU-4 Action   > 0.06   {'✓' if metrics['bleu4_action']   > 0.06 else '✗'}")
    print(f"    BLEU-4 Reason   > 0.01   {'✓' if metrics['bleu4_reason']   > 0.01 else '✗'}")
    print(f"  {'─'*45}\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: config not found at {config_path}")
        print("Run from V2_AdVox_QFormer/ directory.")
        sys.exit(1)

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Building model...")
    model = AdVox(config).to(device)

    # Load checkpoint
    ckpt_path = os.path.join(
        config["paths"]["checkpoints_dir"], "advox_v2_best.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("Train the model first using train.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Get val loader
    _, val_loader = get_dataloaders(config)

    # Run evaluation
    metrics = evaluate(model, val_loader, config, device)
    print_metrics(metrics)