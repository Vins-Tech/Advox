# inference.py
# AdVox V2 — Single image inference script.
# Loads a trained checkpoint and runs all 4 tasks on one image.
#
# Run from V2_AdVox_QFormer/ :
#   python src/inference.py --image path/to/image.jpg
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import argparse
import torch
from PIL import Image

from utils import load_config, get_transforms, load_topics_map, load_sentiments_map
from model import AdVox

# ── Inference Function ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: AdVox, image_path: str,
                  config: dict, device: torch.device,
                  topics_map: dict, sentiments_map: dict) -> dict:
    """
    Run all 4 AdVox tasks on a single image.

    Args:
        model         : loaded AdVox model
        image_path    : path to image file
        config        : config dict
        device        : cuda or cpu
        topics_map    : id_to_label dict for topics
        sentiments_map: id_to_label dict for sentiments

    Returns:
        results : dict with topic, sentiments, action, reason
    """
    gen_cfg = config["generation"]

    # ── Load and preprocess image ──────────────────────────
    image = Image.open(image_path).convert("RGB")
    tf    = get_transforms(config["dataset"]["image_size"], mode="val")
    img_t = tf(image).unsqueeze(0).to(device)      # (1, 3, 224, 224)

    # ── Forward pass — classification only ────────────────
    model.eval()
    out = model(img_t)                              # no labels = inference mode

    # ── Topic ──────────────────────────────────────────────
    topic_probs = out["topic_log_probs"].exp()      # (1, 38) — convert from log
    topic_idx   = topic_probs.argmax(dim=-1).item()
    topic_conf  = topic_probs[0, topic_idx].item()
    topic_label = topics_map.get(topic_idx, f"class_{topic_idx}")

    # Top-3 topics
    top3_vals, top3_idxs = topic_probs[0].topk(3)
    top3_topics = [
        (topics_map.get(i.item(), f"class_{i.item()}"), v.item())
        for i, v in zip(top3_idxs, top3_vals)
    ]

    # ── Sentiment ──────────────────────────────────────────
    sent_probs = torch.sigmoid(out["sentiment_logits"])[0]  # (30,)

    # Top-3 sentiments above threshold 0.3
    threshold   = 0.3
    active_sent = [
        (sentiments_map.get(i, f"class_{i}"), sent_probs[i].item())
        for i in range(30)
        if sent_probs[i].item() > threshold
    ]
    active_sent.sort(key=lambda x: x[1], reverse=True)
    top3_sentiments = active_sent[:3]

    # ── Action text generation ─────────────────────────────
    action_texts = model.generate_text(
        img_t, task="action",
        max_new_tokens = gen_cfg["max_new_tokens"],
        num_beams      = gen_cfg["num_beams"],
        min_new_tokens = gen_cfg["min_new_tokens"],
    )
    action_text = action_texts[0]

    # ── Reason text generation ─────────────────────────────
    reason_texts = model.generate_text(
        img_t, task="reason",
        max_new_tokens = gen_cfg["max_new_tokens"],
        num_beams      = gen_cfg["num_beams"],
        min_new_tokens = gen_cfg["min_new_tokens"],
    )
    reason_text = reason_texts[0]

    return {
        "topic":          topic_label,
        "topic_conf":     topic_conf,
        "top3_topics":    top3_topics,
        "top3_sentiments": top3_sentiments,
        "action":         action_text,
        "reason":         reason_text,
    }


# ── Pretty Print Results ───────────────────────────────────────────────────────

def print_results(results: dict, image_path: str):
    """Pretty print inference results."""
    print(f"\n{'='*55}")
    print(f"AdVox V2 — Inference Results")
    print(f"Image : {os.path.basename(image_path)}")
    print(f"{'='*55}")

    print(f"\n📦 Topic")
    print(f"   Predicted : {results['topic']}  ({results['topic_conf']*100:.1f}%)")
    print(f"   Top-3     :")
    for label, conf in results["top3_topics"]:
        print(f"     {label:<35} {conf*100:.1f}%")

    print(f"\n💬 Sentiments (threshold > 0.3)")
    if results["top3_sentiments"]:
        for label, prob in results["top3_sentiments"]:
            print(f"     {label:<20} {prob:.3f}")
    else:
        print(f"     None above threshold")

    action_text = results['action']
    reason_text = results['reason']
    for prefix in ["I should ", "i should "]:
        if action_text.lower().startswith(prefix.lower()):
            action_text = action_text[len(prefix):]
    for prefix in ["Because ", "because "]:
        if reason_text.lower().startswith(prefix.lower()):
            reason_text = reason_text[len(prefix):]

    print(f"\n✅ Action")
    print(f"   I should {action_text}")

    print(f"\n💡 Reason")
    print(f"   Because {reason_text}")

    print(f"\n{'='*55}\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdVox V2 Inference")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.config):
        print(f"ERROR: config not found at {args.config}")
        print("Run from V2_AdVox_QFormer/ directory.")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"ERROR: image not found at {args.image}")
        sys.exit(1)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load label maps
    ann_dir       = config["paths"]["annotations_dir"]
    topics_map, _ = load_topics_map(ann_dir)
    sents_map,  _ = load_sentiments_map(ann_dir)

    # Load model
    print("Building model...")
    model = AdVox(config).to(device)

    ckpt_path = os.path.join(
        config["paths"]["checkpoints_dir"], "advox_v2_best.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("Train the model first using train.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}\n")

    # Run inference
    results = run_inference(
        model, args.image, config, device, topics_map, sents_map
    )
    print_results(results, args.image)