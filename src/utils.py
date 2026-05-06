# utils.py
# AdVox V2 — Reusable helper functions.
# Imported by dataset.py, train.py, evaluate.py — never run directly.

import re
import os
import yaml
import numpy as np
from torchvision import transforms


# ── Config Loader ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load config.yaml and return as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Label Map Loaders ──────────────────────────────────────────────────────────

def load_topics_map(annotations_dir: str):
    """
    Parse Topics_List.txt into two lookup dicts.

    File is UTF-16 LE encoded.
    Format per line:
        1   "Restaurants, cafe, fast food" (ABBREVIATION: "restaurant")

    Returns:
        id_to_label : { 0: 'Restaurants, cafe, fast food', ... }  (0-indexed)
        text_to_id  : { 'restaurants, cafe, fast food': 0, ... }  (0-indexed)
    """
    path = os.path.join(annotations_dir, "Topics_List.txt")
    id_to_label = {}
    text_to_id  = {}

    with open(path, encoding="utf-16") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'(\d+)\s+"([^"]+)"', line)
            if m:
                idx   = int(m.group(1)) - 1        # 1-indexed → 0-indexed
                label = m.group(2).strip()
                id_to_label[idx]          = label
                text_to_id[label.lower()] = idx

    return id_to_label, text_to_id


def load_sentiments_map(annotations_dir: str):
    """
    Parse Sentiments_List.txt into two lookup dicts.

    File is latin-1 encoded.
    Format per line:
        1. "Active\xa0(energetic, adventurous...)" (ABBREVIATION: "active")

    Returns:
        id_to_label : { 0: 'Active', ... }  (0-indexed)
        text_to_id  : { 'active': 0, ... }  (0-indexed)
    """
    path = os.path.join(annotations_dir, "Sentiments_List.txt")
    id_to_label = {}
    text_to_id  = {}

    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'(\d+)\.\s+"([^\xa0"(]+)', line)
            if m:
                idx   = int(m.group(1)) - 1
                label = m.group(2).strip()
                id_to_label[idx]          = label
                text_to_id[label.lower()] = idx

    return id_to_label, text_to_id


# ── Soft Label Helpers ─────────────────────────────────────────────────────────

def compute_soft_label(votes: list, num_classes: int,
                       text_to_id: dict = None) -> np.ndarray:
    """
    Convert a list of annotator votes into a soft probability vector.

    Handles mixed format — some annotators write numeric IDs ('28'),
    others write text labels ('Girl Scouts'). Both are handled.

    Used for: Topic head (38 classes, single-label per annotator)

    Args:
        votes       : list of vote strings e.g. ['28', 'Girl Scouts', '28']
        num_classes : total number of classes (38 for topics)
        text_to_id  : dict mapping lowercase label text → 0-indexed ID

    Returns:
        np.ndarray of shape (num_classes,) summing to 1.0
    """
    vector = np.zeros(num_classes, dtype=np.float32)

    if not votes:
        return vector

    for v in votes:
        v = str(v).strip()
        if v.isdigit():
            idx = int(v) - 1       # 1-indexed in file → 0-indexed
        else:
            if text_to_id is None:
                continue
            idx = text_to_id.get(v.lower(), -1)

        if 0 <= idx < num_classes:
            vector[idx] += 1.0

    total = vector.sum()
    if total > 0:
        vector /= total

    return vector


def compute_soft_multilabel(annotations: list, num_classes: int,
                             text_to_id: dict = None) -> np.ndarray:
    """
    Convert multi-label annotator responses into a soft fractional vector.

    Handles mixed format (numeric IDs or text labels).

    Used for: Sentiment head (30 classes, multi-label per annotator)

    Args:
        annotations : list of lists e.g. [['12'], ['8', '11'], ['Calm']]
        num_classes : total number of classes (30 for sentiments)
        text_to_id  : dict mapping lowercase label text → 0-indexed ID

    Returns:
        np.ndarray of shape (num_classes,) with values in [0.0, 1.0]
    """
    vector = np.zeros(num_classes, dtype=np.float32)

    if not annotations:
        return vector

    n_annotators = len(annotations)

    for annotator_picks in annotations:
        for v in annotator_picks:
            v = str(v).strip()
            if v.isdigit():
                idx = int(v) - 1
            else:
                if text_to_id is None:
                    continue
                idx = text_to_id.get(v.lower(), -1)

            if 0 <= idx < num_classes:
                vector[idx] += 1.0

    vector /= n_annotators

    return vector


# ── Image Transforms ───────────────────────────────────────────────────────────

def get_transforms(image_size: int = 224, mode: str = "train") -> transforms.Compose:
    """
    Return torchvision transforms for ViT-B/16 CLIP input.

    CLIP-pretrained ViT uses different normalization stats than ImageNet.
    mean = [0.48145466, 0.4578275,  0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]

    Args:
        image_size : target size (default 224)
        mode       : 'train' applies augmentation, 'val' does not

    Returns:
        torchvision.transforms.Compose object
    """
    # CLIP normalization stats — important: V1 used ImageNet stats
    clip_mean = [0.48145466, 0.4578275,  0.40821073]
    clip_std  = [0.26862954, 0.26130258, 0.27577711]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])


# ── Quick Sanity Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test Topics map
    id_to_label, topics_text_to_id = load_topics_map("data/annotations")
    print(f"Topics map loaded   — {len(topics_text_to_id)} entries")
    print(f"  topic 0  : {id_to_label.get(0)}")
    print(f"  topic 27 : {id_to_label.get(27)}")

    # Test mixed format soft label
    votes  = ['28', 'Girl Scouts', '28']
    result = compute_soft_label(votes, num_classes=38,
                                text_to_id=topics_text_to_id)
    print(f"\nsoft_label sum    : {result.sum():.4f}")           # 1.0
    print(f"non-zero indices  : {list(np.where(result > 0)[0])}")

    # Test Sentiments map
    id_to_label2, sent_text_to_id = load_sentiments_map("data/annotations")
    print(f"\nSentiments map loaded — {len(sent_text_to_id)} entries")
    print(f"  sentiment 0 : {id_to_label2.get(0)}")
    print(f"  sentiment 1 : {id_to_label2.get(1)}")

    # Test soft multilabel
    annotations = [['12'], ['12'], ['8', '11', '15', '21']]
    result2     = compute_soft_multilabel(annotations, num_classes=30,
                                          text_to_id=sent_text_to_id)
    print(f"\nsoft_multilabel class 11 : {result2[11]:.4f}")     # 0.3333
    print(f"soft_multilabel class 12 : {result2[12]:.4f}")      # 0.6667 (index 11 = label 12)

    # Test transforms — check CLIP stats are applied
    train_tf = get_transforms(224, mode="train")
    val_tf   = get_transforms(224, mode="val")
    print(f"\ntrain transforms : {len(train_tf.transforms)} steps")   # 5
    print(f"val transforms   : {len(val_tf.transforms)} steps")       # 3

    print("\nutils.py sanity check passed!")