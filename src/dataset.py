# dataset.py
# AdVox V2 — PyTorch Dataset for Q-Former + Flan-T5 architecture.
# Builds annotator-expanded rows from annotation JSONs.
# One sample = one annotator's complete annotation for one image.
#
# Key changes from V1:
#   - GPT2Tokenizer replaced with T5Tokenizer (Flan-T5)
#   - Token encoding uses T5 format (no pad_token hack needed)
#   - Labels for T5 use -100 masking on padding (standard seq2seq practice)

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import T5Tokenizer
from utils import (load_config, compute_soft_label, compute_soft_multilabel,
                   get_transforms, load_topics_map, load_sentiments_map)


# ── Annotation Loader ──────────────────────────────────────────────────────────

def load_annotations(annotations_dir: str) -> dict:
    """Load the 4 core annotation JSON files into a single dict."""

    def load_json(filename):
        path = os.path.join(annotations_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "topics":     load_json("Topics.json"),
        "sentiments": load_json("Sentiments.json"),
        "qa_action":  load_json("QA_Action.json"),
        "qa_reason":  load_json("QA_Reason.json"),
    }


# ── Dataset Class ──────────────────────────────────────────────────────────────

class AdVoxDataset(Dataset):
    """
    Annotator-expanded PyTorch Dataset for AdVox V2.

    One sample = one annotator's complete annotation for one image.
    Only images physically present on disk are included.
    Works on sf0 only now — grows automatically as more subfolders are added.

    T5 label encoding:
        - input_ids  : tokenized text, padded to max_text_length
        - labels     : same as input_ids but padding positions replaced with -100
                       T5 ignores -100 positions in cross-entropy loss
    """

    def __init__(self, config: dict, mode: str = "train"):
        self.config    = config
        self.mode      = mode
        self.data_root = config["paths"]["data_root"]
        self.ann_dir   = config["paths"]["annotations_dir"]
        self.img_size  = config["dataset"]["image_size"]
        self.max_len   = config["model"]["max_text_length"]

        # Load annotations
        print("Loading annotations...")
        ann             = load_annotations(self.ann_dir)
        self.topics     = ann["topics"]
        self.sentiments = ann["sentiments"]
        self.qa_action  = ann["qa_action"]
        self.qa_reason  = ann["qa_reason"]

        # Load label maps
        print("Loading label maps...")
        _, self.topics_text_to_id     = load_topics_map(self.ann_dir)
        _, self.sentiments_text_to_id = load_sentiments_map(self.ann_dir)

        # T5 Tokenizer
        # T5Tokenizer handles padding natively — no pad_token hack needed
        print("Loading T5 tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        # Image transforms
        self.transforms = get_transforms(self.img_size, mode=mode)

        # Build sample index
        print("Building sample index...")
        self.samples = self._build_index()
        print(f"Dataset ready — {len(self.samples)} samples ({mode})")

    def _build_index(self) -> list:
        """
        Build flat annotator-expanded sample list.
        Skips images not yet downloaded to disk.
        Action and reason from the SAME annotator index are always kept paired.
        """
        samples   = []
        sent_keys = set(self.sentiments.keys())
        skipped   = 0

        for image_key in self.qa_action:

            # Must have matching reason + topic annotations
            if image_key not in self.qa_reason:
                continue
            if image_key not in self.topics:
                continue

            # Skip images not yet on disk
            img_path = os.path.join(self.data_root, image_key)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            action_list = self.qa_action[image_key]
            reason_list = self.qa_reason[image_key]

            # Pair by annotator index — never mix across annotators
            n = min(len(action_list), len(reason_list))

            # Soft labels — shared across all annotators of this image
            topic_soft = compute_soft_label(
                self.topics[image_key],
                num_classes=38,
                text_to_id=self.topics_text_to_id
            )

            has_sentiment = image_key in sent_keys
            if has_sentiment:
                sent_soft = compute_soft_multilabel(
                    self.sentiments[image_key],
                    num_classes=30,
                    text_to_id=self.sentiments_text_to_id
                )
            else:
                sent_soft = None

            for i in range(n):
                samples.append({
                    "image_key":     image_key,
                    "topic_soft":    topic_soft,
                    "sent_soft":     sent_soft,
                    "has_sentiment": has_sentiment,
                    "action":        action_list[i],
                    "reason":        reason_list[i],
                })

        print(f"  Skipped {skipped} images not on disk")
        return samples

    def _tokenize(self, text: str) -> tuple:
        """
        Tokenize a text string for T5 seq2seq training.

        Returns:
            input_ids : (max_len,) — tokenized + padded
            labels    : (max_len,) — same but -100 on padding positions
                        T5 loss ignores -100 positions automatically
        """
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)       # (max_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (max_len,)

        # Replace padding token ids with -100 for loss masking
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return input_ids, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Image ──────────────────────────────────────────
        img_path = os.path.join(self.data_root, sample["image_key"])
        image    = Image.open(img_path).convert("RGB")
        image    = self.transforms(image)                  # (3, 224, 224)

        # ── Topic soft label ───────────────────────────────
        topic_soft = torch.tensor(sample["topic_soft"], dtype=torch.float32)  # (38,)

        # ── Sentiment soft label + mask ────────────────────
        if sample["has_sentiment"]:
            sent_soft = torch.tensor(sample["sent_soft"], dtype=torch.float32)
            sent_mask = torch.tensor(True)
        else:
            sent_soft = torch.zeros(30, dtype=torch.float32)
            sent_mask = torch.tensor(False)

        # ── Action tokens ──────────────────────────────────
        action_ids, action_labels = self._tokenize(sample["action"])

        # ── Reason tokens ──────────────────────────────────
        reason_ids, reason_labels = self._tokenize(sample["reason"])

        return {
            "image":          image,           # (3, 224, 224)
            "topic_soft":     topic_soft,      # (38,)
            "sent_soft":      sent_soft,        # (30,)
            "sent_mask":      sent_mask,        # bool scalar
            "action_ids":     action_ids,       # (64,)  — T5 input tokens
            "action_labels":  action_labels,    # (64,)  — T5 labels (-100 on pad)
            "reason_ids":     reason_ids,       # (64,)  — T5 input tokens
            "reason_labels":  reason_labels,    # (64,)  — T5 labels (-100 on pad)
        }


# ── DataLoader Factory ─────────────────────────────────────────────────────────

def get_dataloaders(config: dict):
    """
    Build train and val DataLoaders with 85/15 split.
    Split is deterministic — same seed every run.
    """
    full_dataset = AdVoxDataset(config, mode="train")

    total      = len(full_dataset)
    train_size = int(total * config["dataset"]["train_split"])
    val_size   = total - train_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Set val mode transforms on the underlying dataset
    val_ds.dataset.mode = "val"

    train_loader = DataLoader(
        train_ds,
        batch_size  = config["training"]["batch_size"],
        shuffle     = True,
        num_workers = config["dataset"]["num_workers"],
        persistent_workers = True,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config["training"]["batch_size"],
        shuffle     = False,
        num_workers = config["dataset"]["num_workers"],
        persistent_workers = True,
        pin_memory  = True,
    )

    return train_loader, val_loader


# ── Sanity Check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("configs/config.yaml")

    dataset = AdVoxDataset(config, mode="train")

    print(f"\nTotal samples : {len(dataset)}")
    print("Fetching sample 0...")
    sample = dataset[0]

    print(f"  image shape     : {sample['image'].shape}")           # (3, 224, 224)
    print(f"  topic_soft      : sum={sample['topic_soft'].sum():.4f}, shape={sample['topic_soft'].shape}")
    print(f"  sent_soft       : sum={sample['sent_soft'].sum():.4f}, shape={sample['sent_soft'].shape}")
    print(f"  sent_mask       : {sample['sent_mask']}")
    print(f"  action_ids      : {sample['action_ids'].shape}")       # (64,)
    print(f"  action_labels   : {sample['action_labels'].shape}")    # (64,)
    print(f"  reason_ids      : {sample['reason_ids'].shape}")       # (64,)
    print(f"  reason_labels   : {sample['reason_labels'].shape}")    # (64,)

    # Verify -100 masking is working
    n_masked_action = (sample['action_labels'] == -100).sum().item()
    n_masked_reason = (sample['reason_labels'] == -100).sum().item()
    print(f"\n  action_labels masked positions : {n_masked_action}")
    print(f"  reason_labels masked positions : {n_masked_reason}")

    print("\ndataset.py sanity check passed!")