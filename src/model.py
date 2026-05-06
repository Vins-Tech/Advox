# model.py
# AdVox V2 — Q-Former + Flan-T5 Architecture
#
# Architecture summary:
#   Image (3, 224, 224)
#     └─ ViT-B/16 CLIP encoder          [always frozen]
#         ├─ CLS token (768)
#         │   ├─ TopicHead      → (38,)  soft label
#         │   └─ SentimentHead  → (30,)  soft multi-hot
#         └─ Patch tokens (196, 768)
#             └─ QFormer                [always trainable]
#                 └─ 32 query tokens (32, 768)
#                     └─ LinearProjection (768 → 768)  [always trainable]
#                         └─ [task_token] + [projected queries] + [prompt tokens]
#                             └─ Flan-T5-base           [always frozen]
#                                 └─ Action text / Reason text

import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BertConfig,
    BertModel,
)
import timm
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Q-Former ───────────────────────────────────────────────────────────────────

class QFormer(nn.Module):
    """
    Lightweight Q-Former bridge between ViT patch tokens and Flan-T5.

    Architecture:
        - num_query_tokens learned query vectors (randomly initialized,
          then updated via cross-attention with image patch tokens)
        - Uses a BERT-base style transformer initialized from pretrained
          BERT-base weights for stable training start
        - Cross-attention layers attend to ViT patch tokens
        - Self-attention layers allow queries to communicate with each other

    Why BERT init?
        BERT and Q-Former share the same transformer architecture.
        Starting from pretrained BERT weights gives the Q-Former a head
        start on learning meaningful attention patterns instead of
        training from random noise.

    Args:
        hidden_dim   : transformer hidden dimension (768, matches BERT-base)
        num_heads    : attention heads (12, matches BERT-base)
        num_layers   : number of transformer layers (6 — half of BERT-base)
        ffn_dim      : feedforward dimension (3072, matches BERT-base)
        num_queries  : number of learnable query tokens (32)
        encoder_dim  : ViT patch token dimension (768)
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int,
                 ffn_dim: int, num_queries: int, encoder_dim: int):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim  = hidden_dim

        # Learnable query tokens — shape (1, num_queries, hidden_dim)
        # Expanded to (batch, num_queries, hidden_dim) during forward
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, hidden_dim)
        )
        nn.init.normal_(self.query_tokens, std=0.02)

        # Project ViT patch tokens into Q-Former hidden space if dims differ
        # Here both are 768 so this is identity-like, but kept for correctness
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)

        # Build transformer layers
        # Each layer has: self-attention + cross-attention + FFN
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize from BERT-base pretrained weights
        self._init_from_bert(num_layers, hidden_dim, num_heads, ffn_dim)

    def _init_from_bert(self, num_layers: int, hidden_dim: int,
                        num_heads: int, ffn_dim: int):
        """
        Load BERT-base pretrained weights into Q-Former layers.

        BERT-base has 12 layers — we use the first num_layers of them.
        This gives Q-Former a pretrained initialization instead of
        starting from random weights.
        """
        print("  Initializing Q-Former from BERT-base pretrained weights...")
        bert_config = BertConfig(
            hidden_size          = hidden_dim,
            num_attention_heads  = num_heads,
            num_hidden_layers    = num_layers,
            intermediate_size    = ffn_dim,
        )
        bert = BertModel.from_pretrained(
            "bert-base-uncased",
            config = bert_config,
            ignore_mismatched_sizes = True,
        )

        # Copy self-attention + FFN weights layer by layer
        #
        # NOTE: nn.MultiheadAttention stores Q/K/V as a single fused
        # in_proj_weight of shape (3*hidden, hidden) and in_proj_bias
        # of shape (3*hidden,). We must pack BERT's separate Q/K/V
        # weight matrices into this fused format manually.
        for i, layer in enumerate(self.layers):
            bert_layer = bert.encoder.layer[i]
            H = hidden_dim   # 768

            # ── Self-attention ──────────────────────────────
            # Pack Q, K, V into fused in_proj_weight (3H, H)
            q_w = bert_layer.attention.self.query.weight.data   # (H, H)
            k_w = bert_layer.attention.self.key.weight.data     # (H, H)
            v_w = bert_layer.attention.self.value.weight.data   # (H, H)
            layer.self_attn.in_proj_weight.data = torch.cat([q_w, k_w, v_w], dim=0)  # (3H, H)

            # Pack Q, K, V biases into fused in_proj_bias (3H,)
            q_b = bert_layer.attention.self.query.bias.data     # (H,)
            k_b = bert_layer.attention.self.key.bias.data       # (H,)
            v_b = bert_layer.attention.self.value.bias.data     # (H,)
            layer.self_attn.in_proj_bias.data = torch.cat([q_b, k_b, v_b], dim=0)    # (3H,)

            # Output projection
            layer.self_attn.out_proj.weight.data = bert_layer.attention.output.dense.weight.data.clone()
            layer.self_attn.out_proj.bias.data   = bert_layer.attention.output.dense.bias.data.clone()

            # ── FFN ────────────────────────────────────────
            layer.ffn[1].weight.data = bert_layer.intermediate.dense.weight.data.clone()
            layer.ffn[1].bias.data   = bert_layer.intermediate.dense.bias.data.clone()
            layer.ffn[3].weight.data = bert_layer.output.dense.weight.data.clone()
            layer.ffn[3].bias.data   = bert_layer.output.dense.bias.data.clone()

            # ── LayerNorms ─────────────────────────────────
            layer.norm1.weight.data = bert_layer.attention.output.LayerNorm.weight.data.clone()
            layer.norm1.bias.data   = bert_layer.attention.output.LayerNorm.bias.data.clone()
            layer.norm3.weight.data = bert_layer.output.LayerNorm.weight.data.clone()
            layer.norm3.bias.data   = bert_layer.output.LayerNorm.bias.data.clone()

        del bert   # free memory after copying weights
        print("  Q-Former BERT init complete.")

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens : (B, 196, 768) — ViT patch tokens (no CLS)

        Returns:
            query_output : (B, num_queries, hidden_dim) — attended query tokens
        """
        B = patch_tokens.size(0)

        # Project patch tokens into Q-Former space
        kv = self.encoder_proj(patch_tokens)           # (B, 196, 768)

        # Expand query tokens to batch size
        queries = self.query_tokens.expand(B, -1, -1)  # (B, 32, 768)

        # Pass through Q-Former layers
        # Each layer: queries self-attend + cross-attend to patch tokens
        for layer in self.layers:
            queries = layer(queries, kv)

        queries = self.norm(queries)                   # (B, 32, 768)
        return queries


class QFormerLayer(nn.Module):
    """
    Single Q-Former transformer layer.

    Order of operations:
        1. Self-attention  (queries attend to each other)
        2. Cross-attention (queries attend to image patch tokens)
        3. FFN

    Each sub-layer has residual connection + LayerNorm (Pre-LN style).
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention — queries attend to ViT patch tokens
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN: Linear → GELU → Linear
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor,
                patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries      : (B, num_queries, hidden_dim)
            patch_tokens : (B, 196, hidden_dim)

        Returns:
            queries      : (B, num_queries, hidden_dim)
        """
        # 1. Self-attention (queries ↔ queries)
        residual = queries
        queries  = self.norm1(queries)
        queries, _ = self.self_attn(queries, queries, queries)
        queries  = queries + residual

        # 2. Cross-attention (queries ↔ patch tokens)
        residual = queries
        queries  = self.norm2(queries)
        queries, _ = self.cross_attn(queries, patch_tokens, patch_tokens)
        queries  = queries + residual

        # 3. FFN
        residual = queries
        queries  = self.ffn(queries) + residual

        return queries


# ── Classification Heads ───────────────────────────────────────────────────────

class TopicHead(nn.Module):
    """
    Topic classification head.
    Input: CLS token (768,) from ViT encoder directly.
    Output: log-softmax over 38 topic classes.
    Loss: KL Divergence against soft label targets.
    """

    def __init__(self, encoder_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token : (B, 768)
        Returns:
            log_probs : (B, 38) — log-softmax output for KL loss
        """
        return torch.log_softmax(self.head(cls_token), dim=-1)


class SentimentHead(nn.Module):
    """
    Sentiment classification head.
    Input: CLS token (768,) from ViT encoder directly.
    Output: raw logits over 30 sentiment classes (no sigmoid here).
    Sigmoid applied inside loss during training, manually during inference.
    Loss: Binary Cross-Entropy with masking for sf4-sf9 images.
    """

    def __init__(self, encoder_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token : (B, 768)
        Returns:
            logits    : (B, 30) — raw logits, NO sigmoid applied
        """
        return self.head(cls_token)


# ── Main AdVox Model ───────────────────────────────────────────────────────────

class AdVox(nn.Module):
    """
    AdVox V2 — Multi-task vision-language model for advertisement understanding.

    Components:
        ViT-B/16 CLIP    : shared visual encoder
        TopicHead        : 38-class soft label classification
        SentimentHead    : 30-class soft multi-hot classification
        QFormer          : visual-language bridge (BERT-base initialized)
        LinearProjection : maps Q-Former output → T5 embedding space
        task_tokens      : 2 learned vectors (action / reason)
        Flan-T5-base     : frozen seq2seq LLM for text generation

    Forward pass:
        1. ViT encodes image → CLS token + patch tokens
        2. CLS token → TopicHead, SentimentHead
        3. Patch tokens → QFormer → 32 query vectors
        4. Query vectors → LinearProjection → T5 embedding space
        5. [task_token] + [projected queries] + [prompt tokens] → T5
        6. T5 generates action / reason text
    """

    # T5 prompt strings — prepended to encoder input for each task
    ACTION_PROMPT = "What action should I take after seeing this advertisement? I should"
    REASON_PROMPT = "Why should I take that action after seeing this advertisement? Because"

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]

        encoder_dim  = cfg["encoder_dim"]           # 768
        num_queries  = cfg["num_query_tokens"]       # 32
        proj_dim     = cfg["projection_dim"]         # 768
        t5_name      = cfg["text_decoder"]           # google/flan-t5-base

        # ── 1. ViT-B/16 CLIP encoder ──────────────────────
        print("Loading ViT-B/16 CLIP encoder...")
        self.encoder = timm.create_model(
            cfg["encoder"],
            pretrained   = True,
            num_classes  = 0,        # remove classification head
            global_pool  = "",       # return all tokens, not just pooled
        )
        # encoder output shape: (B, 197, 768)
        # index 0 = CLS token, index 1: = patch tokens

        # ── 2. Classification heads ────────────────────────
        self.topic_head     = TopicHead(encoder_dim, cfg["num_topic_classes"])
        self.sentiment_head = SentimentHead(encoder_dim, cfg["num_sentiment_classes"])

        # ── 3. Q-Former ───────────────────────────────────
        print("Building Q-Former...")
        self.qformer = QFormer(
            hidden_dim  = cfg["qformer_hidden_dim"],   # 768
            num_heads   = cfg["qformer_num_heads"],    # 12
            num_layers  = cfg["qformer_num_layers"],   # 6
            ffn_dim     = cfg["qformer_ffn_dim"],      # 3072
            num_queries = num_queries,                 # 32
            encoder_dim = encoder_dim,                 # 768
        )

        # ── 4. Linear projection ───────────────────────────
        # Translates Q-Former output (BERT space) → T5 embedding space
        self.projection = nn.Linear(encoder_dim, proj_dim)

        # ── 5. Task tokens ─────────────────────────────────
        # Two learned vectors — one for action, one for reason
        # Shape: (1, 1, t5_embed_dim) — prepended to T5 input sequence
        self.task_token_action = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.task_token_reason = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.normal_(self.task_token_action, std=0.02)
        nn.init.normal_(self.task_token_reason, std=0.02)

        # ── 6. Flan-T5-base ────────────────────────────────
        print(f"Loading {t5_name}...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)

        # Freeze T5 completely — never train LLM weights
        for param in self.t5.parameters():
            param.requires_grad = False
        print("  Flan-T5 frozen.")

        # Cache T5 embedding dimension
        self.t5_embed_dim = self.t5.config.d_model  # 768 for flan-t5-base

        # Verify projection dim matches T5 embed dim
        assert proj_dim == self.t5_embed_dim, \
            f"projection_dim ({proj_dim}) must equal T5 d_model ({self.t5_embed_dim})"

        # Pre-tokenize prompts and cache on CPU
        # Moved to device during forward pass
        self._cache_prompts()

        print("AdVox V2 model ready.")

    def _cache_prompts(self):
        """Tokenize action and reason prompts once and cache."""
        def tokenize_prompt(text):
            return self.tokenizer(
                text,
                return_tensors = "pt",
                add_special_tokens = True,
            ).input_ids  # (1, prompt_len)

        self.register_buffer(
            "action_prompt_ids",
            tokenize_prompt(self.ACTION_PROMPT)
        )
        self.register_buffer(
            "reason_prompt_ids",
            tokenize_prompt(self.REASON_PROMPT)
        )

    def _get_prompt_embeds(self, prompt_ids: torch.Tensor,
                           batch_size: int) -> torch.Tensor:
        """
        Convert cached prompt token ids → T5 embeddings, expanded to batch.

        Args:
            prompt_ids : (1, prompt_len) — cached prompt token ids
            batch_size : B

        Returns:
            embeds : (B, prompt_len, t5_embed_dim)
        """
        embeds = self.t5.encoder.embed_tokens(prompt_ids)  # (1, prompt_len, 768)
        return embeds.expand(batch_size, -1, -1)            # (B, prompt_len, 768)

    def _build_t5_input(self, projected: torch.Tensor,
                        task: str) -> torch.Tensor:
        """
        Build full T5 encoder input sequence.

        Order: [task_token] + [projected Q-Former tokens] + [prompt tokens]

        Args:
            projected : (B, num_queries, 768) — projected Q-Former output
            task      : 'action' or 'reason'

        Returns:
            inputs_embeds : (B, 1 + num_queries + prompt_len, 768)
        """
        B = projected.size(0)

        # Task token — expand to batch
        if task == "action":
            task_tok = self.task_token_action.expand(B, -1, -1)  # (B, 1, 768)
            prompt_e = self._get_prompt_embeds(self.action_prompt_ids, B)
        else:
            task_tok = self.task_token_reason.expand(B, -1, -1)  # (B, 1, 768)
            prompt_e = self._get_prompt_embeds(self.reason_prompt_ids, B)

        # Concatenate: [task_token | Q-Former tokens | prompt]
        inputs_embeds = torch.cat([task_tok, projected, prompt_e], dim=1)
        # Shape: (B, 1 + 32 + prompt_len, 768)

        return inputs_embeds

    def forward(self, images: torch.Tensor,
                action_labels: torch.Tensor = None,
                reason_labels: torch.Tensor = None) -> dict:
        """
        Full forward pass.

        Args:
            images        : (B, 3, 224, 224)
            action_labels : (B, max_len) — T5 labels with -100 on padding, or None
            reason_labels : (B, max_len) — T5 labels with -100 on padding, or None

        Returns dict with keys:
            topic_log_probs  : (B, 38)   — for KL loss
            sentiment_logits : (B, 30)   — for BCE loss (no sigmoid)
            action_loss      : scalar    — T5 cross-entropy loss for action
            reason_loss      : scalar    — T5 cross-entropy loss for reason
            action_loss and reason_loss are None if labels not provided (inference)
        """

        # ── Step 1: ViT encoder ────────────────────────────
        tokens = self.encoder(images)          # (B, 197, 768)
        cls_token    = tokens[:, 0, :]         # (B, 768) — global image repr
        patch_tokens = tokens[:, 1:, :]        # (B, 196, 768) — spatial tokens

        # ── Step 2: Classification heads (from CLS token) ──
        topic_log_probs  = self.topic_head(cls_token)      # (B, 38)
        sentiment_logits = self.sentiment_head(cls_token)  # (B, 30)

        # ── Step 3: Q-Former (from patch tokens) ───────────
        query_output = self.qformer(patch_tokens)          # (B, 32, 768)

        # ── Step 4: Linear projection ──────────────────────
        projected = self.projection(query_output)          # (B, 32, 768)

        # ── Step 5 + 6: T5 for Action and Reason ───────────
        # IMPORTANT: T5 must run in float32.
        # AMP float16 causes NaN in cross-entropy due to logit overflow
        # across T5's large vocabulary (32,128 tokens).
        # We disable autocast and cast inputs to float32 explicitly.
        action_loss = None
        reason_loss = None

        with torch.amp.autocast("cuda", enabled=False):
            if action_labels is not None:
                action_embeds = self._build_t5_input(projected, task="action")
                action_out    = self.t5(
                    inputs_embeds = action_embeds.float(),
                    labels        = action_labels,
                )
                action_loss = action_out.loss

            if reason_labels is not None:
                reason_embeds = self._build_t5_input(projected, task="reason")
                reason_out    = self.t5(
                    inputs_embeds = reason_embeds.float(),
                    labels        = reason_labels,
                )
                reason_loss = reason_out.loss

        return {
            "topic_log_probs":  topic_log_probs,
            "sentiment_logits": sentiment_logits,
            "action_loss":      action_loss,
            "reason_loss":      reason_loss,
        }

    def generate_text(self, images: torch.Tensor, task: str,
                      max_new_tokens: int = 40, num_beams: int = 4,
                      min_new_tokens: int = 5) -> list:
        """
        Generate action or reason text for a batch of images.
        Used during evaluation and inference — no labels needed.

        Args:
            images        : (B, 3, 224, 224)
            task          : 'action' or 'reason'
            max_new_tokens: max tokens to generate
            num_beams     : beam search width
            min_new_tokens: minimum tokens to generate

        Returns:
            texts : list of B decoded strings
        """
        with torch.no_grad():
            tokens       = self.encoder(images)
            patch_tokens = tokens[:, 1:, :]

            query_output  = self.qformer(patch_tokens)
            projected     = self.projection(query_output)
            inputs_embeds = self._build_t5_input(projected, task=task)

            generated = self.t5.generate(
                inputs_embeds  = inputs_embeds,
                max_new_tokens = max_new_tokens,
                min_new_tokens = min_new_tokens,
                num_beams      = num_beams,
                early_stopping = True,
            )

        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts

    def freeze_encoder(self):
        """Freeze ViT encoder — called at training start."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("ViT encoder frozen.")

    def count_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ── Sanity Check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils import load_config

    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Build model
    model = AdVox(config).to(device)

    # Freeze encoder (as in epoch 1-2)
    model.freeze_encoder()

    print(f"\nTotal params     : {model.count_total_params():,}")
    print(f"Trainable params : {model.count_trainable_params():,}")

    # Dummy batch
    B      = 2
    images = torch.randn(B, 3, 224, 224).to(device)
    a_lbl  = torch.randint(0, 100, (B, 64)).to(device)
    r_lbl  = torch.randint(0, 100, (B, 64)).to(device)

    # Set some positions to -100 (padding mask)
    a_lbl[:, 40:] = -100
    r_lbl[:, 40:] = -100

    print("\nRunning forward pass...")
    out = model(images, action_labels=a_lbl, reason_labels=r_lbl)

    print(f"  topic_log_probs  : {out['topic_log_probs'].shape}")    # (2, 38)
    print(f"  sentiment_logits : {out['sentiment_logits'].shape}")   # (2, 30)
    print(f"  action_loss      : {out['action_loss'].item():.4f}")
    print(f"  reason_loss      : {out['reason_loss'].item():.4f}")

    # Test generation
    print("\nTesting text generation (action)...")
    texts = model.generate_text(images, task="action", max_new_tokens=20)
    for i, t in enumerate(texts):
        print(f"  sample {i}: {t}")

    print("\nTesting text generation (reason)...")
    texts = model.generate_text(images, task="reason", max_new_tokens=20)
    for i, t in enumerate(texts):
        print(f"  sample {i}: {t}")

    print("\nmodel.py sanity check passed!")