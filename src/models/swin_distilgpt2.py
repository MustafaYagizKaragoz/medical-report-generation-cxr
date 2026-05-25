"""
swin_distilgpt2.py
==================
Swin Transformer Base (microsoft/swin-base-patch4-window7-224)
+ DistilGPT-2 — Multimodal MTL

Mimari Özeti:
  1. Swin-B Encoder  : görüntü → (B, 49, 1024) spatial token dizisi
                       (stage-4 çıktısı, 7×7 grid)
  2. Projeksiyon     : Linear(1024→2048) → GELU → Linear(2048→768)
                       Swin-B (1024-dim) → DistilGPT-2 (768-dim) uzayına hizalama
  3. Visual Prefix   : 49 Swin token'ı DistilGPT-2 embedding uzayına enjekte edilir
                       → inputs_embeds = concat([visual_prefix, text_embeds], dim=1)
  4. DistilGPT-2    : GPT-2 tabanlı AR text decoder (prefix → rapor tokenları)
  5. MTL Head       : Swin pooler_output (1024-dim) → Linear(1024, 14) → patoloji

Loss:
  Total = 1.0 × CrossEntropyLoss (generation) + 0.5 × BCEWithLogitsLoss (classification)

ViT ile Farklar:
  - ViT CLS token yoktur; sınıflandırma için `swin_out.pooler_output` kullanılır.
  - Spatial token sayısı: 197 (ViT) → 49 (Swin-B stage-4, 7×7 grid)
  - Hidden dim: 768 (ViT) → 1024 (Swin-B)

VRAM Optim:
  - Gradient Checkpointing (Swin + DistilGPT-2)
  - Mixed Precision (AMP) ile train_swin_distilgpt2.py içinde kullanılır
"""

import os
import torch
import torch.nn as nn
from transformers import SwinModel, GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Dict

# ── Offline-first yardımcısı ─────────────────────────────────────────────
def _load_pretrained_offline_first(cls, model_name: str, **kwargs):
    """
    Önce local Hugging Face cache'inden yükler (ağ erişimi olmadan).
    Cache yoksa veya bozuksa normal indir.
    """
    try:
        return cls.from_pretrained(model_name, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(model_name, **kwargs)


# =========================================================================
# SABİTLER
# =========================================================================
NUM_CLASSES = 14

PATHOLOGY_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]

# Swin-B stage-4 çıktısı: 7×7 = 49 spatial token
NUM_PREFIX_TOKENS = 49

# Swin-B son katman gizli boyutu
SWIN_HIDDEN_DIM = 1024

# DistilGPT-2 gömme boyutu
GPT_HIDDEN_DIM = 768


# =========================================================================
# VISUAL PROJECTION BLOCK (1024 → 768)
# =========================================================================
class VisualProjectionBlock(nn.Module):
    """
    Swin-B çıktısını (B, num_tokens, swin_dim=1024)
    DistilGPT-2 embedding uzayına (B, num_tokens, gpt_dim=768) projekte eder.

    Mimari:
        Linear(1024 → 2048) → LayerNorm → GELU → Dropout
        → Linear(2048 → 768) → LayerNorm
    """

    def __init__(
        self,
        swin_dim: int = SWIN_HIDDEN_DIM,
        gpt_dim:  int = GPT_HIDDEN_DIM,
        dropout:  float = 0.1,
    ):
        super().__init__()
        mid_dim = gpt_dim * 2  # 1536
        self.proj = nn.Sequential(
            nn.Linear(swin_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, gpt_dim),
            nn.LayerNorm(gpt_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_tokens, swin_dim) → (B, num_tokens, gpt_dim)"""
        return self.proj(x)


# =========================================================================
# ANA MODEL
# =========================================================================
class SwinDistilGPT2ForMTL(nn.Module):
    """
    Swin Transformer Base + DistilGPT-2 Multimodal Multi-Task Learning Modeli.

    forward() Çıktısı:
        {
          "loss":       toplam kayıp (Tensor veya None),
          "loss_gen":   rapor üretim kaybı (Tensor veya None),
          "loss_cls":   sınıflandırma kaybı (Tensor veya None),
          "logits_lm":  dil modeli logitleri (B, T, vocab_size),
          "logits_cls": sınıflandırma logitleri (B, 14),
        }
    """

    GEN_LOSS_WEIGHT: float = 1.0
    CLS_LOSS_WEIGHT: float = 0.5

    def __init__(
        self,
        swin_model:        SwinModel,
        gpt_model:         GPT2LMHeadModel,
        swin_dim:          int   = SWIN_HIDDEN_DIM,
        gpt_dim:           int   = GPT_HIDDEN_DIM,
        num_classes:       int   = NUM_CLASSES,
        num_prefix_tokens: int   = NUM_PREFIX_TOKENS,
        proj_dropout:      float = 0.1,
    ):
        super().__init__()

        # ── Alt Modeller ──────────────────────────────────────────────────
        self.swin = swin_model       # Swin-B Encoder
        self.gpt  = gpt_model        # DistilGPT-2 Decoder

        # ── Projeksiyon Katmanı (1024 → 768) ─────────────────────────────
        self.visual_projection = VisualProjectionBlock(swin_dim, gpt_dim, proj_dropout)

        # ── MTL Sınıflandırma Kafası ──────────────────────────────────────
        # Swin pooler_output (1024-dim) üzerinden 14 patoloji tahmini
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(swin_dim),
            nn.Linear(swin_dim, num_classes),
        )

        # ── Yardımcı Özellikler ───────────────────────────────────────────
        self.num_prefix_tokens = num_prefix_tokens
        self.swin_dim          = swin_dim
        self.config            = gpt_model.config   # .generate() uyumluluğu için

        # ── Kayıp Fonksiyonları ───────────────────────────────────────────
        self._gen_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self._cls_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self._eos_id      = gpt_model.config.eos_token_id
        self._eos_weight  = 5.0   # EOS token için kayıp çarpanı

    # ──────────────────────────────────────────────────────────────────────
    # İÇ YARDIMCILAR
    # ──────────────────────────────────────────────────────────────────────

    def _encode_image(self, pixel_values: torch.Tensor):
        """
        Swin-B'yi çalıştırır ve spatial token'ları + pooled çıktıyı döndürür.

        Returns:
            pooled_embed   : (B, swin_dim)  — global avg pooled çıktı (CLS yerine)
            visual_prefix  : (B, num_prefix, gpt_dim) — projekte edilmiş vizüel önek
        """
        swin_out = self.swin(pixel_values=pixel_values)

        # last_hidden_state: (B, 49, 1024)  — stage-4 spatial tokens (7×7 grid)
        hidden_states = swin_out.last_hidden_state  # (B, 49, 1024)

        # pooler_output: (B, 1024) — global average pooled (ViT CLS muadili)
        pooled_embed = swin_out.pooler_output       # (B, 1024)

        # Prefix olarak kullanılacak spatial token sayısı
        prefix_tokens = hidden_states[:, : self.num_prefix_tokens, :]  # (B, P, 1024)
        visual_prefix = self.visual_projection(prefix_tokens)           # (B, P, 768)

        return pooled_embed, visual_prefix

    def _build_prefix_attention_mask(
        self,
        visual_prefix: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vizüel önek + metin token'ları için dikkat maskesi oluşturur.
        Tüm vizüel prefix token'ları her zaman görünür (mask = 1).

        Returns: (B, num_prefix + text_len)
        """
        B, P, _ = visual_prefix.shape
        prefix_mask = torch.ones(B, P, dtype=text_attention_mask.dtype,
                                 device=text_attention_mask.device)
        return torch.cat([prefix_mask, text_attention_mask], dim=1)

    def _compute_gen_loss(
        self,
        logits_lm: torch.Tensor,
        labels:    torch.Tensor,
        num_prefix: int,
    ) -> torch.Tensor:
        """
        EOS-boosted kaydırılmış dil modeli kaybı.

        logits_lm : (B, num_prefix + text_len, vocab_size)
        labels    : (B, text_len)  — pad konumları -100, EOS geçerli
        """
        B, text_len = labels.shape

        shift_logits = logits_lm[:, num_prefix - 1 : num_prefix + text_len - 1, :].contiguous()
        shift_labels = labels.contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        per_token_loss = self._gen_loss_fn(flat_logits, flat_labels)

        if self._eos_id is not None:
            eos_mask = (flat_labels == self._eos_id)
            per_token_loss = torch.where(
                eos_mask,
                per_token_loss * self._eos_weight,
                per_token_loss,
            )

        valid_mask = (flat_labels != -100)
        if valid_mask.any():
            loss = per_token_loss[valid_mask].mean()
        else:
            loss = per_token_loss.mean()

        return loss

    # ──────────────────────────────────────────────────────────────────────
    # FORWARD
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        labels_cls:     Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            pixel_values   : (B, C, H, W)
            input_ids      : (B, T) metin token IDs
            attention_mask : (B, T) metin dikkat maskesi
            labels         : (B, T) generation hedefi; pad = -100
            labels_cls     : (B, 14) float, 0/1 etiketler

        Returns: Dict (loss, loss_gen, loss_cls, logits_lm, logits_cls)
        """
        # ── 1. Görsel Kodlama ─────────────────────────────────────────────
        pooled_embed, visual_prefix = self._encode_image(pixel_values)
        num_prefix = visual_prefix.size(1)   # 49

        # ── 2. MTL Sınıflandırma ──────────────────────────────────────────
        logits_cls = self.classifier_head(pooled_embed)   # (B, 14)

        # ── 3. Metin Gömme + Prefix Birleştirme ───────────────────────────
        text_embeds = self.gpt.transformer.wte(input_ids)   # (B, T, 768)
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)  # (B, P+T, 768)

        if attention_mask is not None:
            full_attention_mask = self._build_prefix_attention_mask(visual_prefix, attention_mask)
        else:
            B, T = input_ids.shape
            full_attention_mask = torch.ones(B, num_prefix + T, device=pixel_values.device)

        # ── 4. DistilGPT-2 İleriye Geçiş ─────────────────────────────────
        gpt_outputs = self.gpt(
            inputs_embeds=combined_embeds,
            attention_mask=full_attention_mask,
            **kwargs,
        )
        logits_lm = gpt_outputs.logits   # (B, P+T, vocab_size)

        # ── 5. Kayıp Hesabı ───────────────────────────────────────────────
        loss_gen   = None
        loss_cls   = None
        total_loss = None

        if labels is not None:
            loss_gen = self._compute_gen_loss(logits_lm, labels, num_prefix)

        if labels_cls is not None:
            loss_cls = self._cls_loss_fn(logits_cls, labels_cls)

        if loss_gen is not None or loss_cls is not None:
            gen_part = (self.GEN_LOSS_WEIGHT * loss_gen) if loss_gen is not None else 0.0
            cls_part = (self.CLS_LOSS_WEIGHT * loss_cls) if loss_cls is not None else 0.0
            total_loss = gen_part + cls_part

        return {
            "loss":       total_loss,
            "loss_gen":   loss_gen,
            "loss_cls":   loss_cls,
            "logits_lm":  logits_lm,
            "logits_cls": logits_cls,
        }

    # ──────────────────────────────────────────────────────────────────────
    # GENERATE (HF Uyumlu)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        pixel_values:          torch.Tensor,
        tokenizer=None,
        max_new_tokens:        int   = 150,
        num_beams:             int   = 4,
        repetition_penalty:    float = 2.0,
        no_repeat_ngram_size:  int   = 3,
        temperature:           float = 1.0,
        top_p:                 float = 0.9,
        do_sample:             bool  = False,
        length_penalty:        float = 1.2,
        early_stopping:        bool  = True,
        eos_token_id=None,
        pad_token_id=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Görüntüden rapor üretir.

        BOS seed token olarak "the" (id=1169) kullanılır.
        EOS problemi hakkında detaylı açıklama: vit_distil2gpt.py'ye bakınız.
        """
        _eos_id = self.config.eos_token_id
        _, visual_prefix = self._encode_image(pixel_values)
        B      = visual_prefix.size(0)
        device = pixel_values.device

        # BOS seed token — "the" (id=1169)
        BOS_TOKEN_ID = 1169
        bos_ids = torch.full(
            (B, 1), fill_value=BOS_TOKEN_ID, dtype=torch.long, device=device
        )
        bos_embeds = self.gpt.transformer.wte(bos_ids)  # (B, 1, 768)

        combined_embeds = torch.cat([visual_prefix, bos_embeds], dim=1)  # (B, P+1, 768)
        combined_mask   = torch.ones(
            B, combined_embeds.size(1), dtype=torch.long, device=device
        )

        _do_sample = do_sample if num_beams == 1 else False

        for _k in ("early_stopping", "eos_token_id", "pad_token_id",
                   "forced_eos_token_id", "decoder_input_ids", "input_ids"):
            kwargs.pop(_k, None)

        output_ids = self.gpt.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature if _do_sample else 1.0,
            top_p=top_p          if _do_sample else 1.0,
            do_sample=_do_sample,
            early_stopping=True,
            length_penalty=length_penalty,
            eos_token_id=_eos_id,
            pad_token_id=_eos_id,
            **kwargs,
        )
        return output_ids

    # ──────────────────────────────────────────────────────────────────────
    # FACTORY
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained_mtl(
        cls,
        encoder_name:                 str   = "microsoft/swin-base-patch4-window7-224",
        decoder_name:                 str   = "distilgpt2",
        num_classes:                  int   = NUM_CLASSES,
        num_prefix_tokens:            int   = NUM_PREFIX_TOKENS,
        proj_dropout:                 float = 0.1,
        enable_gradient_checkpointing: bool = True,
    ) -> "SwinDistilGPT2ForMTL":
        """
        Pretrained Swin-B ve DistilGPT-2'den model oluşturur.

        Args:
            encoder_name  : HF Swin model adı
            decoder_name  : HF GPT-2 türevi model adı ('distilgpt2')
            num_classes   : Patoloji sınıfı sayısı (14)
            num_prefix_tokens: Vizüel prefix için kullanılacak Swin token sayısı (49)
            proj_dropout  : Projeksiyon katmanı dropout oranı
            enable_gradient_checkpointing: VRAM tasarrufu için GC aktif et
        """
        print(f"\n{'='*70}")
        print("🧠 Swin-B + DistilGPT-2 MTL Modeli Oluşturuluyor...")
        print(f"{'='*70}")
        print(f"   Encoder : {encoder_name}")
        print(f"   Decoder : {decoder_name}")
        print(f"   Sınıf   : {num_classes}")
        print(f"   Prefix  : {num_prefix_tokens} token  (Swin-B 7×7 grid)")

        # ── Swin-B ───────────────────────────────────────────────────────
        print("\n   📷 Swin-B encoder yükleniyor...")
        swin = _load_pretrained_offline_first(SwinModel, encoder_name)
        swin_dim = swin.config.hidden_size   # Bazı config'lerde bu stage-4 = 1024
        # Swin-B'de num_features: [128, 256, 512, 1024] — son stage 1024
        # hidden_size genellikle son stage boyutunu döndürür
        if hasattr(swin.config, "num_features"):
            swin_dim = swin.config.num_features[-1]
        print(f"   ✅ Swin-B hidden dim (stage-4): {swin_dim}")

        # ── DistilGPT-2 ───────────────────────────────────────────────────
        print("   📝 DistilGPT-2 decoder yükleniyor...")
        gpt     = _load_pretrained_offline_first(GPT2LMHeadModel, decoder_name)
        gpt_dim = gpt.config.n_embd             # 768
        print(f"   ✅ GPT-2 embed dim: {gpt_dim}")

        # ── Gradient Checkpointing ─────────────────────────────────────────
        if enable_gradient_checkpointing:
            try:
                swin.gradient_checkpointing_enable()
                print("   ✅ Swin-B Gradient Checkpointing aktif.")
            except Exception as e:
                print(f"   ⚠️  Swin GC atlandı: {e}")
            try:
                gpt.gradient_checkpointing_enable()
                print("   ✅ DistilGPT-2 Gradient Checkpointing aktif.")
            except Exception as e:
                print(f"   ⚠️  GPT-2 GC atlandı: {e}")

        # ── Model ─────────────────────────────────────────────────────────
        model = cls(
            swin_model=swin,
            gpt_model=gpt,
            swin_dim=swin_dim,
            gpt_dim=gpt_dim,
            num_classes=num_classes,
            num_prefix_tokens=num_prefix_tokens,
            proj_dropout=proj_dropout,
        )

        _print_model_stats(model)
        return model


# =========================================================================
# YARDIMCI: Model İstatistikleri
# =========================================================================
def _print_model_stats(model: SwinDistilGPT2ForMTL):
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    swin_params = sum(p.numel() for p in model.swin.parameters())
    gpt_params  = sum(p.numel() for p in model.gpt.parameters())
    proj_params = sum(p.numel() for p in model.visual_projection.parameters())
    cls_params  = sum(p.numel() for p in model.classifier_head.parameters())

    print(f"\n   📊 Model Parametre Özeti:")
    print(f"   {'─'*50}")
    print(f"   Toplam             : {total:>12,}")
    print(f"   Eğitilebilir       : {trainable:>12,} ({trainable/total*100:.1f}%)")
    print(f"   ├─ Swin-B Encoder  : {swin_params:>12,}")
    print(f"   ├─ DistilGPT-2     : {gpt_params:>12,}")
    print(f"   ├─ Visual Proj.    : {proj_params:>12,}")
    print(f"   └─ Classifier Head : {cls_params:>12,}")
    print(f"   {'─'*50}")
    print(f"{'='*70}\n")
