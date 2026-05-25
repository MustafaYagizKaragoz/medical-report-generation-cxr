"""
train_swin_distilgpt2.py
========================
Swin-B + DistilGPT-2 Multi-Task Learning Eğitim Pipeline'ı

Özellikler:
  - Diferansiyel LR: Swin-B=2e-6, DistilGPT-2+Projeksiyon=2e-4
  - AMP (Mixed Precision FP16)
  - Gradient Checkpointing (Swin-B + GPT-2)
  - Gradient Accumulation (16 adım → efektif batch = batch_size * 16)
  - Linear Warmup + Cosine Decay scheduler
  - SWA (Stochastic Weight Averaging) — epoch 15'ten itibaren
  - Checkpoint resume
  - EarlyStopping (val kaybına göre)
  - Offline-first model yükleme (HuggingFace timeout koruması)

ViT pipeline'ından farklar:
  - Encoder: google/vit-base-patch16-224 → microsoft/swin-base-patch4-window7-224
  - ENCODER_LR: 4e-6 → 2e-6 (Swin-B daha fazla parametre içerir)
  - NUM_PREFIX_TOKENS: 197 → 49 (Swin-B stage-4: 7×7 spatial grid)
  - Model sınıfı: ViTDistilGPT2ForMTL → SwinDistilGPT2ForMTL
  - set_model_phase(): model.vit → model.swin
  - Checkpoint dizini: checkpoints_vit_distilgpt2 → checkpoints_swin_distilgpt2
  - ImageNet normalizasyonu: [0.5,0.5,0.5] → [0.485,0.456,0.406]
"""

# =========================================================================
# MPLBACKEND — matplotlib'in Tkinter ile hicbir zaman etkilesmemesi icin
# Bu satir her seyin onunde olmali (diger import'lardan once)
# =========================================================================
import os
os.environ["MPLBACKEND"] = "Agg"   # Tkinter bagimsiz non-interaktif backend

# =========================================================================
# HUGGINGFACE OFFLINE MODU
# =========================================================================

def _check_hf_cache(model_ids: list) -> bool:
    """HF hub cache dizininde tüm modeller mevcut mu?"""
    cache_root = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    for mid in model_ids:
        folder_name = "models--" + mid.replace("/", "--")
        if not os.path.isdir(os.path.join(cache_root, folder_name)):
            return False
    return True

_REQUIRED_MODELS = ["microsoft/swin-base-patch4-window7-224", "distilgpt2"]

if _check_hf_cache(_REQUIRED_MODELS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("🔒 HF Offline modu aktif (tüm modeller cache'de mevcut).")
else:
    print("🌐 HF modelleri indiriliyor (ilk çalıştırma)...")
    print("   ⚠️  İnternet bağlantısı gerekli — lütfen bekleyin.")
# =========================================================================

import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning)

import gc
import time
import json
import math
import contextlib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

# ── Proje Importları ──────────────────────────────────────────────────────
from config import Config
from src.data_loader.dataset_swin import get_swin_dataloaders
from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL
from src.utils.visualization import plot_loss_curve
from src.utils.early_stopping import EarlyStopping


# =========================================================================
# EĞİTİM SABİTLERİ
# =========================================================================
ENCODER_LR     = 2e-6    # Swin-B Encoder (pretrained, çok düşük LR)
DECODER_LR     = 2e-4    # DistilGPT-2 + Projeksiyon + Classifier
WARMUP_EPOCHS  = 2       # İlk N epoch: sadece projeksiyon/classifier eğitilir
GRAD_ACCUM     = 16      # Gradyan birikimi adımı
GRAD_CLIP      = 1.0     # Gradyan kırpma
USE_AMP        = True    # Mixed Precision

# ── SWA ──────────────────────────────────────────────────────────────────
SWA_START_EPOCH   = 15    # Bu epoch'tan itibaren SWA etkinleşir (0-indexed)
SWA_LR            = 5e-7  # SWALR sabit öğrenme hızı
SWA_ANNEAL_EPOCHS = 3     # SWALR annealing epoch sayısı

SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
GPT_MODEL_NAME  = "distilgpt2"
CHECKPOINT_DIRNAME = "checkpoints_swin_distilgpt2"


# =========================================================================
# YARDIMCI: Eğitim Planı Yazdır
# =========================================================================
def print_training_plan(total_epochs: int, grad_accum: int, batch_size: int):
    eff_batch = batch_size * grad_accum
    print(f"\n{'='*80}")
    print("🚀 Swin-B + DistilGPT-2 MTL EĞİTİM PLANI")
    print(f"{'='*80}")
    print(f"  Encoder  (Swin-B) LR : {ENCODER_LR:.1e}")
    print(f"  Decoder  (GPT)    LR : {DECODER_LR:.1e}")
    print(f"  Warmup Epochs        : {WARMUP_EPOCHS} (sadece Proj+Cls)")
    print(f"  Gradient Accum       : {grad_accum} adım")
    print(f"  Efektif Batch Size   : {eff_batch}")
    print(f"  Mixed Precision      : {'FP16 (AMP)' if USE_AMP else 'FP32'}")
    print(f"  Loss                 : 1.0×CE(gen) + 0.5×BCE(cls)")
    print(f"  Toplam Epoch         : {total_epochs}")
    print(f"  SWA Başlangıç        : Epoch {SWA_START_EPOCH + 1}  (LR={SWA_LR:.1e})")
    print(f"{'='*80}\n")


# =========================================================================
# SCHEDULER: Linear Warmup + Cosine Decay
# =========================================================================
def get_warmup_cosine_scheduler(
    optimizer:      optim.Optimizer,
    warmup_steps:   int,
    total_steps:    int,
    eta_min_ratio:  float = 0.01,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min_ratio, cos_val)

    return LambdaLR(optimizer, lr_lambda)


# =========================================================================
# OPTIMIZER: Diferansiyel LR
# =========================================================================
def build_optimizer(model: SwinDistilGPT2ForMTL, weight_decay: float = 0.01) -> optim.AdamW:
    """
    Parametre grupları:
      - encoder (Swin-B)                     → ENCODER_LR
      - decoder (GPT) + projeksiyon + cls    → DECODER_LR
    """
    enc_params = [p for n, p in model.named_parameters() if n.startswith("swin.")]
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("swin.")]

    optimizer = optim.AdamW(
        [
            {"params": enc_params, "lr": ENCODER_LR, "name": "swin_encoder"},
            {"params": dec_params, "lr": DECODER_LR, "name": "decoder_proj_cls"},
        ],
        weight_decay=weight_decay,
    )

    n_enc_p = sum(p.numel() for p in enc_params)
    n_dec_p = sum(p.numel() for p in dec_params)
    print(f"   ├─ Swin-B Encoder   : {n_enc_p:,} param (LR={ENCODER_LR:.1e})")
    print(f"   └─ GPT+Proj+Cls     : {n_dec_p:,} param (LR={DECODER_LR:.1e})")
    return optimizer


# =========================================================================
# MODEL PHASE
# =========================================================================
def set_model_phase(model: SwinDistilGPT2ForMTL, is_head_only: bool):
    """
    head_only = True  → sadece visual_projection + classifier_head güncellenir.
    head_only = False → tüm parametreler açık; GC use_reentrant=False ile aktif.
    """
    for name, param in model.named_parameters():
        if is_head_only:
            param.requires_grad = ("visual_projection" in name or "classifier_head" in name)
        else:
            param.requires_grad = True

    gc_kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    try:
        if is_head_only:
            model.swin.gradient_checkpointing_disable()
            model.gpt.gradient_checkpointing_disable()
        else:
            model.swin.gradient_checkpointing_enable(**gc_kwargs)
            model.gpt.gradient_checkpointing_enable(**gc_kwargs)
    except TypeError:
        try:
            if is_head_only:
                model.swin.gradient_checkpointing_disable()
                model.gpt.gradient_checkpointing_disable()
            else:
                model.swin.gradient_checkpointing_enable()
                model.gpt.gradient_checkpointing_enable()
        except Exception:
            pass
    except Exception:
        pass


# =========================================================================
# CHECKPOINT
# =========================================================================
CHECKPOINT_DIR = os.path.join(Config.BASE_DIR, CHECKPOINT_DIRNAME)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, phase, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch":       epoch,
        "val_loss":    val_loss,
        "phase":       phase,
        "timestamp":   datetime.now().isoformat(),
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict() if scheduler else None,
        "scaler":      scaler.state_dict()    if scaler    else None,
    }
    torch.save(state, filepath)
    print(f"   💾 Checkpoint kaydedildi: {filepath}")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    checkpoint_dir içindeki en son geçerli checkpoint'i döndürür.
    'BOZUK' içeren dosyalar atlanır.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if (
            f.endswith(".pth")
            and ("swin_" in f or "best_" in f)
            and "BOZUK" not in f
        )
    ]
    if not candidates:
        return None

    latest, max_ep = None, -1
    for fp in candidates:
        try:
            ckpt = torch.load(fp, map_location="cpu", weights_only=False)
            ep   = ckpt.get("epoch", -1)
            if ep > max_ep:
                max_ep = ep
                latest = fp
            del ckpt
        except Exception:
            continue
    return latest


# =========================================================================
# LOGGER
# =========================================================================
class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"swin_distilgpt2_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.history  = {
            "model_type":     "Swin-B-DistilGPT-2-MTL",
            "epochs":         [],
            "train_losses":   [],
            "val_losses":     [],
            "learning_rates": [],
            "phases":         [],
            "timestamps":     [],
        }

    def log_epoch(self, epoch, train_loss, val_loss, lr, phase):
        self.history["epochs"].append(epoch + 1)
        self.history["train_losses"].append(train_loss)
        self.history["val_losses"].append(val_loss)
        self.history["learning_rates"].append(lr)
        self.history["phases"].append(phase)
        self.history["timestamps"].append(datetime.now().isoformat())
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def summary(self):
        return {
            "total_epochs":    len(self.history["epochs"]),
            "best_val_loss":   min(self.history["val_losses"])   if self.history["val_losses"]   else None,
            "best_train_loss": min(self.history["train_losses"]) if self.history["train_losses"] else None,
            "log_file":        str(self.log_file),
        }


# =========================================================================
# TRAIN ONE EPOCH
# =========================================================================
def train_one_epoch(
    loader,
    model:             SwinDistilGPT2ForMTL,
    optimizer:         optim.AdamW,
    scheduler:         LambdaLR,
    device:            torch.device,
    epoch:             int,
    scaler:            GradScaler = None,
    grad_accum_steps:  int   = 16,
    grad_clip:         float = 1.0,
) -> tuple:
    """Returns (avg_total_loss, avg_gen_loss, avg_cls_loss)"""
    model.train()
    total_loss = 0.0
    total_gen  = 0.0
    total_cls  = 0.0
    n_batches  = 0
    optimizer.zero_grad(set_to_none=True)

    loop = tqdm(
        loader, total=len(loader), leave=False,
        desc=f"🚀 Epoch {epoch+1} [Train]", dynamic_ncols=True
    )

    for step, batch in enumerate(loop):
        images         = batch["images"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        labels_cls     = batch.get("labels_cls")
        if labels_cls is not None:
            labels_cls = labels_cls.to(device)

        use_amp = scaler is not None
        ctx = autocast(device_type=device.type) if use_amp else contextlib.nullcontext()

        with ctx:
            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_cls=labels_cls,
            )
            loss = outputs["loss"] / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        actual_loss = loss.item() * grad_accum_steps
        gen_val = outputs["loss_gen"].item() if outputs["loss_gen"] is not None else 0.0
        cls_val = outputs["loss_cls"].item() if outputs["loss_cls"] is not None else 0.0

        total_loss += actual_loss
        total_gen  += gen_val
        total_cls  += cls_val
        n_batches  += 1

        loop.set_postfix({
            "tot": f"{actual_loss:.4f}",
            "gen": f"{gen_val:.4f}",
            "cls": f"{cls_val:.4f}",
            "lr":  f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    n = max(n_batches, 1)
    return total_loss / n, total_gen / n, total_cls / n


# =========================================================================
# VALIDATE
# =========================================================================
def validate(loader, model, device, epoch) -> tuple:
    """Returns (avg_total_loss, avg_gen_loss, avg_cls_loss)"""
    model.eval()
    total_loss = 0.0
    total_gen  = 0.0
    total_cls  = 0.0
    n_batches  = 0

    loop = tqdm(
        loader, total=len(loader), leave=False,
        desc=f"🔍 Epoch {epoch+1} [Val]", dynamic_ncols=True
    )

    with torch.no_grad():
        for batch in loop:
            images         = batch["images"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            labels_cls     = batch.get("labels_cls")
            if labels_cls is not None:
                labels_cls = labels_cls.to(device)

            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_cls=labels_cls,
            )

            tot = outputs["loss"].item()
            gen = outputs["loss_gen"].item() if outputs["loss_gen"] is not None else 0.0
            cls = outputs["loss_cls"].item() if outputs["loss_cls"] is not None else 0.0

            total_loss += tot
            total_gen  += gen
            total_cls  += cls
            n_batches  += 1

            loop.set_postfix({
                "tot": f"{tot:.4f}",
                "gen": f"{gen:.4f}",
                "cls": f"{cls:.4f}",
            })

    n = max(n_batches, 1)
    return total_loss / n, total_gen / n, total_cls / n


# =========================================================================
# ANA FONKSİYON
# =========================================================================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    device = Config.DEVICE

    BEST_CKPT = os.path.join(CHECKPOINT_DIR, "best_model_swin_distilgpt2.pth")

    image_size  = 224
    max_length  = getattr(Config, "VIT_MAX_LENGTH",  150)
    batch_size  = getattr(Config, "VIT_BATCH_SIZE",    4)
    num_workers = getattr(Config, "NUM_WORKERS",       0)

    print(f"\n{'='*80}")
    print("🔬 Swin-B + DistilGPT-2 MTL EĞİTİMİ")
    print(f"{'='*80}")
    print(f"   Cihaz      : {device}")
    print(f"   Batch Size : {batch_size}")
    print(f"   Image Size : {image_size}×{image_size}")
    print(f"   Max Length : {max_length} token")
    print(f"   Epochs     : {Config.EPOCHS}")
    print(f"   Checkpoint : {CHECKPOINT_DIR}")
    print(f"{'='*80}\n")

    # ── VERİ ──────────────────────────────────────────────────────────────
    print("📥 Veri yükleniyor...")
    train_loader, val_loader, tokenizer = get_swin_dataloaders(
        train_csv=Config.TRAIN_PROCESSED_CSV,
        val_csv=Config.VAL_PROCESSED_CSV,
        image_dir=Config.IMAGE_DIR,
        batch_size=batch_size,
        max_length=max_length,
        image_size=image_size,
        num_workers=num_workers,
        tokenizer_name=GPT_MODEL_NAME,
    )

    # ── MODEL ─────────────────────────────────────────────────────────────
    print("\n🧠 Model yükleniyor...")
    model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name=SWIN_MODEL_NAME,
        decoder_name=GPT_MODEL_NAME,
        enable_gradient_checkpointing=True,
    ).to(device)

    model.gpt.config.pad_token_id = tokenizer.pad_token_id
    model.gpt.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id     = tokenizer.pad_token_id
    model.config.eos_token_id     = tokenizer.eos_token_id

    # ── OPTİMİZASYON ──────────────────────────────────────────────────────
    print("\n⚙️  Optimizer ve Scheduler hazırlanıyor...")
    wd = getattr(Config, "VIT_WEIGHT_DECAY", 0.01)
    optimizer = build_optimizer(model, weight_decay=wd)

    steps_per_epoch_raw = len(train_loader)
    steps_per_epoch     = max(steps_per_epoch_raw // GRAD_ACCUM, 1)
    total_steps         = Config.EPOCHS * steps_per_epoch
    warmup_steps        = WARMUP_EPOCHS * steps_per_epoch

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ── AMP ───────────────────────────────────────────────────────────────
    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    # ── CHECKPOINT RESUME ─────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    latest_ckpt   = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt:
        print(f"\n📥 Checkpoint bulundu: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_epoch   = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if ckpt.get("optimizer"):
            try: optimizer.load_state_dict(ckpt["optimizer"])
            except Exception: pass
        if ckpt.get("scheduler"):
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except Exception: pass
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt
        gc.collect()
        print(f"✅ Epoch {start_epoch + 1}'den devam ediliyor. Best val loss: {best_val_loss:.4f}")
    else:
        print("\n✨ Yeni eğitim başlatılıyor...")

    print_training_plan(Config.EPOCHS, GRAD_ACCUM, batch_size)

    # ── SWA HAZIRLIK ──────────────────────────────────────────────────────
    swa_model = AveragedModel(model, device=device)
    try:
        swa_model.module.swin.gradient_checkpointing_disable()
        swa_model.module.gpt.gradient_checkpointing_disable()
    except Exception:
        pass

    swa_scheduler = SWALR(
        optimizer,
        swa_lr=SWA_LR,
        anneal_epochs=SWA_ANNEAL_EPOCHS,
        anneal_strategy="cos",
    )
    swa_active = False

    # ── TRACKING ──────────────────────────────────────────────────────────
    logger        = TrainingLogger(Config.LOG_DIR)
    train_history = []
    val_history   = []
    early_stopper = EarlyStopping(patience=Config.PATIENCE, min_delta=0.001)

    # =========================================================================
    # EĞİTİM DÖNGÜSÜ
    # =========================================================================
    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()

        # SWA aktifleşme kontrolü
        if epoch >= SWA_START_EPOCH and not swa_active:
            swa_active = True
            print(f"\n🌀 SWA ETKİNLEŞTİ — Epoch {epoch+1}'den itibaren ağırlıklar ortalanacak.")

        # Warmup döneminde sadece projeksiyon + sınıflandırıcı
        is_head_only = epoch < WARMUP_EPOCHS
        set_model_phase(model, is_head_only=is_head_only)

        if swa_active:
            phase_label = "swa"
        elif is_head_only:
            phase_label = "head_only"
        else:
            phase_label = "end_to_end"

        if is_head_only:
            print(f"\n⚠️  Epoch {epoch+1} → HEAD-ONLY modu (Visual Proj + Classifier)")

        # ── Train ─────────────────────────────────────────────────────────
        train_loss, train_gen, train_cls = train_one_epoch(
            train_loader, model, optimizer, scheduler, device, epoch,
            scaler=scaler, grad_accum_steps=GRAD_ACCUM, grad_clip=GRAD_CLIP,
        )

        # ── SWA: Ağırlık Ortalaması & Scheduler ───────────────────────────
        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"   🌀 SWA: ağırlıklar güncellendi  "
                  f"(SWA LR≈{optimizer.param_groups[0]['lr']:.2e})")

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_gen, val_cls = validate(val_loader, model, device, epoch)

        duration   = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        train_history.append(train_loss)
        val_history.append(val_loss)
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, phase_label)

        # ── Kayıp Grafiği ─────────────────────────────────────────────────
        plot_loss_curve(
            train_history, val_history,
            save_dir=CHECKPOINT_DIR,
            filename="swin_distilgpt2_loss.png",
        )

        # ── Sonuç Yazdır ──────────────────────────────────────────────────
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swa_tag     = "  [🌀 SWA]" if swa_active else ""
        print(f"\n{'─'*80}")
        print(f"📊 Epoch {epoch+1}/{Config.EPOCHS}  ({duration/60:.1f} dk)  [{phase_label}]{swa_tag}")
        print(f"{'─'*80}")
        print(f"   {'':30s}  {'TRAIN':>10}   {'VAL':>10}")
        print(f"   {'─'*52}")
        print(f"   {'Total Loss':<30s}  {train_loss:>10.4f}   {val_loss:>10.4f}")
        print(f"   {'  ├─ Generation (CE)  ×1.0':<30s}  {train_gen:>10.4f}   {val_gen:>10.4f}")
        print(f"   {'  └─ Classification (BCE)×0.5':<30s}  {train_cls:>10.4f}   {val_cls:>10.4f}")
        print(f"   {'─'*52}")
        print(f"   Swin LR      : {optimizer.param_groups[0]['lr']:.2e}")
        if len(optimizer.param_groups) > 1:
            print(f"   GPT LR       : {optimizer.param_groups[1]['lr']:.2e}")
        print(f"   Trainable    : {n_trainable:,}")
        print(f"{'─'*80}")

        # ── Best Checkpoint ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            print(f"   🏆 Yeni Best!  {best_val_loss:.4f} → {val_loss:.4f}")
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, val_loss, phase_label, BEST_CKPT)
        else:
            print(f"   ❌ Val loss düşmedi. Best: {best_val_loss:.4f}")

        # ── Periyodik Checkpoint ──────────────────────────────────────────
        if (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0:
            periodic = os.path.join(CHECKPOINT_DIR, f"swin_checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, val_loss, phase_label, periodic)

        # ── Early Stopping ────────────────────────────────────────────────
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"\n🛑 EARLY STOPPING — {Config.PATIENCE} epoch boyunca iyileşme olmadı.")
            break

    # ── SWA: BN Güncelleme & Final Checkpoint ─────────────────────────────
    if swa_active:
        print("\n🌀 SWA BatchNorm istatistikleri güncelleniyor...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_ckpt_path = os.path.join(CHECKPOINT_DIR, "swa_model_final.pth")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "model_state": swa_model.module.state_dict(),
                "epoch":       epoch,
                "val_loss":    best_val_loss,
                "phase":       "swa",
                "timestamp":   datetime.now().isoformat(),
            },
            swa_ckpt_path,
        )
        print(f"   💾 SWA modeli kaydedildi: {swa_ckpt_path}")

    # ── Özet ──────────────────────────────────────────────────────────────
    summ = logger.summary()
    print(f"\n{'='*80}")
    print("🎉 EĞİTİM TAMAMLANDI!")
    print(f"{'='*80}")
    print(f"   Total Epochs  : {summ['total_epochs']}")
    print(f"   Best Val Loss : {summ['best_val_loss']:.4f}")
    print(f"   Log File      : {summ['log_file']}")
    print(f"   Best Model    : {BEST_CKPT}")
    if swa_active:
        print(f"   SWA Model     : {swa_ckpt_path}")
    print(f"{'='*80}\n")

    if getattr(Config, "SHUTDOWN_AFTER_TRAIN", False):
        print("⚠️  Bilgisayar 60 saniye içinde kapanacak...")
        time.sleep(5)
        os.system("shutdown /s /t 60")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Kullanıcı tarafından durduruldu.")
    except Exception as e:
        import traceback
        print(f"\n❌ HATA: {e}")
        traceback.print_exc()
