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
import argparse
import random
import numpy as np

pd.set_option("future.no_silent_downcasting", True)

# ── Proje Importları ──────────────────────────────────────────────────────
from config import Config
from src.data_loader.dataset_swin import get_swin_dataloaders, PATHOLOGY_COLS
from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL
from src.utils.visualization import plot_loss_curve
from src.utils.early_stopping import EarlyStopping


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
CHECKPOINT_DIR = Config.SWIN_CHECKPOINT_DIR


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
    def __init__(self, log_dir: str, exp_name: str = "swin_distilgpt2"):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name
        self.log_file = self.log_dir / f"{exp_name}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.history  = {
            "model_type":     "Swin-B-DistilGPT-2-MTL",
            "exp_name":       exp_name,
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
            "exp_name":        self.exp_name,
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
# =========================================================================
# KOMUT SATIRI ARGÜMANLARI
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Swin-B + DistilGPT-2 MTL Training & Ablation Pipeline")
    parser.add_argument("--exp_name", type=str, default="baseline",
                        help="Deney adı (örn: exp_A_no_hflip, exp_B_no_aux_loss, baseline)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint kayıt dizini (None ise otomatik izole edilir)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log ve grafik kayıt dizini (None ise otomatik izole edilir)")
    parser.add_argument("--p_hflip", type=float, default=0.3,
                        help="Yatay çevirme olasılığı (Deney A için 0.0)")
    parser.add_argument("--cls_loss_weight", type=float, default=0.5,
                        help="Yardımcı sınıflandırma kaybı ağırlığı (Deney B için 0.0)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-batch boyutu (varsayılan: 16)")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradyan biriktirme adımı (varsayılan: 16 -> efektif batch 256)")
    parser.add_argument("--max_train_epochs", type=int, default=11,
                        help="Eğitilecek gerçek epoch sayısı (orijinal derinlik: 11)")
    parser.add_argument("--total_scheduler_epochs", type=int, default=100,
                        help="Scheduler'ın toplam cosine epoch ufku (71.500 adım ölçeği korunur)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgelelik tohumu (varsayılan: 42)")
    parser.add_argument("--allow_black_fallback", action="store_true", default=False,
                        help="Eksik görüntüler için siyah fallback (varsayılan False: FileNotFoundError fırlatır)")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Eğitime başlamadan önce genişletilmiş dry-run doğrulaması yap ve çık")
    parser.add_argument("--dry_run_batches", type=int, default=16,
                        help="Dry-run aşama başına batch sayısı (en az 16 = 1 tam optimizer adımı)")
    parser.add_argument("--no_shutdown", action="store_true", default=True,
                        help="Eğitim/dry-run bitiminde bilgisayarın otomatik kapanmasını engelle")
    return parser.parse_args()


# =========================================================================
# GENİŞLETİLMİŞ DRY-RUN DOĞRULAMASI (Danışman Özel Talimatı)
# =========================================================================
def run_extended_dry_run(args, device) -> dict:
    print(f"\n{'='*80}")
    print("🔬 EXTENDED DRY-RUN VALIDATION (Advisor Mandate)")
    print(f"{'='*80}")
    print(f"   Deney Adı        : {args.exp_name}")
    print(f"   Cihaz            : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Batch Size       : {args.batch_size}")
    print(f"   Grad Accum Steps : {args.grad_accum}")
    print(f"   Efektif Batch    : {args.batch_size * args.grad_accum} ({args.dry_run_batches} mini-batch = 1 tam optimizer adımı)")
    print(f"   p_hflip          : {args.p_hflip} ({'KAPALI (Deney A)' if args.p_hflip == 0.0 else 'AÇIK'})")
    print(f"   cls_loss_weight  : {args.cls_loss_weight} ({'SIFIR - Sadece Gen (Deney B)' if args.cls_loss_weight == 0.0 else '0.5 MTL'})")
    print(f"   Black Fallback   : {'AÇIK' if args.allow_black_fallback else 'KAPALI (Eksik görüntü FileNotFoundError fırlatır)'}")
    print(f"   Checkpoint Dir   : {args.checkpoint_dir}")
    print(f"   Log Dir          : {args.log_dir}")
    print(f"{'='*80}\n")

    dry_run_results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "p_hflip": args.p_hflip,
        "cls_loss_weight": args.cls_loss_weight,
        "checks": {}
    }

    # 1. TEST STRICT IMAGE LOADING (Eksik görüntü testi)
    print("1️⃣ Kontrol: Eksik görüntü yükleme katılığı test ediliyor...")
    from src.data_loader.dataset_swin import MedicalSwinDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    import tempfile
    dummy_csv = os.path.join(tempfile.gettempdir(), "dummy_missing_test.csv")
    pd.DataFrame([{
        "image_path": "non_existent_folder/fake_cxr_9999.jpg",
        "final_report": "normal chest radiograph.",
        **{c: 0.0 for c in PATHOLOGY_COLS}
    }]).to_csv(dummy_csv, index=False)

    strict_ds = MedicalSwinDataset(
        csv_file=dummy_csv,
        image_dir=Config.IMAGE_DIR,
        tokenizer=tokenizer,
        allow_black_fallback=False
    )
    try:
        _ = strict_ds[0]
        raise RuntimeError("HATA: Eksik görüntü FileNotFoundError fırlatmadı!")
    except FileNotFoundError as fnf:
        print(f"   ✅ BAŞARILI: allow_black_fallback=False iken eksik görüntü yakalandı: {type(fnf).__name__}")
        dry_run_results["checks"]["strict_image_loading"] = "PASSED (FileNotFoundError raised as expected)"
    finally:
        if os.path.exists(dummy_csv):
            os.remove(dummy_csv)

    # 2. REAL DATALOADERS
    print(f"\n2️⃣ Kontrol: Gerçek veri yükleyicileri hazırlanıyor (Batch Size = {args.batch_size})...")
    train_loader, val_loader, tokenizer = get_swin_dataloaders(
        train_csv=Config.TRAIN_PROCESSED_CSV,
        val_csv=Config.VAL_PROCESSED_CSV,
        image_dir=Config.IMAGE_DIR,
        batch_size=args.batch_size,
        max_length=Config.VIT_MAX_LENGTH,
        image_size=224,
        num_workers=0,
        tokenizer_name=GPT_MODEL_NAME,
        p_hflip=args.p_hflip,
        allow_black_fallback=args.allow_black_fallback,
    )
    dry_run_results["checks"]["train_samples"] = len(train_loader.dataset)
    dry_run_results["checks"]["val_samples"] = len(val_loader.dataset)

    # 3. BUILD MODEL & OPTIMIZER
    print("\n3️⃣ Kontrol: Model ve diferansiyel optimizer oluşturuluyor...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name=SWIN_MODEL_NAME,
        decoder_name=GPT_MODEL_NAME,
        enable_gradient_checkpointing=True,
        cls_loss_weight=args.cls_loss_weight,
    ).to(device)

    model.gpt.config.pad_token_id = tokenizer.pad_token_id
    model.gpt.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id     = tokenizer.pad_token_id
    model.config.eos_token_id     = tokenizer.eos_token_id

    optimizer = build_optimizer(model, weight_decay=Config.SWIN_WEIGHT_DECAY)
    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = args.total_scheduler_epochs * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    # 4. PHASE 1: HEAD-ONLY (Epoch 0 - Warmup) DRY-RUN
    print("\n4️⃣ Kontrol: Aşama 1 — Head-Only (Epoch 0 Warmup) simülasyonu...")
    set_model_phase(model, is_head_only=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    swin_frozen = all(not p.requires_grad for p in model.swin.parameters())
    gpt_frozen = all(not p.requires_grad for p in model.gpt.parameters())
    proj_trainable = all(p.requires_grad for p in model.visual_projection.parameters())
    cls_trainable = all(p.requires_grad for p in model.classifier_head.parameters())
    assert swin_frozen, "Swin encoder head-only modunda donmuş olmalıdır!"
    assert gpt_frozen, "GPT decoder head-only modunda donmuş olmalıdır!"
    assert proj_trainable, "Visual projection head-only modunda eğitilebilir olmalıdır!"
    assert cls_trainable, "Classifier head head-only modunda eğitilebilir olmalıdır!"
    print("   ✅ Head-only requires_grad durumları doğrulandı (Swin/GPT donuk, Proj/Cls açık).")

    head_only_losses = []
    data_iter = iter(train_loader)
    for step in range(args.dry_run_batches):
        batch = next(data_iter)
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels_cls = batch["labels_cls"].to(device)

        ctx = autocast(device_type=device.type) if scaler else contextlib.nullcontext()
        with ctx:
            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_cls=labels_cls,
            )
            loss = outputs["loss"] / args.grad_accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        head_only_losses.append(loss.item() * args.grad_accum)

    proj_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.visual_projection.parameters())
    swin_has_no_grad = all(p.grad is None for p in model.swin.parameters())
    gpt_has_no_grad = all(p.grad is None for p in model.gpt.parameters())

    assert proj_grad_exists, "Visual projection gradyan almalıdır!"
    assert swin_has_no_grad, "Swin encoder head-only'de gradyan almamalıdır!"
    assert gpt_has_no_grad, "GPT decoder head-only'de gradyan almamalıdır!"

    if args.cls_loss_weight == 0.0:
        cls_has_zero_grad = all(p.grad is None or p.grad.abs().sum() == 0 for p in model.classifier_head.parameters())
        assert cls_has_zero_grad, "Deney B'de (cls_loss_weight=0.0) classifier_head SIFIR gradyan almalıdır!"
        print("   ✅ Deney B doğrulaması: classifier_head gradyanı TAMAMEN SIFIR!")
        dry_run_results["checks"]["head_only_cls_grad"] = "ZERO_GRAD (Confirmed for Exp B)"
    else:
        cls_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier_head.parameters())
        assert cls_grad_exists, "MTL baseline / Deney A'da classifier_head gradyan almalıdır!"
        print("   ✅ Deney A / Baseline doğrulaması: classifier_head gradyanı aktif.")
        dry_run_results["checks"]["head_only_cls_grad"] = "ACTIVE_GRAD (Confirmed for MTL/Exp A)"

    if scaler:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    print(f"   ✅ Head-only {args.dry_run_batches} mini-batch ve 1 tam optimizer güncellemesi tamamlandı (Ort. Loss: {np.mean(head_only_losses):.4f}).")
    vram_head_only = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"   📊 Head-only Peak VRAM: {vram_head_only:.2f} MiB")
    dry_run_results["checks"]["head_only_optimizer_step"] = "PASSED"
    dry_run_results["checks"]["head_only_peak_vram_mib"] = round(vram_head_only, 2)

    # 5. PHASE 2: END-TO-END FINE-TUNING (Epoch 2) DRY-RUN
    print("\n5️⃣ Kontrol: Aşama 2 — End-to-End Fine-Tuning (Epoch 2) simülasyonu...")
    set_model_phase(model, is_head_only=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable, "End-to-end modunda tüm parametreler requires_grad=True olmalıdır!"
    print("   ✅ End-to-end requires_grad durumu doğrulandı (tüm parametreler açık).")

    end_to_end_losses = []
    for step in range(args.dry_run_batches):
        batch = next(data_iter)
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels_cls = batch["labels_cls"].to(device)

        ctx = autocast(device_type=device.type) if scaler else contextlib.nullcontext()
        with ctx:
            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_cls=labels_cls,
            )
            loss = outputs["loss"] / args.grad_accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        end_to_end_losses.append(loss.item() * args.grad_accum)

    swin_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.swin.parameters())
    gpt_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gpt.parameters())
    proj_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.visual_projection.parameters())

    assert swin_grad_exists, "Swin encoder end-to-end'de gradyan almalıdır!"
    assert gpt_grad_exists, "GPT decoder end-to-end'de gradyan almalıdır!"
    assert proj_grad_exists, "Visual projection end-to-end'de gradyan almalıdır!"

    if args.cls_loss_weight == 0.0:
        cls_has_zero_grad = all(p.grad is None or p.grad.abs().sum() == 0 for p in model.classifier_head.parameters())
        assert cls_has_zero_grad, "Deney B'de end-to-end aşamasında dahi classifier_head SIFIR gradyan almalıdır!"
        print("   ✅ Deney B end-to-end doğrulaması: classifier_head SIFIR gradyan korundu!")
        dry_run_results["checks"]["e2e_cls_grad"] = "ZERO_GRAD (Confirmed for Exp B)"
    else:
        cls_grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier_head.parameters())
        assert cls_grad_exists, "MTL/Exp A end-to-end aşamasında classifier_head gradyan almalıdır!"
        print("   ✅ Deney A / Baseline end-to-end doğrulaması: classifier_head gradyanı aktif.")
        dry_run_results["checks"]["e2e_cls_grad"] = "ACTIVE_GRAD (Confirmed for MTL/Exp A)"

    if scaler:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    else:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    print(f"   ✅ End-to-end {args.dry_run_batches} mini-batch ve 1 tam optimizer güncellemesi tamamlandı (Ort. Loss: {np.mean(end_to_end_losses):.4f}).")
    vram_e2e = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    vram_reserved = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0
    total_gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0
    headroom = total_gpu_vram - vram_reserved

    print(f"\n{'─'*60}")
    print(f"📊 DONANIM VE VRAM ÖLÇÜM RAPORU (RTX 5070)")
    print(f"{'─'*60}")
    print(f"   Peak Allocated VRAM : {vram_e2e:>8.2f} MiB ({vram_e2e/1024:.2f} GB)")
    print(f"   Peak Reserved VRAM  : {vram_reserved:>8.2f} MiB ({vram_reserved/1024:.2f} GB)")
    print(f"   Toplam VRAM Kapasite: {total_gpu_vram:>8.2f} MiB ({total_gpu_vram/1024:.2f} GB)")
    print(f"   Kalan Güvenli Boşluk: {headroom:>8.2f} MiB ({headroom/1024:.2f} GB)")
    print(f"{'─'*60}")

    dry_run_results["checks"]["e2e_optimizer_step"] = "PASSED"
    dry_run_results["vram"] = {
        "peak_allocated_mib": round(vram_e2e, 2),
        "peak_allocated_gb": round(vram_e2e / 1024, 2),
        "peak_reserved_mib": round(vram_reserved, 2),
        "peak_reserved_gb": round(vram_reserved / 1024, 2),
        "total_capacity_mib": round(total_gpu_vram, 2),
        "total_capacity_gb": round(total_gpu_vram / 1024, 2),
        "headroom_mib": round(headroom, 2),
        "headroom_gb": round(headroom / 1024, 2),
    }

    # 6. VALIDATION PASS
    print("\n6️⃣ Kontrol: Doğrulama (Validation) adımı test ediliyor (4 batch)...")
    model.eval()
    val_losses = []
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(4):
            batch = next(val_iter)
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels_cls = batch["labels_cls"].to(device)

            outputs = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                labels_cls=labels_cls,
            )
            val_losses.append(outputs["loss"].item())

    print(f"   ✅ Validation 4 batch başarıyla çalıştı (Ort. Val Loss: {np.mean(val_losses):.4f}).")
    dry_run_results["checks"]["val_pass"] = "PASSED"

    # 7. SAVE DRY RUN REPORT
    os.makedirs(args.log_dir, exist_ok=True)
    report_file = os.path.join(args.log_dir, f"dry_run_report_{args.exp_name}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(dry_run_results, f, indent=2)
    print(f"\n💾 Dry-run raporu kaydedildi: {report_file}")
    print(f"\n{'='*80}")
    print(f"🎉 TÜM TEKNİK KONTROLLER VE DRY-RUN BAŞARIYLA TAMAMLANDI! [{args.exp_name}]")
    print(f"{'='*80}\n")
    return dry_run_results


# =========================================================================
# ANA FONKSİYON
# =========================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Dizin İzolasyonu ve Orijinal Dosyaları Koruma Garantisi ────────────
    if args.checkpoint_dir is None:
        if args.exp_name == "baseline":
            checkpoint_dir = Config.SWIN_CHECKPOINT_DIR
        else:
            checkpoint_dir = os.path.join("experiments", args.exp_name, "checkpoints")
    else:
        checkpoint_dir = args.checkpoint_dir

    if args.log_dir is None:
        if args.exp_name == "baseline":
            log_dir = Config.LOG_DIR
        else:
            log_dir = os.path.join("experiments", args.exp_name, "logs")
    else:
        log_dir = args.log_dir

    args.checkpoint_dir = checkpoint_dir
    args.log_dir = log_dir

    # Orijinal dosyaların ezilmesini önleyen kritik emniyet kilidi:
    orig_abs = os.path.abspath(Config.SWIN_CHECKPOINT_DIR).lower()
    curr_abs = os.path.abspath(checkpoint_dir).lower()
    if args.exp_name != "baseline" and orig_abs == curr_abs:
        raise RuntimeError(
            f"GÜVENLİK ENGELİ: '{args.exp_name}' deneyi orijinal checkpoint dizinine ({orig_abs}) "
            f"kaydedilemez! Lütfen ayrı bir dizin belirtin veya varsayılan izole yolu kullanın."
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = Config.DEVICE

    # ── Dry-Run Modu Çağrısı ──────────────────────────────────────────────
    if args.dry_run:
        run_extended_dry_run(args, device)
        return

    BEST_CKPT = os.path.join(checkpoint_dir, f"best_model_{args.exp_name}.pth")

    image_size  = 224
    max_length  = getattr(Config, "VIT_MAX_LENGTH",  150)
    batch_size  = args.batch_size
    num_workers = getattr(Config, "NUM_WORKERS",       0)

    print(f"\n{'='*80}")
    print(f"🔬 Swin-B + DistilGPT-2 MTL EĞİTİMİ: [{args.exp_name}]")
    print(f"{'='*80}")
    print(f"   Cihaz             : {device}")
    print(f"   Batch Size        : {batch_size}")
    print(f"   Grad Accum        : {args.grad_accum} (Efektif Batch: {batch_size * args.grad_accum})")
    print(f"   p_hflip           : {args.p_hflip}")
    print(f"   cls_loss_weight   : {args.cls_loss_weight}")
    print(f"   Image Size        : {image_size}×{image_size}")
    print(f"   Max Length        : {max_length} token")
    print(f"   Hedef Epoch       : {args.max_train_epochs} (Toplam scheduler ufku: {args.total_scheduler_epochs})")
    print(f"   Checkpoint Dizin  : {checkpoint_dir}")
    print(f"   Log Dizin         : {log_dir}")
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
        p_hflip=args.p_hflip,
        allow_black_fallback=args.allow_black_fallback,
    )

    # ── MODEL ─────────────────────────────────────────────────────────────
    print("\n🧠 Model yükleniyor...")
    model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name=SWIN_MODEL_NAME,
        decoder_name=GPT_MODEL_NAME,
        enable_gradient_checkpointing=True,
        cls_loss_weight=args.cls_loss_weight,
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
    steps_per_epoch     = max(steps_per_epoch_raw // args.grad_accum, 1)
    total_steps         = args.total_scheduler_epochs * steps_per_epoch
    warmup_steps        = WARMUP_EPOCHS * steps_per_epoch

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ── AMP ───────────────────────────────────────────────────────────────
    scaler = GradScaler() if USE_AMP and device.type == "cuda" else None

    # ── CHECKPOINT RESUME ─────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    latest_ckpt   = find_latest_checkpoint(checkpoint_dir)

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

    print_training_plan(args.max_train_epochs, args.grad_accum, batch_size)

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
    logger        = TrainingLogger(log_dir, exp_name=args.exp_name)
    train_history = []
    val_history   = []
    early_stopper = EarlyStopping(patience=Config.PATIENCE, min_delta=0.001)

    # =========================================================================
    # EĞİTİM DÖNGÜSÜ
    # =========================================================================
    for epoch in range(start_epoch, args.max_train_epochs):
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
            scaler=scaler, grad_accum_steps=args.grad_accum, grad_clip=GRAD_CLIP,
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
            save_dir=checkpoint_dir,
            filename=f"{args.exp_name}_loss.png",
        )

        # ── Sonuç Yazdır ──────────────────────────────────────────────────
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        swa_tag     = "  [🌀 SWA]" if swa_active else ""
        print(f"\n{'─'*80}")
        print(f"📊 Epoch {epoch+1}/{args.max_train_epochs}  ({duration/60:.1f} dk)  [{phase_label}]{swa_tag}")
        print(f"{'─'*80}")
        print(f"   {'':30s}  {'TRAIN':>10}   {'VAL':>10}")
        print(f"   {'─'*52}")
        print(f"   {'Total Loss':<30s}  {train_loss:>10.4f}   {val_loss:>10.4f}")
        print(f"   {'  ├─ Generation (CE)  ×1.0':<30s}  {train_gen:>10.4f}   {val_gen:>10.4f}")
        print(f"   {'  └─ Classification (BCE)×' + str(args.cls_loss_weight):<30s}  {train_cls:>10.4f}   {val_cls:>10.4f}")
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
            periodic = os.path.join(checkpoint_dir, f"{args.exp_name}_checkpoint_epoch_{epoch+1}.pth")
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
        swa_ckpt_path = os.path.join(checkpoint_dir, f"{args.exp_name}_swa_model_final.pth")
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
    print(f"   Deney Adı     : {args.exp_name}")
    print(f"   Total Epochs  : {summ['total_epochs']}")
    print(f"   Best Val Loss : {summ['best_val_loss']:.4f}")
    print(f"   Log File      : {summ['log_file']}")
    print(f"   Best Model    : {BEST_CKPT}")
    if swa_active:
        print(f"   SWA Model     : {swa_ckpt_path}")
    print(f"{'='*80}\n")

    if getattr(Config, "SHUTDOWN_AFTER_TRAIN", False) and not args.no_shutdown:
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
