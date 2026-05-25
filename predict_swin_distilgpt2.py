"""
predict_swin_distilgpt2.py
===========================
Swin-B + DistilGPT-2 MTL modelinden rastgele N örnek üretir,
NLP + klinik metriklerini hesaplar ve kapsamlı görsel (PNG) kaydeder.

Ozellikler:
  - Model      : SwinDistilGPT2ForMTL
  - Encoder    : microsoft/swin-base-patch4-window7-224
  - Heatmap    : Swin Stage-4 Spatial Saliency (7x7 L2-norm)
  - Normalizasyon : ImageNet [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
  - Checkpoint : checkpoints_swin_distilgpt2/
  - Cikti      : predict_swin_distilgpt2/
"""

# ── HF Offline Koruması ───────────────────────────────────────────────────
import os as _os
def _hf_cache_ok(ids):
    root = _os.environ.get("HF_HOME", _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    return all(_os.path.isdir(_os.path.join(root, "models--" + m.replace("/", "--"))) for m in ids)
if _hf_cache_ok(["microsoft/swin-base-patch4-window7-224", "distilgpt2"]):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# ─────────────────────────────────────────────────────────────────────────

import os
import sys
import random
import textwrap
import warnings
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd

warnings.filterwarnings("ignore")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _sbert = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
except ImportError:
    SentenceTransformer = None
    _sbert = None

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import Config
from src.data_loader.dataset_swin import get_eval_transform   # Swin normalizasyonu
from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════════════════════════════
CHECKPOINT_DIR  = os.path.join(Config.BASE_DIR, "checkpoints_swin_distilgpt2")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model_swin_distilgpt2.pth")

# best_model yoksa SWA checkpoint'ini dene
if not os.path.exists(CHECKPOINT_PATH):
    _swa_path = os.path.join(CHECKPOINT_DIR, "swa_model_final.pth")
    if os.path.exists(_swa_path):
        CHECKPOINT_PATH = _swa_path
        print(f"INFO: best_model bulunamadi; SWA checkpoint kullaniliyor: {_swa_path}")

TEST_CSV     = Config.TEST_PROCESSED_CSV
IMAGE_ROOT   = Config.IMAGE_DIR
DEVICE       = Config.DEVICE
IMAGE_SIZE   = 224
NUM_SAMPLES  = 5
BEAM_SIZE    = getattr(Config, "SWIN_NUM_BEAMS",  4)
MAX_NEW_TOKS = getattr(Config, "VIT_MAX_LENGTH",  160)   # token sayısı aynı
REP_PENALTY  = 2.0
NO_RPT_NGRAM = 3
OUTPUT_DIR   = BASE_DIR / "predict_swin_distilgpt2"
RANDOM_SEED  = np.random.randint(0, 1000)

# ── Hastalilik filtresi ────────────────────────────────────────────────────
DISEASE_ONLY = True   # True: yalnizca hastalikli goruntuleri sec
MIN_DISEASES = 1      # En az kac pozitif patoloji olmali

PATHOLOGY_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]

SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
GPT_MODEL_NAME  = "distilgpt2"

# Swin-B ImageNet normalizasyon istatistikleri (dataset_swin.py ile aynı)
SWIN_MEAN = np.array([0.485, 0.456, 0.406])
SWIN_STD  = np.array([0.229, 0.224, 0.225])

# Template collapse'e neden olan kalıp ifadeler
BAD_PHRASES = [
    "by telephone at the time of dictation",
    "by telephone",
    "findings were recognized",
    "as soon as the findings",
    "were recognized",
]


# ═══════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    print(f"   OK Tokenizer: {GPT_MODEL_NAME} (vocab={tok.vocab_size:,})")
    return tok


def build_bad_words_ids(tokenizer) -> list:
    """BAD_PHRASES listesini token id listesine dönüştürür."""
    bad_words_ids = []
    for phrase in BAD_PHRASES:
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        if ids:
            bad_words_ids.append(ids)
    return bad_words_ids


# ═══════════════════════════════════════════════════════════════════════════
# MODEL YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════
def load_model(checkpoint_path: str, device: torch.device, tokenizer) -> tuple:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint bulunamadi: {checkpoint_path}")

    print(f"\nModel yukleniyor: {checkpoint_path}")

    model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name=SWIN_MODEL_NAME,
        decoder_name=GPT_MODEL_NAME,
        enable_gradient_checkpointing=False,   # inference'ta GC gerekmez
    ).to(device)

    model.gpt.config.pad_token_id = tokenizer.pad_token_id
    model.gpt.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id     = tokenizer.pad_token_id
    model.config.eos_token_id     = tokenizer.eos_token_id

    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("val_loss", ckpt.get("loss", float("nan")))
    phase = ckpt.get("phase", "?")

    n_total = sum(p.numel() for p in model.parameters())
    print(f"   OK Model yuklendi!")
    print(f"      Epoch       : {epoch + 1 if isinstance(epoch, int) else epoch}")
    if isinstance(loss, float) and not np.isnan(loss):
        print(f"      Val Loss    : {loss:.4f}")
    else:
        print(f"      Val Loss    : {loss}")
    print(f"      Faz         : {phase}")
    print(f"      Toplam Param: {n_total:,}")
    return model, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# TAHMİN ÜRETİMİ
# ═══════════════════════════════════════════════════════════════════════════
def generate_report(
    model: SwinDistilGPT2ForMTL,
    image_tensor: torch.Tensor,
    tokenizer,
    bad_words_ids: list,
    beam_size: int = 4,
    max_new_tokens: int = 160,
) -> tuple:
    """
    Tek bir görüntü tensörü için rapor üretir.

    Returns:
        pred_text : üretilen metin
        heatmap   : Swin Stage-4 saliency haritası (veya None)

    NOT — Swin Saliency:
        Swin pencere-tabanlı dikkat kullanır; doğrudan CLS-Attention yoktur.
        Bunun yerine last_hidden_state (Stage-4: 7x7=49 spatial token)
        üzerinden L2-norm saliency hesaplanır ve 224x224'e interpolate edilir.
        Bu, "hangi bölgenin daha güçlü aktive olduğunu" gösteren pratik bir
        görsel alternatiftir.
    """
    import torch.nn.functional as F

    # ── 1. Metin Üretimi ──────────────────────────────────────────────────
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values=image_tensor,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            repetition_penalty=REP_PENALTY,
            no_repeat_ngram_size=NO_RPT_NGRAM,
            do_sample=False,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
        )
    pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if not pred_text:
        pred_text = "(bos tahmin)"

    # ── 2. Swin Stage-4 Saliency Haritası ────────────────────────────────
    # Swin'in last_hidden_state: (1, 49, 1024) — 7×7 spatial grid
    # Her token'ın L2-normu hesaplanır → 7×7 saliency; 224×224'e interpolate
    heatmap = None
    try:
        with torch.no_grad():
            swin_out = model.swin(pixel_values=image_tensor)
        hidden = swin_out.last_hidden_state   # (1, 49, 1024)
        # L2-norm her spatial position için
        saliency = hidden.norm(dim=-1)        # (1, 49)
        B, N     = saliency.shape
        grid     = int(N ** 0.5)             # 7

        if grid * grid == N:
            sal_map = saliency.view(B, 1, grid, grid)   # (1, 1, 7, 7)
            sal_map = F.interpolate(
                sal_map.float(),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bicubic",
                align_corners=False,
            )
            heatmap = sal_map.squeeze().cpu().numpy()
            mn, mx  = heatmap.min(), heatmap.max()
            heatmap = (heatmap - mn) / (mx - mn + 1e-8)
            print("   OK Swin Stage-4 Saliency haritasi olusturuldu.")
        else:
            print(f"   WARN Beklenmeyen spatial token sayisi: {N}")
    except Exception as exc:
        print(f"   WARN Saliency haritasi alinamadi: {exc}")
        heatmap = None

    return pred_text, heatmap


# ═══════════════════════════════════════════════════════════════════════════
# METRİK HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════
def compute_sample_metrics(prediction: str, reference: str) -> dict:
    pred_tok = prediction.lower().split()
    ref_tok  = reference.lower().split()
    smooth   = SmoothingFunction().method1

    if not pred_tok or not ref_tok:
        bleu1 = bleu4 = 0.0
    else:
        bleu1 = sentence_bleu([ref_tok], pred_tok, weights=(1,0,0,0), smoothing_function=smooth)
        bleu4 = sentence_bleu([ref_tok], pred_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    rl_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l   = rl_scorer.score(reference, prediction)["rougeL"].fmeasure

    try:
        cider_scorer = Cider()
        cider_val, _ = cider_scorer.compute_score({0: [reference]}, {0: [prediction]})
        cider_val    = float(cider_val)
    except Exception:
        cider_val = 0.0

    bert_f1   = 0.0
    sbert_sim = 0.0

    if bert_score_fn is not None:
        try:
            _, _, F1 = bert_score_fn(
                [prediction], [reference], lang="en",
                rescale_with_baseline=True, verbose=False,
            )
            bert_f1 = float(F1.mean().item())
        except Exception:
            bert_f1 = 0.0

    if _sbert is not None:
        try:
            e1 = _sbert.encode([prediction], convert_to_tensor=True)
            e2 = _sbert.encode([reference],  convert_to_tensor=True)
            sbert_sim = float(st_util.cos_sim(e1, e2)[0][0].item())
        except Exception:
            sbert_sim = 0.0

    return {
        "BLEU-1":  round(bleu1,     4),
        "BLEU-4":  round(bleu4,     4),
        "ROUGE-L": round(rouge_l,   4),
        "CIDEr":   round(cider_val, 4),
        "BERTSc":  round(bert_f1,   4),
        "SBERT":   round(sbert_sim, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Swin ImageNet normalizasyonunu geri alır."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = SWIN_STD * img + SWIN_MEAN
    return np.clip(img, 0, 1)


def wrap_text(text: str, width: int = 72) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def save_prediction_figure(
    image_tensor: torch.Tensor,
    image_path: str,
    predicted_text: str,
    reference_text: str,
    output_dir: Path,
    sample_num: int,
    epoch_info,
    heatmap: np.ndarray | None = None,
    metrics: dict = None,
    disease_labels: list = None,
) -> Path:
    """
    Karanlik temali 5-satir gorsel:
      Satir 0: Baslik
      Satir 1: Hastalik etiket serit
      Satir 2: [Goruntu | Swin Saliency | Metinler]
      Satir 3: Metrik paneli
      Satir 4: Footer
    """
    disease_labels = disease_labels or []

    TAG_COLORS = [
        "#ff7b72", "#f0a30a", "#56d364", "#58a6ff",
        "#bc8cff", "#ff9e64", "#39d353", "#79c0ff",
        "#ffa657", "#d2a8ff", "#7ee787", "#a5d6ff", "#ffb77e",
    ]

    fig = plt.figure(figsize=(19, 13.5), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
        5, 3, figure=fig,
        width_ratios=[1, 1, 1.5],
        height_ratios=[0.042, 0.038, 0.70, 0.175, 0.04],
        hspace=0.07, wspace=0.08,
        left=0.02, right=0.98, top=0.94, bottom=0.03,
    )

    # ── Baslik ────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        f"Swin-B + DistilGPT-2  |  Ornek {sample_num}/{NUM_SAMPLES}"
        f"  |  Epoch {epoch_info + 1 if isinstance(epoch_info, int) else epoch_info}",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color="#e2e8f0", transform=ax_title.transAxes,
    )

    # ── Hastalik Etiket Seridi ────────────────────────────────────────────
    ax_dis = fig.add_subplot(gs[1, :])
    ax_dis.set_facecolor("#0d1117")
    ax_dis.axis("off")

    if disease_labels:
        n_tags = len(disease_labels)
        for ti, lbl in enumerate(disease_labels):
            col = TAG_COLORS[ti % len(TAG_COLORS)]
            x   = (ti + 0.5) / n_tags
            ax_dis.text(
                x, 0.5, lbl,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#0d1117", transform=ax_dis.transAxes,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=col,
                          alpha=0.92, edgecolor="none"),
            )
    else:
        ax_dis.text(
            0.5, 0.5, "Patoloji etiketi belirsiz",
            ha="center", va="center", fontsize=9,
            color="#8b949e", transform=ax_dis.transAxes,
        )

    # ── Goruntu ───────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[2, 0])
    ax_img.set_facecolor("#161b22")
    ax_img.imshow(denormalize(image_tensor))
    ax_img.axis("off")
    ax_img.set_title(Path(image_path).name, color="#8b949e", fontsize=9, pad=6)

    # ── Swin Saliency Haritasi ────────────────────────────────────────────
    ax_attn = fig.add_subplot(gs[2, 1])
    ax_attn.set_facecolor("#161b22")
    img_np = denormalize(image_tensor)
    if heatmap is not None:
        ax_attn.imshow(img_np)
        im = ax_attn.imshow(heatmap, cmap="inferno", alpha=0.50)
        plt.colorbar(im, ax=ax_attn, fraction=0.035, pad=0.02)
        ax_attn.set_title("Swin Stage-4 Saliency", color="#f0a30a", fontsize=10, pad=6)
    else:
        ax_attn.imshow(img_np)
        ax_attn.text(
            0.5, 0.05, "Saliency haritasi yok",
            ha="center", va="bottom", fontsize=9, color="#ff7b72",
            transform=ax_attn.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", alpha=0.85),
        )
        ax_attn.set_title("Swin Stage-4 Saliency (mevcut degil)",
                          color="#8b949e", fontsize=8, pad=6)
    ax_attn.axis("off")

    # ── Metin Paneli ──────────────────────────────────────────────────────
    ax_text = fig.add_subplot(gs[2, 2])
    ax_text.set_facecolor("#161b22")
    ax_text.axis("off")

    box_ref  = dict(boxstyle="round,pad=0.55", facecolor="#0d2137",
                    alpha=0.9, edgecolor="#2f81f7", linewidth=1.5)
    box_pred = dict(boxstyle="round,pad=0.55", facecolor="#0d2b0d",
                    alpha=0.9, edgecolor="#3fb950", linewidth=1.5)

    ax_text.text(0.03, 0.97, "GERCEK RAPOR",
                 transform=ax_text.transAxes, fontsize=11, fontweight="bold",
                 color="#79c0ff", va="top")
    ax_text.text(0.03, 0.91,
                 wrap_text(reference_text or "(bos referans)"),
                 transform=ax_text.transAxes, fontsize=8.5, color="#c9d1d9",
                 va="top", linespacing=1.55, bbox=box_ref)
    ax_text.text(0.03, 0.46, "MODEL TAHMINI",
                 transform=ax_text.transAxes, fontsize=11, fontweight="bold",
                 color="#56d364", va="top")
    ax_text.text(0.03, 0.40,
                 wrap_text(predicted_text or "(bos tahmin)"),
                 transform=ax_text.transAxes, fontsize=8.5, color="#c9d1d9",
                 va="top", linespacing=1.55, bbox=box_pred)

    # ── Metrik Paneli ─────────────────────────────────────────────────────
    ax_met = fig.add_subplot(gs[3, :])
    ax_met.set_facecolor("#0d1117")
    ax_met.axis("off")

    if metrics:
        metric_keys   = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "BERTSc", "SBERT"]
        metric_colors = ["#58a6ff", "#7c6df0", "#3fb950", "#f0883e", "#bc8cff", "#ff7b72"]
        n = len(metric_keys)
        for idx, (mkey, mcolor) in enumerate(zip(metric_keys, metric_colors)):
            x   = (idx + 0.5) / n
            val = metrics.get(mkey, 0.0)
            ax_met.text(x, 0.76, mkey,
                        ha="center", va="center", fontsize=11.5, fontweight="bold",
                        color=mcolor, transform=ax_met.transAxes)
            ax_met.text(x, 0.28, f"{val:.4f}",
                        ha="center", va="center", fontsize=17, fontweight="bold",
                        color="#e6edf3", transform=ax_met.transAxes,
                        bbox=dict(boxstyle="round,pad=0.38",
                                  facecolor=mcolor, alpha=0.2,
                                  edgecolor=mcolor, linewidth=1.3))

    # ── Footer ────────────────────────────────────────────────────────────
    ax_foot = fig.add_subplot(gs[4, :])
    ax_foot.axis("off")
    ax_foot.text(
        0.5, 0.5,
        f"Swin-base-patch4-window7-224  +  DistilGPT-2  |  Beam={BEAM_SIZE}"
        f"  |  Pen={REP_PENALTY}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="center", va="center", fontsize=8, color="#6e7681",
        transform=ax_foot.transAxes,
    )

    out_path = output_dir / f"sample_{sample_num:02d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Kaydedildi: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("  Swin-B + DistilGPT-2  ·  Rastgele 5 Ornek Tahmin & Gorsellestime")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer     = load_tokenizer()
    bad_words_ids = build_bad_words_ids(tokenizer)
    model, ckpt   = load_model(CHECKPOINT_PATH, DEVICE, tokenizer)
    epoch_info    = ckpt.get("epoch", "?")

    # ── Veri ──────────────────────────────────────────────────────────────
    print(f"\nTest verisi: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["final_report"])
    df = df[df["final_report"].astype(str).str.strip() != ""]
    df = df[df["final_report"].apply(lambda x: len(str(x).split()) >= 8)]
    df = df[~df["final_report"].astype(str).str.strip().str.startswith(".")]
    df = df.reset_index(drop=True)
    print(f"   Toplam gecerli ornek: {len(df):,}")

    # ── Hastalik filtresi ─────────────────────────────────────────────────
    avail_path_cols = [c for c in PATHOLOGY_COLS if c in df.columns and c != "No Finding"]

    if DISEASE_ONLY and avail_path_cols:
        # En az MIN_DISEASES pozitif patoloji AND No Finding != 1
        pos_count = (df[avail_path_cols].fillna(0) == 1.0).sum(axis=1)
        no_finding_mask = df["No Finding"].fillna(0) != 1.0 if "No Finding" in df.columns \
                          else pd.Series(True, index=df.index)
        disease_mask = (pos_count >= MIN_DISEASES) & no_finding_mask
        df_filtered  = df[disease_mask].reset_index(drop=True)
        print(f"   Hastalikli ornek (>={MIN_DISEASES} patoloji, No Finding!=1): {len(df_filtered):,}")
        if len(df_filtered) == 0:
            print("   UYARI: Hastalikli ornek bulunamadi, tum veri kullaniliyor.")
            df_filtered = df
    else:
        df_filtered = df
        print("   Filtre: Tum ornekler (DISEASE_ONLY=False)")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    indices    = random.sample(range(len(df_filtered)), min(NUM_SAMPLES, len(df_filtered)))
    samples_df = df_filtered.iloc[indices].reset_index(drop=True)

    transform = get_eval_transform(IMAGE_SIZE)   # Swin ImageNet normalizasyonu

    # ── Tahmin Döngüsü ────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    all_preds   = []
    all_metrics = []

    for i, (_, row) in enumerate(samples_df.iterrows(), start=1):
        # Hangi hastalıklar pozitif?
        active_diseases = [
            c for c in avail_path_cols
            if c in row and str(row[c]).strip() not in ("", "nan", "0", "0.0", "-1", "-1.0")
            and float(str(row[c]).replace(",", ".")) == 1.0
        ] if avail_path_cols else []
        disease_str = ", ".join(active_diseases) if active_diseases else "Belirsiz"
        print(f"\nOrnek {i}/{NUM_SAMPLES} — {row['image_path']}")
        print(f"   Patolojiler : {disease_str}")

        img_path_full = Path(IMAGE_ROOT) / row["image_path"]
        try:
            pil_img = Image.open(img_path_full).convert("RGB")
        except Exception as e:
            print(f"   WARN Goruntu yuklenemedi ({e}) — siyah goruntu kullaniliyor.")
            pil_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="black")

        image_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        pred_text, heatmap = generate_report(
            model, image_tensor, tokenizer,
            bad_words_ids=bad_words_ids,
            beam_size=BEAM_SIZE,
            max_new_tokens=MAX_NEW_TOKS,
        )
        ref_text = str(row["final_report"]).strip()

        metrics = compute_sample_metrics(pred_text, ref_text)
        all_metrics.append(dict(metrics))
        all_preds.append(pred_text)

        print(f"   Gercek  : {ref_text[:110]}{'...' if len(ref_text) > 110 else ''}")
        print(f"   Tahmin  : {pred_text[:110]}{'...' if len(pred_text) > 110 else ''}")
        print(f"   BLEU-1: {metrics['BLEU-1']:.4f}  |  BLEU-4: {metrics['BLEU-4']:.4f}"
              f"  |  ROUGE-L: {metrics['ROUGE-L']:.4f}")
        print(f"   CIDEr : {metrics['CIDEr']:.4f}  |  BERTSc: {metrics['BERTSc']:.4f}"
              f"  |  SBERT  : {metrics['SBERT']:.4f}")

        save_prediction_figure(
            image_tensor=image_tensor.squeeze(0).cpu(),
            image_path=str(row["image_path"]),
            predicted_text=pred_text,
            reference_text=ref_text,
            output_dir=OUTPUT_DIR,
            sample_num=i,
            epoch_info=epoch_info,
            heatmap=heatmap,
            metrics=metrics,
            disease_labels=active_diseases,
        )

    # ── Ortalama Özet ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ORTALAMA METRIKLER")
    print("=" * 70)
    metric_keys = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "BERTSc", "SBERT"]
    if all_metrics:
        for key in metric_keys:
            avg = np.mean([m.get(key, 0.0) for m in all_metrics])
            print(f"   {key:>8s} : {avg:.4f}")
    print("-" * 70)

    unique = set(all_preds)
    if len(unique) == 1:
        print("UYARI: Tum tahminler ayni (model henüz erken asama).")
    else:
        print(f"OK {len(unique)}/{NUM_SAMPLES} farkli tahmin uretildi.")

    print(f"\nTamamlandi -> {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
