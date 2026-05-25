"""
demo_presentation.py
====================
Sunum demosı: CNN-LSTM (DenseNet-121+LSTM+Attention) vs Swin-DistilGPT2
10 kalici goruntu secilir, her ikisinden tahmin + isi haritasi uretilir.
Ciktı: demo_presentation/ klasorunde ornek_01.png ... ornek_10.png
"""

import os
import sys
import json
import random
import textwrap
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image
from torchvision import transforms
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import Config
from src.data_loader.vocabulary import Vocabulary
from src.models.cnn_lstm import ImageCaptioningModel
from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL
from transformers import AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 2024          # Sabit seed → her seferinde aynı 10 goruntu
NUM_SAMPLES = 10
OUTPUT_DIR  = BASE_DIR / "demo_presentation"

CNN_CKPT    = BASE_DIR / "checkpoints_densenet_findings" / "best_model.pth"
VOCAB_PATH  = Config.VOCAB_PATH
CNN_MAX_LEN = 150

SWIN_CKPT   = BASE_DIR / "checkpoints_swin_distilgpt2" / "best_model_swin_distilgpt2.pth"
SWIN_CKPT2  = BASE_DIR / "checkpoints_swin_distilgpt2" / "swa_model_final.pth"
SWIN_BEAM   = Config.SWIN_NUM_BEAMS
SWIN_MAX    = 160
REP_PENALTY = 2.0
NO_RPT_NGRAM = 3

TEST_CSV    = Config.TEST_PROCESSED_CSV
IMAGE_DIR   = Config.IMAGE_DIR
DEVICE      = Config.DEVICE

# CNN-LSTM transform (512x512, egitimle ayni)
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Swin transform (224x224)
SWIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════════════
# MODEL YUKLEME
# ═══════════════════════════════════════════════════════════════════════════

def load_cnn_lstm():
    print(f"  Checkpoint: {CNN_CKPT}")
    if not CNN_CKPT.exists():
        raise FileNotFoundError(f"CNN-LSTM checkpoint bulunamadi: {CNN_CKPT}")

    vocab = Vocabulary()
    vocab.load(str(VOCAB_PATH))
    print(f"  Vocabulary: {len(vocab):,} kelime")

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_size=Config.CNN_EMBED_SIZE,
        hidden_size=Config.CNN_HIDDEN_SIZE,
        attention_dim=Config.CNN_ATTENTION_DIM,
        num_layers=2,
        dropout=0.0,
        freeze_backbone=False,
    ).to(DEVICE)

    ckpt = torch.load(str(CNN_CKPT), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"  Epoch: {epoch} | Yuklendi!")
    return model, vocab


def load_swin():
    ckpt_path = SWIN_CKPT if SWIN_CKPT.exists() else SWIN_CKPT2
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Swin checkpoint bulunamadi: {SWIN_CKPT} / {SWIN_CKPT2}")
    print(f"  Checkpoint: {ckpt_path.name}")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name=Config.SWIN_ENCODER,
        decoder_name=Config.SWIN_DECODER,
        num_classes=14,
        num_prefix_tokens=Config.SWIN_NUM_PREFIX_TOKENS,
        proj_dropout=0.0,
        enable_gradient_checkpointing=False,
    ).to(DEVICE)

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("  Yuklendi!")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def ids_to_text(cap_ids, vocab):
    eos_id = vocab.word2idx.get("<EOS>", -1)
    pad_id = vocab.word2idx.get("<PAD>", -1)
    sos_id = vocab.word2idx.get("<SOS>", -1)
    words  = []
    for idx in cap_ids:
        if idx == eos_id:
            break
        if idx in (pad_id, sos_id):
            continue
        words.append(vocab.idx2word.get(idx, "<UNK>"))
    return " ".join(words)


def cnn_predict(model, vocab, cnn_tensor):
    """Greedy decode → metin + attention heatmap."""
    with torch.no_grad():
        captions, alphas_list = model.generate(cnn_tensor, vocab, max_len=CNN_MAX_LEN)

    pred_text = ids_to_text(captions[0], vocab)

    heatmap = None
    if alphas_list:
        # alphas_list: list of [B, seq_len] tensors (greedy, 1 per timestep)
        stacked = torch.stack(alphas_list, dim=0)   # [T, B, seq_len]
        avg_alpha = stacked[:, 0, :].mean(dim=0).cpu().float()  # [seq_len]
        seq_len = avg_alpha.shape[0]
        grid = int(seq_len ** 0.5)
        if grid * grid == seq_len:
            h = avg_alpha.reshape(1, 1, grid, grid)
            h = F.interpolate(h, size=(224, 224), mode="bicubic", align_corners=False)
            heatmap = h.squeeze().numpy()
            mn, mx = heatmap.min(), heatmap.max()
            heatmap = (heatmap - mn) / (mx - mn + 1e-8)

    return pred_text, heatmap


def swin_predict(model, tokenizer, swin_tensor):
    """Beam search → metin + saliency heatmap."""
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values=swin_tensor,
            max_new_tokens=SWIN_MAX,
            num_beams=SWIN_BEAM,
            repetition_penalty=REP_PENALTY,
            no_repeat_ngram_size=NO_RPT_NGRAM,
            do_sample=False,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    pred_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if not pred_text:
        pred_text = "(bos tahmin)"

    # Swin Stage-4 Saliency: L2-norm of last_hidden_state (7x7=49 tokens)
    heatmap = None
    try:
        with torch.no_grad():
            swin_out = model.swin(pixel_values=swin_tensor)
        hidden   = swin_out.last_hidden_state   # (1, 49, 1024)
        saliency = hidden.norm(dim=-1)           # (1, 49)
        B, N = saliency.shape
        grid = int(N ** 0.5)
        if grid * grid == N:
            sal = saliency.view(B, 1, grid, grid)
            sal = F.interpolate(sal.float(), size=(224, 224), mode="bicubic", align_corners=False)
            heatmap = sal.squeeze().cpu().numpy()
            mn, mx = heatmap.min(), heatmap.max()
            heatmap = (heatmap - mn) / (mx - mn + 1e-8)
    except Exception as e:
        print(f"    WARN Swin heatmap hatasi: {e}")

    return pred_text, heatmap


# ═══════════════════════════════════════════════════════════════════════════
# GORSELLIK
# ═══════════════════════════════════════════════════════════════════════════

def denormalize(tensor_chw, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize edilmis tensoru [0,1] numpy goruntusune donustur."""
    m = np.array(mean)
    s = np.array(std)
    img = tensor_chw.cpu().numpy().transpose(1, 2, 0)
    img = s * img + m
    return np.clip(img, 0, 1)


def wrap(text, width=65):
    return "\n".join(textwrap.wrap(text.strip(), width=width)) or "(bos)"


def draw_heatmap_panel(ax, base_img_224, heatmap, title, cmap, label):
    """Goruntu + overlay heatmap ciz."""
    ax.set_facecolor("#161b22")
    ax.imshow(base_img_224)
    if heatmap is not None:
        ax.imshow(heatmap, cmap=cmap, alpha=0.50, vmin=0, vmax=1)
    ax.set_title(title, color="#94a3b8", fontsize=8.5, pad=5)
    ax.text(0.01, 0.01, label, transform=ax.transAxes,
            fontsize=7, color="#f8fafc", va="bottom",
            bbox=dict(facecolor="#00000088", edgecolor="none", pad=2))
    ax.axis("off")


def draw_text_panel(ax, header, text, header_color, text_color, bg_color, edge_color):
    ax.set_facecolor(bg_color)
    ax.axis("off")
    ax.text(0.015, 0.97, header, transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=header_color, va="top")
    ax.text(0.015, 0.83, wrap(text, width=120), transform=ax.transAxes,
            fontsize=7.5, color=text_color, va="top", linespacing=1.5,
            fontfamily="monospace")


def save_sample_figure(sample_num, cnn_tensor, swin_tensor, image_path,
                        ref_text, cnn_pred, cnn_heatmap, swin_pred, swin_heatmap):
    """
    Duzen (5 satirlik GridSpec):
      [0,:]   Baslik
      [1,0]   Orijinal goruntu
      [1,1]   CNN-LSTM attention overlay
      [1,2]   Swin saliency overlay
      [2,:]   Gercek rapor
      [3,:2]  CNN-LSTM tahmini
      [3,2]   Swin tahmini
      [4,:]   Footer
    """
    fig = plt.figure(figsize=(20, 13), facecolor="#0d1117")

    gs = gridspec.GridSpec(
        5, 3,
        figure=fig,
        height_ratios=[0.045, 1.05, 0.40, 0.40, 0.035],
        hspace=0.07, wspace=0.05,
        left=0.015, right=0.985, top=0.97, bottom=0.025,
    )

    # ── Baslik ──────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.set_facecolor("#161b22")
    ax_title.text(
        0.5, 0.5,
        f"Tibbi Goruntu Rapor Olusturma  ·  "
        f"CNN-LSTM (DenseNet-121+LSTM)  vs  Swin-DistilGPT2  ·  "
        f"Ornek {sample_num} / {NUM_SAMPLES}",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#f1f5f9", transform=ax_title.transAxes,
    )

    # ── Goruntular satiri ────────────────────────────────────────────────────
    # Orijinal goruntu (CNN tensor'undan denormalize, 224'e resize)
    base_big  = denormalize(cnn_tensor.squeeze(0))          # HxW=512 numpy
    pil_orig  = Image.fromarray((base_big * 255).astype(np.uint8))
    base_224  = np.array(pil_orig.resize((224, 224))) / 255.0  # 224x224 numpy

    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.set_facecolor("#161b22")
    ax_orig.imshow(base_big)
    ax_orig.set_title(
        f"Orijinal X-Ray\n{Path(image_path).name}",
        color="#94a3b8", fontsize=8.5, pad=5,
    )
    ax_orig.axis("off")

    ax_cnn_h = fig.add_subplot(gs[1, 1])
    draw_heatmap_panel(
        ax_cnn_h, base_224, cnn_heatmap,
        "CNN-LSTM  –  Attention Haritasi",
        cmap="hot",
        label="DenseNet-121 spatial attention",
    )

    ax_swin_h = fig.add_subplot(gs[1, 2])
    draw_heatmap_panel(
        ax_swin_h, base_224, swin_heatmap,
        "Swin-B  –  Saliency Haritasi",
        cmap="plasma",
        label="Stage-4 L2-norm (7×7 → 224×224)",
    )

    # ── Gercek rapor ────────────────────────────────────────────────────────
    ax_ref = fig.add_subplot(gs[2, :])
    draw_text_panel(
        ax_ref,
        header       = "GERCEK RAPOR (Referans Annotation):",
        text         = ref_text,
        header_color = "#60a5fa",
        text_color   = "#cbd5e1",
        bg_color     = "#0f172a",
        edge_color   = "#1e40af",
    )

    # ── Tahminler ───────────────────────────────────────────────────────────
    ax_cnn_t = fig.add_subplot(gs[3, :2])
    draw_text_panel(
        ax_cnn_t,
        header       = "CNN-LSTM TAHMINI  (DenseNet-121 + LSTM + Additive Attention):",
        text         = cnn_pred,
        header_color = "#4ade80",
        text_color   = "#bbf7d0",
        bg_color     = "#071a0c",
        edge_color   = "#166534",
    )

    ax_swin_t = fig.add_subplot(gs[3, 2])
    draw_text_panel(
        ax_swin_t,
        header       = "SWIN-DISTILGPT2 TAHMINI:",
        text         = swin_pred,
        header_color = "#c084fc",
        text_color   = "#e9d5ff",
        bg_color     = "#0d0a1e",
        edge_color   = "#6b21a8",
    )

    # ── Footer ──────────────────────────────────────────────────────────────
    ax_foot = fig.add_subplot(gs[4, :])
    ax_foot.axis("off")
    ax_foot.text(
        0.5, 0.5,
        f"Bitirme Projesi  ·  Medikal Goruntu Rapor Olusturma  ·  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="center", va="center", fontsize=7.5, color="#475569",
        transform=ax_foot.transAxes,
    )

    out_path = OUTPUT_DIR / f"ornek_{sample_num:02d}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Kaydedildi: {out_path.name}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# ORNEK SECIMI (KALICI)
# ═══════════════════════════════════════════════════════════════════════════

def select_samples(df):
    """
    Ilk calistirmada: sabit seed ile 10 satir sec, JSON'a kaydet.
    Sonraki calistirmalarda: JSON'dan yukle (ayni resimler).
    """
    selection_file = OUTPUT_DIR / "selected_indices.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if selection_file.exists():
        with open(selection_file) as f:
            indices = json.load(f)
        print(f"  Kalici secim yuklendi ({len(indices)} ornek): {selection_file.name}")
    else:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        indices = sorted(random.sample(range(len(df)), min(NUM_SAMPLES, len(df))))
        with open(selection_file, "w") as f:
            json.dump(indices, f, indent=2)
        print(f"  Yeni secim kaydedildi ({len(indices)} ornek): {selection_file.name}")

    return df.iloc[indices].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# ANA PROGRAM
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  SUNUM DEMO  |  CNN-LSTM  vs  Swin-DistilGPT2")
    print(f"  Cihaz : {DEVICE}  |  {NUM_SAMPLES} goruntu  |  Cikti: {OUTPUT_DIR.name}/")
    print("=" * 70)

    # Veri
    print(f"\n[Veri] {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["final_report"])
    df = df[df["final_report"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"  Gecerli test ornegi: {len(df):,}")
    samples = select_samples(df)

    # Modeller
    print("\n[CNN-LSTM yükleniyor...]")
    cnn_model, vocab = load_cnn_lstm()

    print("\n[Swin-DistilGPT2 yükleniyor...]")
    swin_model, tokenizer = load_swin()

    # Her örnek
    print(f"\n{'─' * 70}")
    saved = []
    for i, row in samples.iterrows():
        sample_num = i + 1
        print(f"\nOrnek {sample_num}/{NUM_SAMPLES} | {row['image_path']}")

        img_path = Path(IMAGE_DIR) / row["image_path"]
        try:
            pil = Image.open(str(img_path)).convert("RGB")
        except Exception as e:
            print(f"  WARN goruntu yuklenemedi: {e} — siyah goruntu kullaniliyor")
            pil = Image.new("RGB", (512, 512), "black")

        cnn_tensor  = CNN_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
        swin_tensor = SWIN_TRANSFORM(pil).unsqueeze(0).to(DEVICE)

        # CNN-LSTM
        cnn_pred, cnn_heat = cnn_predict(cnn_model, vocab, cnn_tensor)
        print(f"  CNN-LSTM : {cnn_pred[:110]}{'...' if len(cnn_pred) > 110 else ''}")

        # Swin
        swin_pred, swin_heat = swin_predict(swin_model, tokenizer, swin_tensor)
        print(f"  Swin-GPT2: {swin_pred[:110]}{'...' if len(swin_pred) > 110 else ''}")

        ref_text = str(row["final_report"]).strip()

        out = save_sample_figure(
            sample_num  = sample_num,
            cnn_tensor  = cnn_tensor,
            swin_tensor = swin_tensor,
            image_path  = str(row["image_path"]),
            ref_text    = ref_text,
            cnn_pred    = cnn_pred,
            cnn_heatmap = cnn_heat,
            swin_pred   = swin_pred,
            swin_heatmap= swin_heat,
        )
        saved.append(out)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Özet
    print(f"\n{'=' * 70}")
    print(f"  TAMAMLANDI!  {len(saved)} PNG -> {OUTPUT_DIR}")
    for f in saved:
        print(f"    {f.name}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
