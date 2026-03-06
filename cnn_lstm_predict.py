"""
cnn_lstm_predict.py
===================
CNN-LSTM modelini yükler, test setinden rastgele 5 örnek seçer,
her biri için tahmin üretir ve sonuçları görsel olarak kaydeder.

Çıktı: cnn_lstm_predict/ klasörü altında 5 adet .png dosyası
"""

import os
import sys
import random
import textwrap
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # GUI gerektirmeyen backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms

# ── Proje kök dizinini Python path'e ekle ──────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import Config
from src.data_loader.vocabulary import Vocabulary
from src.data_loader.dataset import MIMICCXRDatasetCNNLSTM
from src.models.cnn_lstm import ImageCaptioningModel
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = Config.CHECKPOINT_FILE
VOCAB_PATH      = Config.VOCAB_PATH
TEST_CSV        = Config.TEST_PROCESSED_CSV
IMAGE_ROOT      = Config.IMAGE_DIR
DEVICE          = Config.DEVICE
NUM_SAMPLES     = 5            # kaç örnek
BEAM_SIZE       = 5            # beam search genişliği
MAX_LEN         = 150          # maksimum token sayısı
OUTPUT_DIR      = BASE_DIR / "cnn_lstm_predict"
RANDOM_SEED     = 42


# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════

def load_vocabulary(vocab_path: str) -> Vocabulary:
    """Kaydedilmiş vocabulary'yi yükle."""
    vocab = Vocabulary()
    vocab.load(vocab_path)
    print(f"✅ Vocabulary yüklendi  |  Boyut: {len(vocab):,} kelime")
    return vocab


def load_model(checkpoint_path: str, vocab_size: int, device) -> ImageCaptioningModel:
    """Checkpoint'ten CNN-LSTM modelini yükle."""
    print(f"\n📥 Model yükleniyor: {checkpoint_path}")

    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=Config.CNN_EMBED_SIZE,
        hidden_size=Config.CNN_HIDDEN_SIZE,
        attention_dim=Config.CNN_ATTENTION_DIM,
        num_layers=2,
        dropout=0.0,           # Inference sırasında dropout kapalı
        freeze_backbone=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("loss", float("nan"))
    if isinstance(val_loss, float):
        print(f"✅ Model yüklendi!  Epoch: {epoch}  |  Val Loss: {val_loss:.4f}")
    else:
        print(f"✅ Model yüklendi!  Epoch: {epoch}")
    return model


def ids_to_text(token_ids, vocab: Vocabulary, stop_at_eos: bool = True) -> str:
    """Token ID listesini okunabilir metne çevirir."""
    special = {
        vocab.word2idx.get("<PAD>", -1),
        vocab.word2idx.get("<SOS>", -1),
    }
    eos_id = vocab.word2idx.get("<EOS>", -1)
    words = []
    for idx in token_ids:
        if stop_at_eos and idx == eos_id:
            break
        if idx in special:
            continue
        words.append(vocab.idx2word.get(idx, "<UNK>"))
    return " ".join(words)


def beam_search(model: ImageCaptioningModel, features: torch.Tensor,
                vocab: Vocabulary, beam_size: int = 5, max_len: int = 150):
    """
    Tek bir görüntü için Beam Search ile tahmin üretir.

    Args:
        features: [1, seq_len, embed_size]
    Returns:
        best_seq: list[int]  — token ID listesi (SOS/EOS hariç)
    """
    device  = features.device
    sos_id  = vocab.word2idx["<SOS>"]
    eos_id  = vocab.word2idx["<EOS>"]
    decoder = model.decoder

    h, c = decoder.init_hidden(features)

    beams     = [(0.0, [sos_id], h, c)]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for score, seq, h_beam, c_beam in beams:
            last_tok = torch.tensor([seq[-1]], device=device)

            emb        = decoder.embedding(last_tok)
            context, _ = decoder.attention(features, h_beam[-1])

            lstm_in    = torch.cat([emb, context], dim=1).unsqueeze(1)
            out, (h_new, c_new) = decoder.lstm(lstm_in, (h_beam, c_beam))
            out        = out.squeeze(1)

            combined   = torch.cat([out, context], dim=1)
            logits     = decoder.fc(combined)
            log_probs  = torch.log_softmax(logits, dim=-1)

            topk_lp, topk_ids = log_probs[0].topk(beam_size)

            for lp, tok in zip(topk_lp, topk_ids):
                tok       = tok.item()
                new_score = score + lp.item()
                new_seq   = seq + [tok]

                if tok == eos_id:
                    normalized = new_score / len(new_seq)
                    completed.append((normalized, new_seq))
                else:
                    new_beams.append((new_score, new_seq, h_new, c_new))

        if not new_beams:
            break

        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        if len(completed) >= beam_size:
            break

    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        best_seq = completed[0][1]
    else:
        beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = beams[0][1]

    # SOS'u at, EOS'tan sonrasını kes
    best_seq = best_seq[1:]
    if eos_id in best_seq:
        best_seq = best_seq[:best_seq.index(eos_id)]

    return best_seq


def get_image_transform():
    """Model için gerekli görüntü dönüşümleri (augmentation yok)."""
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Normalize edilmiş tensörü görüntülenebilir numpy dizisine çevir."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return img


def wrap_text(text: str, width: int = 80) -> str:
    """Uzun metni belirli genişlikte satırlara böl."""
    return "\n".join(textwrap.wrap(text, width=width))


# ═══════════════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════

def save_prediction_figure(
    sample_idx: int,
    image_tensor: torch.Tensor,
    image_path: str,
    predicted_text: str,
    reference_text: str,
    output_dir: Path,
    sample_num: int,
):
    """
    Tek bir örnek için görsel oluştur ve kaydet.

    Düzen:
      ┌──────────────────────────────────────────┐
      │  Başlık (Örnek #N)                       │
      ├──────────────┬───────────────────────────┤
      │              │  📋 GERÇEK RAPOR           │
      │  Röntgen     │  ...                       │
      │  Görüntüsü   ├───────────────────────────┤
      │              │  🤖 MODEL TAHMİNİ          │
      │              │  ...                       │
      └──────────────┴───────────────────────────┘
    """
    fig = plt.figure(figsize=(18, 10), facecolor="#0f1117")

    # Izgara: 1 satır × 2 sütun  (sol: görüntü, sağ: metinler)
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[1, 1.4],
        height_ratios=[0.06, 1, 0.05],
        hspace=0.0,
        wspace=0.05,
        left=0.04, right=0.97,
        top=0.93, bottom=0.04
    )

    # ── Başlık ──────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        f"CNN-LSTM  ·  Örnek {sample_num} / {NUM_SAMPLES}",
        ha="center", va="center",
        fontsize=16, fontweight="bold",
        color="#e2e8f0",
        transform=ax_title.transAxes
    )

    # ── Röntgen görüntüsü ───────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.set_facecolor("#1a1d2e")
    ax_img.imshow(denormalize(image_tensor))
    ax_img.axis("off")
    ax_img.set_title(
        f"📷  {Path(image_path).name}",
        color="#94a3b8", fontsize=9, pad=6
    )

    # ── Metin alanı (gerçek + tahmin) ───────────────────────────────────
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.set_facecolor("#1a1d2e")
    ax_text.axis("off")

    # Metin kutusu içeriğini oluştur
    box_props_ref  = dict(boxstyle="round,pad=0.6", facecolor="#1e3a5f", alpha=0.9, edgecolor="#3b82f6", linewidth=1.5)
    box_props_pred = dict(boxstyle="round,pad=0.6", facecolor="#1e4a2e", alpha=0.9, edgecolor="#22c55e", linewidth=1.5)

    wrapped_ref  = wrap_text(reference_text,  width=70)
    wrapped_pred = wrap_text(predicted_text,  width=70)

    # Gerçek rapor
    ax_text.text(
        0.03, 0.97,
        "📋  GERÇEK RAPOR",
        transform=ax_text.transAxes,
        fontsize=11, fontweight="bold",
        color="#93c5fd", va="top"
    )
    ax_text.text(
        0.03, 0.91,
        wrapped_ref if wrapped_ref.strip() else "(boş referans)",
        transform=ax_text.transAxes,
        fontsize=8.5, color="#cbd5e1",
        va="top", linespacing=1.55,
        wrap=True,
        bbox=box_props_ref
    )

    # Model tahmini
    ax_text.text(
        0.03, 0.46,
        "🤖  MODEL TAHMİNİ",
        transform=ax_text.transAxes,
        fontsize=11, fontweight="bold",
        color="#86efac", va="top"
    )
    ax_text.text(
        0.03, 0.40,
        wrapped_pred if wrapped_pred.strip() else "(boş tahmin)",
        transform=ax_text.transAxes,
        fontsize=8.5, color="#cbd5e1",
        va="top", linespacing=1.55,
        wrap=True,
        bbox=box_props_pred
    )

    # ── Alt bilgi ───────────────────────────────────────────────────────
    ax_footer = fig.add_subplot(gs[2, :])
    ax_footer.axis("off")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax_footer.text(
        0.5, 0.5,
        f"DenseNet-121  ←→  LSTM + Attention  ·  Beam Size: {BEAM_SIZE}  ·  {timestamp}",
        ha="center", va="center",
        fontsize=8, color="#64748b",
        transform=ax_footer.transAxes
    )

    # ── Kaydet ──────────────────────────────────────────────────────────
    output_path = output_dir / f"sample_{sample_num:02d}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Görsel kaydedildi: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# ANA FONKSİYON
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 70)
    print("  CNN-LSTM  ·  Rastgele 5 Örnek Tahmin & Görselleştirme")
    print("═" * 70)
    print(f"  Cihaz     : {DEVICE}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Test CSV  : {TEST_CSV}")
    print(f"  Çıktı     : {OUTPUT_DIR}")
    print("═" * 70 + "\n")

    # ── Çıktı klasörü ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Vocabulary ──────────────────────────────────────────────────────
    vocab = load_vocabulary(VOCAB_PATH)

    # ── Model ───────────────────────────────────────────────────────────
    model = load_model(CHECKPOINT_PATH, len(vocab), DEVICE)

    # ── Veri: test CSV'den rastgele 5 satır ─────────────────────────────
    print(f"\n📂 Test verisi okunuyor: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["final_report"])
    df = df[df["final_report"].astype(str).str.strip() != ""]
    print(f"   Toplam geçerli test örneği: {len(df):,}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    sample_indices = random.sample(range(len(df)), min(NUM_SAMPLES, len(df)))
    samples_df = df.iloc[sample_indices].reset_index(drop=True)
    print(f"   Seçilen {len(samples_df)} rastgele örnek.\n")

    # ── Görüntü dönüşümü ────────────────────────────────────────────────
    transform = get_image_transform()

    # ── Her örnek için tahmin üret ──────────────────────────────────────
    print("─" * 70)
    for i, (_, row) in enumerate(samples_df.iterrows(), start=1):
        print(f"\n🔍 Örnek {i}/{NUM_SAMPLES}")

        img_path_full = Path(IMAGE_ROOT) / row["image_path"]

        # Görüntüyü yükle
        try:
            pil_img = Image.open(img_path_full).convert("RGB")
        except Exception as e:
            print(f"   ⚠️  Görüntü yüklenemedi: {img_path_full}  →  {e}")
            pil_img = Image.new("RGB", (512, 512), color="black")

        image_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

        # Encoder → özellikler
        with torch.no_grad():
            features = model.encoder(image_tensor)  # [1, seq_len, embed_size]

            # Beam Search ile tahmin
            pred_ids   = beam_search(model, features, vocab,
                                     beam_size=BEAM_SIZE, max_len=MAX_LEN)

        pred_text = ids_to_text(pred_ids, vocab)
        ref_text  = str(row["final_report"]).strip()

        print(f"   📷  Görüntü : {row['image_path']}")
        print(f"   📋  Gerçek  : {ref_text[:120]}{'...' if len(ref_text) > 120 else ''}")
        print(f"   🤖  Tahmin  : {pred_text[:120]}{'...' if len(pred_text) > 120 else ''}")

        # Görsel kaydet (image_tensor'u CPU'ya al ve batch boyutunu kaldır)
        save_prediction_figure(
            sample_idx    = i - 1,
            image_tensor  = image_tensor.squeeze(0).cpu(),
            image_path    = str(row["image_path"]),
            predicted_text= pred_text,
            reference_text= ref_text,
            output_dir    = OUTPUT_DIR,
            sample_num    = i,
        )

    # ── Özet ────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"✅ Tamamlandı!  {NUM_SAMPLES} görsel kaydedildi → {OUTPUT_DIR}")
    saved_files = sorted(OUTPUT_DIR.glob("sample_*.png"))
    for f in saved_files:
        print(f"   • {f.name}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
