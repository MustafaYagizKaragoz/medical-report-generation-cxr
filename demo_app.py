"""
demo_app.py
===========
Gradio sunum uygulamasi — CNN-LSTM vs Swin-DistilGPT2
Startup'ta 10 goruntu precompute edilir, tikla → aninda goster.

Calistir:
    python demo_app.py
Tarayicida ac:
    http://127.0.0.1:7860
"""

import sys
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from matplotlib import colormaps as mpl_cm
import gradio as gr

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
RANDOM_SEED  = 2024
NUM_SAMPLES  = 10
SELECTION_FILE = BASE_DIR / "demo_presentation" / "selected_indices.json"

CNN_CKPT     = BASE_DIR / "checkpoints_densenet_findings" / "best_model.pth"
VOCAB_PATH   = Config.VOCAB_PATH
CNN_MAX_LEN  = 150

SWIN_CKPT    = BASE_DIR / "checkpoints_swin_distilgpt2" / "best_model_swin_distilgpt2.pth"
SWIN_CKPT2   = BASE_DIR / "checkpoints_swin_distilgpt2" / "swa_model_final.pth"
SWIN_BEAM    = Config.SWIN_NUM_BEAMS
SWIN_MAX     = 160
REP_PENALTY  = 2.0
NO_RPT_NGRAM = 3

TEST_CSV     = Config.TEST_PROCESSED_CSV
IMAGE_DIR    = Config.IMAGE_DIR
DEVICE       = Config.DEVICE

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
SWIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════════════════════════════════════
# MODEL YUKLEME
# ═══════════════════════════════════════════════════════════════════════════

def load_cnn_lstm():
    vocab = Vocabulary()
    vocab.load(str(VOCAB_PATH))

    model = ImageCaptioningModel(
        vocab_size    = len(vocab),
        embed_size    = Config.CNN_EMBED_SIZE,
        hidden_size   = Config.CNN_HIDDEN_SIZE,
        attention_dim = Config.CNN_ATTENTION_DIM,
        num_layers    = 2,
        dropout       = 0.0,
        freeze_backbone = False,
    ).to(DEVICE)

    ckpt = torch.load(str(CNN_CKPT), map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    print(f"  CNN-LSTM yuklendi (epoch={ckpt.get('epoch', '?')}, vocab={len(vocab):,})")
    return model, vocab


def load_swin():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = SWIN_CKPT if SWIN_CKPT.exists() else SWIN_CKPT2

    model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name  = Config.SWIN_ENCODER,
        decoder_name  = Config.SWIN_DECODER,
        num_classes   = 14,
        num_prefix_tokens = Config.SWIN_NUM_PREFIX_TOKENS,
        proj_dropout  = 0.0,
        enable_gradient_checkpointing = False,
    ).to(DEVICE)

    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Swin-DistilGPT2 yuklendi ({ckpt_path.name})")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def ids_to_text(cap_ids, vocab):
    eos = vocab.word2idx.get("<EOS>", -1)
    pad = vocab.word2idx.get("<PAD>", -1)
    sos = vocab.word2idx.get("<SOS>", -1)
    words = []
    for idx in cap_ids:
        if idx == eos:
            break
        if idx in (pad, sos):
            continue
        words.append(vocab.idx2word.get(idx, "<UNK>"))
    return " ".join(words)


def cnn_infer(model, vocab, cnn_tensor):
    with torch.no_grad():
        captions, alphas_list = model.generate(cnn_tensor, vocab, max_len=CNN_MAX_LEN)

    text = ids_to_text(captions[0], vocab)

    heatmap = None
    if alphas_list:
        stacked   = torch.stack(alphas_list, dim=0)           # [T, B, seq_len]
        avg_alpha = stacked[:, 0, :].mean(dim=0).cpu().float()
        seq_len   = avg_alpha.shape[0]
        grid      = int(seq_len ** 0.5)
        if grid * grid == seq_len:
            h = avg_alpha.reshape(1, 1, grid, grid)
            h = F.interpolate(h, size=(224, 224), mode="bicubic", align_corners=False)
            heatmap = h.squeeze().numpy()
            mn, mx  = heatmap.min(), heatmap.max()
            heatmap = (heatmap - mn) / (mx - mn + 1e-8)

    return text, heatmap


def swin_infer(model, tokenizer, swin_tensor):
    with torch.no_grad():
        gen_ids = model.generate(
            pixel_values      = swin_tensor,
            max_new_tokens    = SWIN_MAX,
            num_beams         = SWIN_BEAM,
            repetition_penalty= REP_PENALTY,
            no_repeat_ngram_size = NO_RPT_NGRAM,
            do_sample         = False,
            early_stopping    = True,
            eos_token_id      = tokenizer.eos_token_id,
            pad_token_id      = tokenizer.pad_token_id,
        )

    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if not text:
        text = "(bos tahmin)"

    heatmap = None
    try:
        with torch.no_grad():
            swin_out = model.swin(pixel_values=swin_tensor)
        hidden   = swin_out.last_hidden_state          # (1, 49, 1024)
        saliency = hidden.norm(dim=-1)                  # (1, 49)
        B, N     = saliency.shape
        grid     = int(N ** 0.5)
        if grid * grid == N:
            sal = saliency.view(B, 1, grid, grid)
            sal = F.interpolate(sal.float(), size=(224, 224), mode="bicubic", align_corners=False)
            heatmap = sal.squeeze().cpu().numpy()
            mn, mx  = heatmap.min(), heatmap.max()
            heatmap = (heatmap - mn) / (mx - mn + 1e-8)
    except Exception as e:
        print(f"  WARN swin heatmap: {e}")

    return text, heatmap


# ═══════════════════════════════════════════════════════════════════════════
# GORSEL ISLEMLER
# ═══════════════════════════════════════════════════════════════════════════

def denormalize(tensor_chw, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    m   = np.array(mean)
    s   = np.array(std)
    img = tensor_chw.cpu().numpy().transpose(1, 2, 0)
    return np.clip(s * img + m, 0, 1)


def make_overlay(base_np_224, heatmap, cmap_name="hot", alpha=0.50):
    """
    base_np_224: (224, 224, 3) float [0,1]
    heatmap    : (224, 224)    float [0,1]  veya None
    Dondurur   : (224, 224, 3) uint8 RGB PIL Image
    """
    base_uint8 = (base_np_224 * 255).astype(np.uint8)
    if heatmap is None:
        return Image.fromarray(base_uint8)

    cmap      = mpl_cm[cmap_name]
    heat_rgba = (cmap(heatmap) * 255).astype(np.uint8)[:, :, :3]  # (224,224,3)
    blended   = (
        (1 - alpha) * base_uint8.astype(np.float32) +
        alpha       * heat_rgba.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def pil_resize_224(pil_img):
    return pil_img.resize((224, 224), Image.LANCZOS)


# ═══════════════════════════════════════════════════════════════════════════
# PRECOMPUTE — TUM ORNEKLER STARTUP'TA HESAPLANIR
# ═══════════════════════════════════════════════════════════════════════════

# Global cache: list of 10 result dicts
CACHE = []


def select_samples(df):
    """Kalici secim: JSON varsa yukle, yoksa yeni olustur."""
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    if SELECTION_FILE.exists():
        with open(SELECTION_FILE) as f:
            indices = json.load(f)
        print(f"  Kalici secim yuklendi: {SELECTION_FILE.name}")
    else:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        indices = sorted(random.sample(range(len(df)), min(NUM_SAMPLES, len(df))))
        with open(SELECTION_FILE, "w") as f:
            json.dump(indices, f, indent=2)
        print(f"  Yeni secim kaydedildi: {SELECTION_FILE.name}")
    return df.iloc[indices].reset_index(drop=True)


def precompute_all():
    global CACHE

    print("\n" + "=" * 60)
    print("  SUNUM DEMO BASLIYOR — precompute asamasi")
    print("=" * 60)

    # Veri yukle
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["final_report"])
    df = df[df["final_report"].astype(str).str.strip() != ""].reset_index(drop=True)
    samples = select_samples(df)

    # Modeller
    print("\n[1/2] CNN-LSTM yukleniyor...")
    cnn_model, vocab = load_cnn_lstm()

    print("\n[2/2] Swin-DistilGPT2 yukleniyor...")
    swin_model, tokenizer = load_swin()

    print(f"\n  Cihaz: {DEVICE} | {NUM_SAMPLES} ornek isleniyor...\n")
    print("-" * 60)

    for i, row in samples.iterrows():
        n = i + 1
        print(f"  [{n}/{NUM_SAMPLES}] {row['image_path']}")

        img_path = Path(IMAGE_DIR) / row["image_path"]
        try:
            pil = Image.open(str(img_path)).convert("RGB")
        except Exception as e:
            print(f"    WARN: {e} — siyah goruntu")
            pil = Image.new("RGB", (512, 512), "black")

        cnn_tensor  = CNN_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
        swin_tensor = SWIN_TRANSFORM(pil).unsqueeze(0).to(DEVICE)

        # Orijinal goruntu (512'den 224'e)
        orig_big    = denormalize(cnn_tensor.squeeze(0))
        pil_orig    = Image.fromarray((orig_big * 255).astype(np.uint8))
        base_224_np = np.array(pil_resize_224(pil_orig)) / 255.0   # (224,224,3) float

        # CNN-LSTM
        cnn_text, cnn_heat = cnn_infer(cnn_model, vocab, cnn_tensor)
        print(f"    CNN: {cnn_text[:80]}...")

        # Swin
        swin_text, swin_heat = swin_infer(swin_model, tokenizer, swin_tensor)
        print(f"    Swin: {swin_text[:80]}...")

        # Overlay gorseller (PIL Image)
        img_orig     = pil_resize_224(pil_orig)
        img_cnn_ov   = make_overlay(base_224_np, cnn_heat,  cmap_name="hot",    alpha=0.50)
        img_swin_ov  = make_overlay(base_224_np, swin_heat, cmap_name="plasma",  alpha=0.50)

        CACHE.append({
            "title"     : f"Örnek {n}  —  {Path(row['image_path']).name}",
            "orig"      : img_orig,
            "cnn_ov"    : img_cnn_ov,
            "swin_ov"   : img_swin_ov,
            "ref"       : str(row["final_report"]).strip(),
            "cnn_pred"  : cnn_text,
            "swin_pred" : swin_text,
        })

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("-" * 60)
    print(f"  Tamamlandi! {len(CACHE)} ornek hazir.\n")


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO ARAYUZ
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
#title-md h1 { text-align: center; color: #a78bfa; }
#title-md h3 { text-align: center; color: #94a3b8; margin-top: -8px; }
.sample-btn { min-width: 64px !important; }
#ref-box textarea, #cnn-box textarea, #swin-box textarea {
    font-family: monospace; font-size: 13px; line-height: 1.55;
}
"""

def show_sample(idx):
    r = CACHE[idx]
    return (
        r["orig"],
        r["cnn_ov"],
        r["swin_ov"],
        r["ref"],
        r["cnn_pred"],
        r["swin_pred"],
        r["title"],
    )


def build_ui():
    choices = [f"Örnek {i+1}" for i in range(NUM_SAMPLES)]

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        title="Medikal Rapor Demo",
        css=CSS,
    ) as demo:

        # Baslik
        gr.Markdown(
            """# Medikal Görüntü Rapor Oluşturma
### CNN-LSTM (DenseNet-121 + LSTM + Attention)  ·  vs  ·  Swin-B + DistilGPT-2""",
            elem_id="title-md",
        )

        # Ornek secici (radio butonlar, yatay)
        with gr.Row():
            sample_radio = gr.Radio(
                choices   = choices,
                value     = choices[0],
                label     = "Görüntü Seçin",
                elem_classes = ["sample-btn"],
                interactive  = True,
            )

        # Secilen ornek basligi
        sample_title = gr.Markdown("", elem_id="sample-title")

        # Goruntu ustasi
        with gr.Row(equal_height=True):
            img_orig = gr.Image(
                label    = "Orijinal X-Ray",
                height   = 280,
                show_download_button = False,
            )
            img_cnn = gr.Image(
                label    = "CNN-LSTM  –  Attention Haritasi (hot)",
                height   = 280,
                show_download_button = False,
            )
            img_swin = gr.Image(
                label    = "Swin-B  –  Saliency Haritasi (plasma)",
                height   = 280,
                show_download_button = False,
            )

        gr.Markdown("---")

        # Metinler
        ref_box = gr.Textbox(
            label    = "Gerçek Rapor (Referans Annotation)",
            lines    = 4,
            max_lines= 8,
            elem_id  = "ref-box",
            show_copy_button = True,
            interactive      = False,
        )

        with gr.Row():
            cnn_box = gr.Textbox(
                label    = "CNN-LSTM Tahmini  (DenseNet-121 + LSTM + Additive Attention)",
                lines    = 4,
                max_lines= 8,
                elem_id  = "cnn-box",
                show_copy_button = True,
                interactive      = False,
            )
            swin_box = gr.Textbox(
                label    = "Swin-DistilGPT2 Tahmini  (Transformer Encoder + LM Decoder)",
                lines    = 4,
                max_lines= 8,
                elem_id  = "swin-box",
                show_copy_button = True,
                interactive      = False,
            )

        # Olay baglantisi
        def on_select(choice):
            idx = int(choice.split()[1]) - 1
            orig, cnn_ov, swin_ov, ref, cnn_p, swin_p, title = show_sample(idx)
            return orig, cnn_ov, swin_ov, ref, cnn_p, swin_p, f"### {title}"

        sample_radio.change(
            fn      = on_select,
            inputs  = [sample_radio],
            outputs = [img_orig, img_cnn, img_swin, ref_box, cnn_box, swin_box, sample_title],
        )

        # Baslangicta ilk ornegi goster
        demo.load(
            fn      = lambda: on_select(choices[0]),
            inputs  = None,
            outputs = [img_orig, img_cnn, img_swin, ref_box, cnn_box, swin_box, sample_title],
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    precompute_all()

    app = build_ui()
    app.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        share       = False,
        inbrowser   = True,   # tarayici otomatik acar
    )
