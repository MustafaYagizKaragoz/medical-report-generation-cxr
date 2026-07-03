"""
visual_grounding_check.py
==========================
Modelin görüntüyü gerçekten kullanıp kullanmadığını iki yöntemle test eder:

Yöntem 1 — Ablasyon Testi:
   Aynı model; 3 farklı prefix ile metin üretir:
     (A) Gerçek görsel prefix  (ViT çıktısı)
     (B) Sıfır prefix          (tüm tokenlar sıfır)
     (C) Gürültü prefix        (rastgele Gaussian gürültü)
   Eğer (A) ≠ (B) ve (A) ≠ (C) → model görüntüyü kullanıyor ✓

Yöntem 2 — GPT-2 Attention'ı Görsel Token'lara:
   GPT-2'nin son katmanındaki attention ağırlıklarını çıkarır.
   Her metin token'ının görsel prefix token'larına ortalama attention'ını gösterir.
   Yüksek değer → model metin üretirken görsel bilgiye bakıyor ✓
"""

# ── HF Offline Koruması ────────────────────────────────────────────────────
import os as _os
def _hf_cache_ok(ids):
    root = _os.environ.get("HF_HOME", _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    return all(_os.path.isdir(_os.path.join(root, "models--" + m.replace("/", "--"))) for m in ids)
if _hf_cache_ok(["google/vit-base-patch16-224", "distilgpt2"]):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# ──────────────────────────────────────────────────────────────────────────

import os
import sys

# Support emoji/Unicode prints in Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
try:
    from src.models.vit_distil2gpt import ViTDistilGPT2ForMTL
except ImportError:
    ViTDistilGPT2ForMTL = None

try:
    from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL
except ImportError:
    SwinDistilGPT2ForMTL = None

# =========================================================================
# AKTIF MODEL VE YOL TESPITI
# =========================================================================
SWIN_CKPT = os.path.join(Config.SWIN_CHECKPOINT_DIR, "best_model_swin_distilgpt2.pth")
if not os.path.exists(SWIN_CKPT):
    _swa_path = os.path.join(Config.SWIN_CHECKPOINT_DIR, "swa_model_final.pth")
    if os.path.exists(_swa_path):
        SWIN_CKPT = _swa_path

VIT_CKPT = os.path.join(Config.BASE_DIR, "checkpoints_vit_distilgpt2", "best_model_vit_distilgpt2.pth")

if os.path.exists(SWIN_CKPT):
    MODEL_TYPE = "swin"
    BEST_CKPT = SWIN_CKPT
    MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
else:
    MODEL_TYPE = "vit"
    BEST_CKPT = VIT_CKPT
    MODEL_NAME = "google/vit-base-patch16-224"
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

GPT_MODEL_NAME = "distilgpt2"
IMAGE_SIZE     = 224
MAX_NEW_TOKENS = 120
NUM_BEAMS      = 4

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# =========================================================================
# YARDIMCILAR
# =========================================================================
def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return _transform(img).unsqueeze(0)   # (1, 3, H, W)


def denorm(t: torch.Tensor) -> np.ndarray:
    """Normalize edilmiş tensörü görüntüye döndür."""
    t = t.squeeze().cpu().float()
    t = t * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
    t = t.permute(1, 2, 0).clamp(0, 1).numpy()
    return t


def load_model_and_tokenizer(device: torch.device):
    if not os.path.exists(BEST_CKPT):
        raise FileNotFoundError(f"❌ Checkpoint bulunamadı: {BEST_CKPT}")

    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if MODEL_TYPE == "swin":
        if SwinDistilGPT2ForMTL is None:
            raise ImportError("SwinDistilGPT2ForMTL model dosyası bulunamadı.")
        model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
            encoder_name=MODEL_NAME,
            decoder_name=GPT_MODEL_NAME,
            enable_gradient_checkpointing=False,
        ).to(device)
    else:
        if ViTDistilGPT2ForMTL is None:
            raise ImportError("ViTDistilGPT2ForMTL model dosyası bulunamadı.")
        model = ViTDistilGPT2ForMTL.from_pretrained_mtl(
            encoder_name=MODEL_NAME,
            decoder_name=GPT_MODEL_NAME,
            enable_gradient_checkpointing=False,
        ).to(device)

    model.gpt.config.pad_token_id = tokenizer.pad_token_id
    model.gpt.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id     = tokenizer.pad_token_id
    model.config.eos_token_id     = tokenizer.eos_token_id

    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("val_loss", float("nan"))
    print(f"   ✅ Model yüklendi  [epoch={epoch+1 if isinstance(epoch,int) else epoch}, val_loss={loss:.4f}]")
    return model, tokenizer


# =========================================================================
# YÖNTEM 1: ABLASYON TESTİ
# =========================================================================
def run_ablation_test(model, tokenizer, pixel_values: torch.Tensor, device: torch.device) -> dict:
    """
    Üç farklı prefix ile metin üretir:
      real   : gerçek ViT çıktısı
      zero   : sıfır vektör prefix (görüntü bilgisi yok)
      noise  : Gaussian gürültü prefix (anlamsız bilgi)
    """
    eos_id = tokenizer.eos_token_id

    def generate_with_prefix(prefix: torch.Tensor) -> str:
        B = prefix.size(0)
        # BOS token ekle — predict koduyla aynı davranış
        bos_ids = torch.full((B, 1), fill_value=1169, dtype=torch.long, device=device)
        bos_embeds = model.gpt.transformer.wte(bos_ids)
        combined = torch.cat([prefix, bos_embeds], dim=1)
        mask = torch.ones(B, combined.size(1), dtype=torch.long, device=device)
        
        with torch.no_grad():
            ids = model.gpt.generate(
                inputs_embeds=combined,
                attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,           # Config'den değil, sabit 4
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    # ── Gerçek prefix ──────────────────────────────────────────────────
    with torch.no_grad():
        _, real_prefix = model._encode_image(pixel_values)

    D = real_prefix.shape[-1]
    P = real_prefix.shape[1]

    # ── Sıfır prefix ───────────────────────────────────────────────────
    zero_prefix  = torch.zeros_like(real_prefix)

    # ── Gürültü prefix ─────────────────────────────────────────────────
    noise_prefix = torch.randn_like(real_prefix) * real_prefix.std()

    print("\n   🔬 Ablasyon üretimi yapılıyor...")
    real_text  = generate_with_prefix(real_prefix)
    zero_text  = generate_with_prefix(zero_prefix)
    noise_text = generate_with_prefix(noise_prefix)

    return {
        "real":  real_text,
        "zero":  zero_text,
        "noise": noise_text,
    }


def token_overlap(a: str, b: str) -> float:
    """İki metin arasındaki token örtüşme oranı (Jaccard)."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# =========================================================================
# YÖNTEM 2: GPT-2 ATTENTION → GÖRSEL TOKEN'LAR
# =========================================================================
def compute_visual_attention_ratio(model, pixel_values: torch.Tensor,
                                    tokenizer, device: torch.device) -> dict:
    """
    GPT-2'nin attention ağırlıklarında metin token'larının görsel prefix
    token'larına ne kadar baktığını ölçer.

    NOT: DistilGPT-2 varsayılan olarak SDPA kullanır (PyTorch 2.0+).
    SDPA attention weight'leri döndürmez → geçici olarak 'eager' moda geç.
    """
    with torch.no_grad():
        _, visual_prefix = model._encode_image(pixel_values)

    B, P, D = visual_prefix.shape
    prefix_mask = torch.ones(B, P, dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id

    # Greedy decode (attention analizi için num_beams=1, early_stopping geçersiz)
    with torch.no_grad():
        gen_ids = model.gpt.generate(
            inputs_embeds=visual_prefix,
            attention_mask=prefix_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            forced_eos_token_id=eos_id,
        )

    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
    n_text_tokens  = gen_ids.shape[1]

    # ── Eager moda geç → GPT-2 attention weight'leri dönsün ─────────────
    # SDPA (varsayılan) attention weight döndürmez → 'eager' gerekli
    _cfg      = model.gpt.config
    _prev_impl = getattr(_cfg, "_attn_implementation", "sdpa")

    layer_ratios = []

    try:
        _cfg._attn_implementation = "eager"

        text_embeds = model.gpt.transformer.wte(gen_ids)   # (1, T_gen, D)
        combined    = torch.cat([visual_prefix, text_embeds], dim=1)  # (1, P+T, D)
        total_len   = combined.shape[1]
        full_mask   = torch.ones(1, total_len, dtype=torch.long, device=device)

        with torch.no_grad():
            gpt_out = model.gpt(
                inputs_embeds=combined,
                attention_mask=full_mask,
                output_attentions=True,
            )

        if gpt_out.attentions is not None:
            for layer_attn in gpt_out.attentions:
                if layer_attn is None:
                    continue
                # layer_attn: (B, H, T, T)
                # Metin pozisyonlarının (P sonrası) görsel token'lara (0:P) bakışı
                text_to_visual = layer_attn[0, :, P:, :P]   # (H, T_text, P)
                text_to_all    = layer_attn[0, :, P:, :]     # (H, T_text, T_all)

                visual_sum = text_to_visual.sum(dim=-1).mean().item()   # skaler
                total_sum  = text_to_all.sum(dim=-1).mean().item()
                ratio = visual_sum / (total_sum + 1e-9)
                layer_ratios.append(float(ratio))

            if layer_ratios:
                print(f"   ✅ GPT-2 attention (eager mod) başarıyla alındı — {len(layer_ratios)} katman")
            else:
                print("   ⚠️  Attention döndürüldü fakat hiç katman yoktu.")
        else:
            print("   ⚠️  output_attentions=True'ya rağmen attentions=None geldi.")

    except Exception as exc:
        print(f"   ⚠️  GPT-2 attention alınamadı: {exc}")
        layer_ratios = []

    finally:
        _cfg._attn_implementation = _prev_impl   # orijinal moda geri dön

    return {
        "layer_ratios":      layer_ratios,
        "mean_ratio":        float(np.mean(layer_ratios)) if layer_ratios else 0.0,
        "generated":         generated_text,
        "num_prefix_tokens": P,
        "num_text_tokens":   n_text_tokens,
    }


# =========================================================================
# GÖRSELLEŞTIRME
# =========================================================================
def visualize_results(image_path: str, pixel_values: torch.Tensor,
                       ablation: dict, attention: dict, save_path: str):
    fig = plt.figure(figsize=(20, 11), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.30,
                            left=0.04, right=0.97, top=0.92, bottom=0.05)

    # Başlık
    fig.suptitle(f"Visual Grounding Check — {MODEL_TYPE.upper()}-DistilGPT-2",
                 color="#e2e8f0", fontsize=14, fontweight="bold", y=0.97)

    dark = {"facecolor": "#161b22"}
    txt_kw = dict(color="#c9d1d9", fontsize=8.5, wrap=True,
                  verticalalignment="top", horizontalalignment="left")
    box_kw = dict(boxstyle="round,pad=0.4", alpha=0.85)

    # ── Görüntü ──────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(denorm(pixel_values))
    ax_img.axis("off")
    ax_img.set_title(Path(image_path).name, color="#8b949e", fontsize=8, pad=4)

    # ── Ablasyon sonuçları ────────────────────────────────────────────────
    ax_ab = fig.add_subplot(gs[0, 1:])
    ax_ab.axis("off")
    ax_ab.set_title("Ablasyon Testi — Prefix Türüne Göre Üretim", color="#58a6ff", fontsize=10, pad=4)

    overlap_zero  = token_overlap(ablation["real"], ablation["zero"])
    overlap_noise = token_overlap(ablation["real"], ablation["noise"])

    # Renk sinyali: düşük örtüşme = model görsel bilgiyi kullanıyor
    def signal_color(v):
        if v < 0.2:   return "#3fb950"   # yeşil: iyi (az örtüşme)
        if v < 0.45:  return "#f0a30a"   # sarı: orta
        return "#ff7b72"                  # kırmızı: sorunlu

    texts = [
        ("✅ GERÇEK prefix", ablation["real"],  "#3fb950", "#0d2b0d"),
        ("⬜ SIFIR prefix",  ablation["zero"],  "#ff7b72", "#2b0d0d"),
        ("🔀 GÜRÜLTÜ prefix",ablation["noise"], "#f0a30a", "#2b220d"),
    ]
    y = 0.98
    for title, text, ec, fc in texts:
        ax_ab.text(0.01, y, title, color=ec, fontsize=9, fontweight="bold",
                   transform=ax_ab.transAxes)
        y -= 0.06
        snippet = text[:280] + ("…" if len(text) > 280 else "")
        ax_ab.text(0.01, y, snippet, transform=ax_ab.transAxes,
                   bbox=dict(**box_kw, facecolor=fc, edgecolor=ec), **txt_kw)
        y -= 0.30

    # Jaccard skorları
    ax_ab.text(0.01, 0.04,
               f"Token örtüşmesi  real↔zero: {overlap_zero:.2f}   real↔noise: {overlap_noise:.2f}"
               f"   {'✅ Görsel bilgi KULLANILIYOR' if overlap_zero < 0.4 else '⚠️  Örtüşme yüksek (daha fazla eğitim gerekebilir)'}",
               transform=ax_ab.transAxes,
               fontsize=9, color="#e2e8f0",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#444c56", alpha=0.9))

    # ── Attention ratio — katman başına ──────────────────────────────────
    ax_att = fig.add_subplot(gs[1, :2])
    ratios = attention["layer_ratios"]
    if ratios:
        layers = list(range(1, len(ratios) + 1))
        colors = ["#3fb950" if r > 0.25 else "#f0a30a" if r > 0.10 else "#ff7b72"
                  for r in ratios]
        bars = ax_att.bar(layers, ratios, color=colors, width=0.6, alpha=0.85)
        ax_att.axhline(y=attention["mean_ratio"], color="#58a6ff", linewidth=1.5,
                       linestyle="--", label=f"Ort. {attention['mean_ratio']:.3f}")
        ax_att.axhline(y=1.0/attention["num_prefix_tokens"] if attention["num_prefix_tokens"] else 0,
                       color="#f0a30a", linewidth=1, linestyle=":",
                       label="Beklenen rastgele oran")
        ax_att.set_xlabel("GPT-2 Katmanı", color="#8b949e", fontsize=9)
        ax_att.set_ylabel("Görsel Prefix Attention Oranı", color="#8b949e", fontsize=9)
        ax_att.set_title("GPT-2 Attention → Görsel Prefix Token'ları  (yüksek = görüntüye bakıyor ✓)",
                         color="#58a6ff", fontsize=9, pad=4)
        ax_att.tick_params(colors="#8b949e")
        for spine in ax_att.spines.values(): spine.set_color("#30363d")
        ax_att.legend(fontsize=8, labelcolor="#e2e8f0",
                      facecolor="#21262d", edgecolor="#444c56")
    else:
        ax_att.text(0.5, 0.5, "Attention verisi alınamadı",
                    ha="center", va="center", color="#ff7b72",
                    transform=ax_att.transAxes, fontsize=10)
        ax_att.axis("off")

    # ── Özet kutu ─────────────────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.axis("off")
    ax_sum.set_title("Özet", color="#58a6ff", fontsize=10, pad=4)

    mean_r = attention["mean_ratio"]
    if mean_r > 0.25:
        verdict = "✅ Model görüntüyü ETKİN kullanıyor"
        vc = "#3fb950"
    elif mean_r > 0.10:
        verdict = "⚠️  Model görüntüye KISMI bakıyor"
        vc = "#f0a30a"
    else:
        verdict = "❌ Model görüntüyü ihmal ediyor\n(daha fazla eğitim gerekli)"
        vc = "#ff7b72"

    summary = (
        f"Prefix token sayısı : {attention['num_prefix_tokens']}\n"
        f"Üretilen token sayısı: {attention['num_text_tokens']}\n"
        f"Ort. görsel att.     : {mean_r:.3f}\n\n"
        f"Ablasyon örtüşme ↔0 : {overlap_zero:.2f}\n"
        f"Ablasyon örtüşme ↔🔀: {overlap_noise:.2f}\n\n"
        f"{verdict}"
    )
    ax_sum.text(0.05, 0.95, summary, transform=ax_sum.transAxes,
                color="#e2e8f0", fontsize=9, verticalalignment="top",
                linespacing=1.6,
                bbox=dict(boxstyle="round,pad=0.6",
                          facecolor="#21262d", edgecolor=vc, linewidth=2))

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n   💾 Görsel kaydedildi: {save_path}")


# =========================================================================
# ANA FONKSİYON
# =========================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visual Grounding Check")
    parser.add_argument("image_path", help="Test edilecek röntgen görüntüsü")
    parser.add_argument("--save_dir", default="grounding_results",
                        help="Sonuçların kaydedileceği klasör")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Görüntü bulunamadı: {args.image_path}")
        return

    device = Config.DEVICE
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("🔬 VISUAL GROUNDING CHECK")
    print(f"{'='*70}")
    print(f"   Görüntü  : {args.image_path}")
    print(f"   Cihaz    : {device}")
    print(f"   Checkpoint: {BEST_CKPT}")

    # ── Model ──────────────────────────────────────────────────────────
    print("\n📥 Model yükleniyor...")
    model, tokenizer = load_model_and_tokenizer(device)

    # ── Görüntü ────────────────────────────────────────────────────────
    pixel_values = load_image(args.image_path).to(device)

    # ── Yöntem 1: Ablasyon ─────────────────────────────────────────────
    print("\n\n📌 YÖNTEM 1: ABLASYON TESTİ")
    print("─" * 60)
    ablation = run_ablation_test(model, tokenizer, pixel_values, device)

    overlap_zero  = token_overlap(ablation["real"], ablation["zero"])
    overlap_noise = token_overlap(ablation["real"], ablation["noise"])

    print(f"\n   GERÇEK prefix üretimi:\n   → {ablation['real'][:200]}")
    print(f"\n   SIFIR  prefix üretimi:\n   → {ablation['zero'][:200]}")
    print(f"\n   GÜRÜLTÜ prefix üretimi:\n   → {ablation['noise'][:200]}")
    print(f"\n   Token örtüşmesi  real↔zero : {overlap_zero:.3f}")
    print(f"   Token örtüşmesi  real↔noise: {overlap_noise:.3f}")

    if overlap_zero < 0.3:
        print("\n   ✅ SONUÇ: Model görüntüyü ETKİLİ kullanıyor (örtüşme düşük)")
    elif overlap_zero < 0.5:
        print("\n   ⚠️  SONUÇ: Model görüntüyü KISMI kullanıyor (daha fazla eğitim önerilir)")
    else:
        print("\n   ❌ SONUÇ: Model görüntüyü ihmal ediyor (sıfır prefix ile neredeyse aynı çıktı)")

    # ── Yöntem 2: Attention Analizi ────────────────────────────────────
    print("\n\n📌 YÖNTEM 2: GPT-2 ATTENTION ANALİZİ")
    print("─" * 60)
    attention = compute_visual_attention_ratio(model, pixel_values, tokenizer, device)

    print(f"\n   Prefix token sayısı : {attention['num_prefix_tokens']}")
    print(f"   Üretilen metin      : {attention['generated'][:150]}...")
    print(f"\n   Katman bazlı görsel attention oranları:")
    for i, r in enumerate(attention["layer_ratios"], 1):
        bar = "█" * int(r * 40) + "░" * (40 - int(r * 40))
        flag = "✅" if r > 0.25 else "⚠️ " if r > 0.10 else "❌"
        print(f"   Katman {i:2d}: {r:.4f}  [{bar}]  {flag}")

    print(f"\n   Ortalama görsel attention oranı: {attention['mean_ratio']:.4f}")
    if attention["mean_ratio"] > 0.25:
        print("   ✅ Model metin üretirken görüntüye ETKİLİ bakıyor")
    elif attention["mean_ratio"] > 0.10:
        print("   ⚠️  Model görüntüye KISMI bakıyor")
    else:
        print("   ❌ Model görüntüyü ihmal ediyor")

    # ── Görselleştirme ─────────────────────────────────────────────────
    save_path = os.path.join(args.save_dir, "grounding_result.png")
    print("\n\n📊 Görselleştirme oluşturuluyor...")
    visualize_results(args.image_path, pixel_values, ablation, attention, save_path)

    print(f"\n{'='*70}")
    print("✅ Visual Grounding Check tamamlandı!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 İptal edildi.")
    except Exception as e:
        import traceback
        print(f"\n❌ HATA: {e}")
        traceback.print_exc()
