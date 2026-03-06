"""
plot_loss.py  —  Tüm log dosyalarını birleştirerek Training vs Validation Loss grafiği çizer.
Epoch 1 değerleri kullanıcı tarafından sağlandı: train=2.5, val=2.05
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Epoch 1: kullanıcı tarafından sağlandı ──────────────────────────────────
epochs      = [1]
train_losses = [2.5]
val_losses   = [2.05]

# ── Log dosyası 1: epoch 2-11 ───────────────────────────────────────────────
epochs      += [2,3,4,5,6,7,8,9,10,11]
train_losses += [2.312986528772836, 2.156647737566927, 2.0823013693024683,
                 2.0169378822593464, 1.9717398733141436, 1.9368714632519615,
                 1.9088428459725746, 1.8853399014853987, 1.8650410917893114,
                 1.8470223880280796]
val_losses  += [1.963851251634666, 1.8610165878698814, 1.8071206812151854,
                1.7599804909103367, 1.7253808831031319, 1.6945479966103523,
                1.672034304227293, 1.6525361602919675, 1.634868939243752,
                1.6210421425317378]

# ── Log dosyası 2: epoch 12-21 ──────────────────────────────────────────────
epochs      += [12,13,14,15,16,17,18,19,20,21]
train_losses += [1.841599469147201, 1.824302415463329, 1.8103088909386518,
                 1.797857904973632, 1.7863916523860288, 1.7770879848317198,
                 1.7677188828723995, 1.759032558722692, 1.7510985648060513,
                 1.7433187385332352]
val_losses  += [1.6169257271635065, 1.6025384652350587, 1.5895413068937077,
                1.5792356089798312, 1.570133960429814, 1.5635986175813301,
                1.5549412451976405, 1.5482512058553743, 1.5425068934081563,
                1.5353737878636973]

# ── Log dosyası 3: epoch 22-29 ──────────────────────────────────────────────
epochs      += [22,23,24,25,26,27,28,29]
train_losses += [1.743563056398838, 1.7347718375357204, 1.7276247634903108,
                 1.7211326902485082, 1.7154098201170498, 1.7096996233522797,
                 1.7038411902105557, 1.698709973999533]
val_losses  += [1.5373799904444758, 1.5291153692102513, 1.523034378454673,
                1.5194700357251922, 1.5150153171873986, 1.5107382068845239,
                1.5060803754163155, 1.5029881372159344]

# ── Numpy'e çevir ────────────────────────────────────────────────────────────
epochs       = np.array(epochs)
train_losses = np.array(train_losses)
val_losses   = np.array(val_losses)

# ── En iyi değerler ──────────────────────────────────────────────────────────
best_val_epoch = epochs[np.argmin(val_losses)]
best_val_loss  = val_losses.min()
best_tr_epoch  = epochs[np.argmin(train_losses)]
best_tr_loss   = train_losses.min()

# ════════════════════════════════════════════════════════════════════════════
# GRAFİK
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 10),
    facecolor="#0d1117",
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
)

# ── Renk ve stil ────────────────────────────────────────────────────────────
TRAIN_COLOR  = "#60a5fa"   # blue-400
VAL_COLOR    = "#34d399"   # emerald-400
GRID_COLOR   = "#1f2937"
BG_COLOR     = "#111827"
TICK_COLOR   = "#9ca3af"
LABEL_COLOR  = "#e5e7eb"
ACCENT_COLOR = "#f59e0b"   # amber

ax1, ax2 = axes

for ax in axes:
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TICK_COLOR, labelsize=10)
    ax.spines[:].set_color("#374151")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

# ── Ana grafik: Loss eğrileri ────────────────────────────────────────────────
ax1.plot(epochs, train_losses,
         color=TRAIN_COLOR, linewidth=2.2, marker='o', markersize=4,
         markerfacecolor=TRAIN_COLOR, markeredgewidth=0,
         label="Train Loss", zorder=3)

ax1.plot(epochs, val_losses,
         color=VAL_COLOR, linewidth=2.2, marker='s', markersize=4,
         markerfacecolor=VAL_COLOR, markeredgewidth=0,
         label="Val Loss", zorder=3)

# Gölge (fill_between)
ax1.fill_between(epochs, train_losses, val_losses,
                 alpha=0.08, color="#818cf8")
ax1.fill_between(epochs, val_losses, val_losses.min() - 0.02,
                 alpha=0.10, color=VAL_COLOR)

# En iyi val loss noktası
ax1.axvline(best_val_epoch, color=ACCENT_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.7, zorder=2)
ax1.scatter([best_val_epoch], [best_val_loss],
            color=ACCENT_COLOR, s=120, zorder=5,
            label=f"Best Val Loss  (Epoch {best_val_epoch}: {best_val_loss:.4f})")
ax1.annotate(
    f"  Best Val\n  Epoch {best_val_epoch}\n  {best_val_loss:.4f}",
    xy=(best_val_epoch, best_val_loss),
    xytext=(best_val_epoch + 0.8, best_val_loss + 0.03),
    fontsize=9, color=ACCENT_COLOR,
    arrowprops=dict(arrowstyle="->", color=ACCENT_COLOR, lw=1.2)
)

# LR değişim anotasyonu (epoch 4 civarında fine-tune başlıyor)
ax1.axvline(4, color="#a78bfa", linewidth=1.0, linestyle=":", alpha=0.6, zorder=2)
ax1.text(4.2, train_losses.max() * 0.99, "Fine-tune\nStart",
         color="#a78bfa", fontsize=8, va="top")

# Grid
ax1.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8, zorder=1)
ax1.set_xlim(0.5, epochs.max() + 0.5)
ax1.set_ylim(val_losses.min() - 0.05, train_losses.max() + 0.12)
ax1.set_ylabel("Loss", color=LABEL_COLOR, fontsize=12)
ax1.tick_params(labelbottom=False)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

# Legend
leg = ax1.legend(
    loc="upper right", fontsize=10,
    facecolor="#1f2937", edgecolor="#374151",
    labelcolor=LABEL_COLOR, framealpha=0.9
)

# Başlık
fig.text(
    0.5, 0.96,
    "CNN-LSTM Medical Report Generation  —  Training & Validation Loss",
    ha="center", va="top",
    fontsize=14, fontweight="bold", color=LABEL_COLOR
)
fig.text(
    0.5, 0.925,
    f"DenseNet-121 Encoder  ✦  2-Layer LSTM Decoder  ✦  Additive Attention  ✦  Epochs 1–{epochs.max()}",
    ha="center", va="top",
    fontsize=9.5, color=TICK_COLOR
)

# ── Alt grafik: Fark (Train - Val Gap) ──────────────────────────────────────
gap = train_losses - val_losses
ax2.bar(epochs, gap, color="#818cf8", alpha=0.7, width=0.7, zorder=3)
ax2.axhline(0, color="#374151", linewidth=0.8)
ax2.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7, axis="y", zorder=1)
ax2.set_xlabel("Epoch", color=LABEL_COLOR, fontsize=12)
ax2.set_ylabel("Train−Val\nGap", color=LABEL_COLOR, fontsize=9)
ax2.set_xlim(0.5, epochs.max() + 0.5)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ── İstatistik kutusu ────────────────────────────────────────────────────────
stats_text = (
    f"Train  │  Start: {train_losses[0]:.4f}   End: {train_losses[-1]:.4f}   Best: {best_tr_loss:.4f} (ep.{best_tr_epoch})\n"
    f"Val    │  Start: {val_losses[0]:.4f}   End: {val_losses[-1]:.4f}   Best: {best_val_loss:.4f} (ep.{best_val_epoch})\n"
    f"Δ Drop │  Train: {train_losses[0]-train_losses[-1]:.4f}   Val: {val_losses[0]-val_losses[-1]:.4f}"
)
fig.text(
    0.02, 0.02, stats_text,
    fontsize=8.5, color=TICK_COLOR,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#1f2937", edgecolor="#374151", alpha=0.9)
)

# ── Kaydet ───────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "training_loss_full.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"✅ Grafik kaydedildi: {out_path}")
