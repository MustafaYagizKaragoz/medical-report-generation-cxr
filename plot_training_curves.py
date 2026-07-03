# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np

# Set high-quality style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.5
})

# --- DATA ---
# CNN-LSTM Losses (22 Epochs)
cnn_epochs = np.arange(1, 23)
cnn_train = [3.0200, 2.3730, 2.2178, 2.1407, 2.0769, 2.0311, 2.0079, 1.9758, 1.9597, 1.9353, 
             1.9157, 1.8988, 1.8920, 1.8764, 1.8629, 1.8516, 1.8407, 1.8382, 1.8276, 1.8180, 
             1.8095, 1.8020]
cnn_val = [2.1880, 1.9935, 1.8950, 1.8469, 1.8023, 1.7721, 1.7546, 1.7331, 1.7202, 1.7034, 
           1.6896, 1.6787, 1.6756, 1.6642, 1.6565, 1.6492, 1.6429, 1.6405, 1.6364, 1.6316, 
           1.6267, 1.6252]

# Swin-GPT MTL Losses (11 Epochs)
swin_epochs = np.arange(1, 12)
swin_train = [4.4714, 3.7727, 1.7234, 1.4250, 1.3363, 1.2781, 1.2279, 1.1897, 1.1547, 1.1222, 1.0913]
swin_val = [3.5385, 3.3559, 1.3725, 1.2916, 1.2493, 1.2266, 1.2133, 1.2048, 1.2035, 1.2004, 1.2012]

# --- PLOTTING ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Left Panel: CNN-LSTM
axes[0].plot(cnn_epochs, cnn_train, marker='o', markersize=4, linestyle='-', linewidth=2, color='#1f77b4', label='Training Loss')
axes[0].plot(cnn_epochs, cnn_val, marker='s', markersize=4, linestyle='--', linewidth=2, color='#aec7e8', label='Validation Loss')
axes[0].set_title('CNN-LSTM (DenseNet121 + LSTM)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_xticks(np.arange(1, 23, 2))
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
axes[0].grid(True)

# Right Panel: Swin-GPT (MTL)
axes[1].plot(swin_epochs, swin_train, marker='o', markersize=4, linestyle='-', linewidth=2, color='#d62728', label='Training Loss')
axes[1].plot(swin_epochs, swin_val, marker='s', markersize=4, linestyle='--', linewidth=2, color='#ff9896', label='Validation Loss')
axes[1].set_title('Swin-GPT MTL (Swin-B + DistilGPT2)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_xticks(swin_epochs)
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
axes[1].grid(True)

# Best weights indicator for Swin (Epoch 10)
axes[1].annotate('Best Model (Epoch 10)', 
                 xy=(10, swin_val[9]), 
                 xytext=(7, 2.0),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6, headlength=6),
                 fontsize=10, 
                 fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffebeb", ec="red", lw=1))

plt.suptitle('Training and Validation Loss Curves Comparison', y=0.98, weight='bold')
plt.tight_layout()

# Save high-resolution PNG
output_path = 'training_loss_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Loss curves saved successfully to: {output_path}")
