import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cnn_lstm():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw boxes
    # (x, y, width, height)
    boxes = {
        'input': (0.5, 2.25, 1.2, 1.5, 'Input Chest X-ray\n(384x384)', '#f0f0f0'),
        'encoder': (2.2, 2.25, 1.5, 1.5, 'DenseNet-121\nEncoder\n(Feature Map:\n7x7x1024)', '#d6e4f0'),
        'attention': (4.2, 2.25, 1.5, 1.5, 'Bahdanau\nAttention Module\n(Hidden States + \nVisual Context)', '#fff1ac'),
        'decoder': (6.2, 2.25, 1.5, 1.5, 'LSTM Decoder\n(2-Layer, 512-dim\nwith Dropout)', '#d1e8e2'),
        'output': (8.2, 2.25, 1.3, 1.5, 'Generated Report\n(Findings +\nImpression)', '#e8dff5'),
    }

    # Add text labels inside boxes
    for name, (x, y, w, h, text, color) in boxes.items():
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="#4a4a4a", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        ((1.8, 3.0), (2.1, 3.0)),   # Input -> Encoder
        ((3.8, 3.0), (4.1, 3.0)),   # Encoder -> Attention
        ((5.8, 3.0), (6.1, 3.0)),   # Attention -> Decoder
        ((7.8, 3.0), (8.1, 3.0)),   # Decoder -> Output
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.5, mutation_scale=20))

    # Add descriptive title
    plt.title('CNN-LSTM Architecture (DenseNet121 + Attention + LSTM Decoder)', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('cnn_lstm_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_swin_gpt():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Draw boxes
    boxes = {
        'input': (0.5, 2.75, 1.2, 1.5, 'Input Chest X-ray\n(224x224)', '#f0f0f0'),
        'encoder': (2.2, 2.75, 1.5, 1.5, 'Swin-B Encoder\n(Stage 4 Output:\n7x7x1024)', '#d6e4f0'),
        'proj': (4.2, 2.75, 1.4, 1.5, 'Linear Projection\n(Feature Projection\n1024 -> 768)', '#fff1ac'),
        'prefix': (6.1, 2.75, 1.6, 1.5, 'Visual Prefix\n(49 Image Tokens)\n+\nText Tokens', '#fce1e4'),
        'decoder': (8.2, 2.75, 1.5, 1.5, 'DistilGPT-2\nDecoder\n(Autoregressive\nLanguage Model)', '#d1e8e2'),
        
        # Dual-heads
        'head_text': (10.2, 3.8, 1.4, 1.2, 'Generated Report\n(Findings +\nImpression)', '#e8dff5'),
        'head_cls': (10.2, 1.8, 1.4, 1.2, 'Multi-Task\nClassification Head\n(14 Pathology\nProbabilities)', '#fcd5ce')
    }

    # Add text labels inside boxes
    for name, (x, y, w, h, text, color) in boxes.items():
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec="#4a4a4a", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw arrows
    arrows = [
        ((1.8, 3.5), (2.1, 3.5)),   # Input -> Encoder
        ((3.8, 3.5), (4.1, 3.5)),   # Encoder -> Projection
        ((5.7, 3.5), (6.0, 3.5)),   # Projection -> Prefix Block
        ((7.8, 3.5), (8.1, 3.5)),   # Prefix Block -> Decoder
        ((9.8, 3.7), (10.1, 4.4)),  # Decoder -> Text Output
        ((9.8, 3.3), (10.1, 2.4)),  # Decoder -> MTL Head
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.5, mutation_scale=20))

    # Add title
    plt.title('Swin-GPT MTL Architecture (Swin-B Encoder + DistilGPT2 Decoder + Dual Heads)', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('swin_distilgpt2_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    draw_cnn_lstm()
    draw_swin_gpt()
    print("Architectures drawn successfully: cnn_lstm_architecture.png & swin_distilgpt2_architecture.png")
