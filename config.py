import torch
import os
from pathlib import Path

# =========================================================================
# 🎯 CONFIG - CNN-LSTM & Transformer Medical Report Generation
# =========================================================================

class Config:
    
    # =========================================================================
    # 📂 DIRECTORY SETUP
    # =========================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_DIR = os.path.join(BASE_DIR, 'Data', 'processed')
    IMAGE_DIR = r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\OriginalData\official_data_iccv_final'
    VOCAB_PATH = os.path.join(BASE_DIR, 'Data', 'vocab', 'vocabulary.pkl')
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints_densenet_findings')
    TRANSFORMER_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints_transformer')
    
    TRAIN_PROCESSED_CSV = os.path.join(DATA_DIR, "train_processed.csv")
    VAL_PROCESSED_CSV = os.path.join(DATA_DIR, "val_processed.csv")
    TEST_PROCESSED_CSV = os.path.join(DATA_DIR, "test_processed.csv")
    
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # =========================================================================
    # 🖥️ DEVICE SETUP
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        # 🚀 Hız optimizasyonu: cuDNN auto-tuner aktif
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        GPU_NAME = "CPU"
        GPU_MEMORY = None
    
    # =========================================================================
    # 🎛️ GENERAL TRAINING PARAMETERS
    # =========================================================================
    MODEL_TYPE = 'cnn_lstm'
    
    EPOCHS = 50
    
    # 🔥 DÜZELTİLDİ: 3 → 7
    # Büyük veri setlerinde (178K örnek) model 3 epoch'ta stagnate edebilir
    # ama daha fazla zamanla kurtarılabilir. 7 daha güvenli.
    PATIENCE = 7
    
    # =========================================================================
    # 🖼️ IMAGE & DATA PARAMETERS
    # =========================================================================
    CNN_IMAGE_SIZE = 224
    TRANSFORMER_IMAGE_SIZE = 384
    NUM_WORKERS = 4  # Kaggle/Linux: 4, Windows local: 0 (multiprocessing kısıtı)
    
    # =========================================================================
    # 🧠 CNN-LSTM MODEL PARAMETERS
    # =========================================================================
    CNN_BATCH_SIZE = 32
    
    CNN_LEARNING_RATE = 1e-4
    CNN_FINE_TUNE_LR = 1e-5
    
    CNN_ENCODER_DIM = 1024     # DenseNet-121 çıkışı
    CNN_EMBED_SIZE = 1024       # 🔥 DÜZELTİLDİ: 1024→512 (bellek tasarrufu)
    CNN_HIDDEN_SIZE = 512
    CNN_ATTENTION_DIM = 256
    
    CNN_WEIGHT_DECAY = 1e-5
    CNN_GRAD_CLIP = 5.0
    CNN_DROPOUT = 0.5
    
    CNN_FINE_TUNE_START_EPOCH = 3  # 🔥 DÜZELTİLDİ: 2→5 (önce backbone olmadan öğrensin)
    
    CNN_ATTENTION_LOSS_WEIGHT = 0.5  # 🔥 DÜZELTİLDİ: 1.0→0.5 (çok agresif olmasın)
    
    # =========================================================================
    # 🤖 TRANSFORMER MODEL PARAMETERS
    # =========================================================================
    TRANSFORMER_ENCODER = "microsoft/swin-base-patch4-window12-384"
    TRANSFORMER_DECODER = "microsoft/BioGPT"
    
    TRANSFORMER_BATCH_SIZE = 4
    TRANSFORMER_LEARNING_RATE = 5e-5
    TRANSFORMER_FINE_TUNE_LR = 1e-5
    TRANSFORMER_WEIGHT_DECAY = 0.01
    TRANSFORMER_GRAD_CLIP = 1.0
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_FINE_TUNE_START_EPOCH = 3
    TRANSFORMER_MAX_LENGTH = 150
    TRANSFORMER_NUM_BEAMS = 5
    
    # Bellek tasarrufu ayarları
    TRANSFORMER_GRAD_ACCUM_STEPS = 4  # Efektif batch = BATCH_SIZE × GRAD_ACCUM_STEPS
    TRANSFORMER_USE_AMP = True         # Mixed precision (FP16)
    
    # =========================================================================
    # 📋 CHECKPOINT & LOGGING
    # =========================================================================
    CHECKPOINT_INTERVAL = 5
    SAVE_BEST_ONLY = True
    
    # CNN-LSTM Resume
    RESUME = True
    CHECKPOINT_FILE = "C:\\Users\\ygz70\\Desktop\\Bitirme Projesi Yeni\\checkpoints_densenet_findings\\best_model.pth"
    
    # Transformer Resume
    TRANSFORMER_RESUME = False
    TRANSFORMER_CHECKPOINT_FILE = os.path.join(TRANSFORMER_CHECKPOINT_DIR, "best_model")
    
    SHUTDOWN_AFTER_TRAIN = False
    
    # =========================================================================
    # ⚙️ ADVANCED PARAMETERS
    # =========================================================================
    USE_WARMUP = False
    WARMUP_EPOCHS = 2
    
    # 🔥 DÜZELTİLDİ: GPU varsa True yap → ~2x hız artışı, %40 az bellek
    # False bırakırsan CPU'da veya eski GPU'larda güvenli çalışır
    USE_AMP = True  # GPU varsa True önerilir
    
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # =========================================================================
    # 🛠️ SETUP
    # =========================================================================
    @staticmethod
    def setup():
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.TRANSFORMER_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🎯 CONFIG INITIALIZED")
        print(f"{'='*70}")
        print(f"\n📱 DEVICE:")
        print(f"   Device:          {Config.DEVICE}")
        print(f"   GPU Name:        {Config.GPU_NAME}")
        if Config.GPU_MEMORY:
            print(f"   GPU Memory:      {Config.GPU_MEMORY:.2f} GB")
        
        print(f"\n📂 DIRECTORIES:")
        print(f"   Base:            {Config.BASE_DIR}")
        print(f"   Data:            {Config.DATA_DIR}")
        print(f"   Checkpoints:     {Config.CHECKPOINT_DIR}")
        print(f"   Logs:            {Config.LOG_DIR}")
        
        print(f"\n🧠 CNN-LSTM PARAMETERS:")
        print(f"   Image Size:      {Config.CNN_IMAGE_SIZE}×{Config.CNN_IMAGE_SIZE}")
        print(f"   Batch Size:      {Config.CNN_BATCH_SIZE}")
        print(f"   Learning Rate:   {Config.CNN_LEARNING_RATE}")
        print(f"   Fine-tune LR:    {Config.CNN_FINE_TUNE_LR}")
        print(f"   Embed Size:      {Config.CNN_EMBED_SIZE}")
        print(f"   Hidden Size:     {Config.CNN_HIDDEN_SIZE}")
        print(f"   Attention Dim:   {Config.CNN_ATTENTION_DIM}")
        print(f"   Fine-tune Epoch: {Config.CNN_FINE_TUNE_START_EPOCH}")
        print(f"   Attn LW:         {Config.CNN_ATTENTION_LOSS_WEIGHT}")

        print(f"\n📊 TRAINING:")
        print(f"   Total Epochs:    {Config.EPOCHS}")
        print(f"   Early Stopping:  patience={Config.PATIENCE}")
        print(f"   AMP (FP16):      {Config.USE_AMP}")
        print(f"   Grad Accum:      {Config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Resume:          {Config.RESUME}")
        print(f"\n{'='*70}\n")


# =========================================================================
# 📝 HELPER FUNCTIONS
# =========================================================================
def get_model_config(model_type: str = None) -> dict:
    model_type = model_type or Config.MODEL_TYPE
    if model_type == 'cnn_lstm':
        return {
            'batch_size': Config.CNN_BATCH_SIZE,
            'image_size': Config.CNN_IMAGE_SIZE,
            'learning_rate': Config.CNN_LEARNING_RATE,
            'fine_tune_lr': Config.CNN_FINE_TUNE_LR,
            'encoder_dim': Config.CNN_ENCODER_DIM,
            'embed_size': Config.CNN_EMBED_SIZE,
            'hidden_size': Config.CNN_HIDDEN_SIZE,
            'attention_dim': Config.CNN_ATTENTION_DIM,
            'weight_decay': Config.CNN_WEIGHT_DECAY,
            'grad_clip': Config.CNN_GRAD_CLIP,
            'dropout': Config.CNN_DROPOUT,
            'fine_tune_start_epoch': Config.CNN_FINE_TUNE_START_EPOCH,
            'attention_loss_weight': Config.CNN_ATTENTION_LOSS_WEIGHT,
        }
    elif model_type == 'transformer':
        return {
            'batch_size': Config.TRANSFORMER_BATCH_SIZE,
            'image_size': Config.TRANSFORMER_IMAGE_SIZE,
            'learning_rate': Config.TRANSFORMER_LEARNING_RATE,
            'fine_tune_lr': Config.TRANSFORMER_FINE_TUNE_LR,
            'weight_decay': Config.TRANSFORMER_WEIGHT_DECAY,
            'grad_clip': Config.TRANSFORMER_GRAD_CLIP,
            'dropout': Config.TRANSFORMER_DROPOUT,
            'fine_tune_start_epoch': Config.TRANSFORMER_FINE_TUNE_START_EPOCH,
            'max_length': Config.TRANSFORMER_MAX_LENGTH,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_device_info():
    print(f"\n{'='*70}")
    print(f"🖥️ DEVICE INFORMATION")
    print(f"{'='*70}")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"CUDA Available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version:       {torch.version.cuda}")
        print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"⚠️  CPU training (slow!)")
    print(f"{'='*70}\n")


class ConfigPresets:
    @staticmethod
    def quick_test():
        Config.EPOCHS = 1
        Config.CNN_BATCH_SIZE = 4
        Config.PATIENCE = 1
        print("⚡ Quick test configuration loaded")
    
    @staticmethod
    def gpu_limited():
        Config.CNN_BATCH_SIZE = 8
        Config.TRANSFORMER_BATCH_SIZE = 2
        Config.CNN_IMAGE_SIZE = 256
        Config.USE_AMP = True
        print("💾 Limited GPU configuration loaded")
    
    @staticmethod
    def gpu_high_memory():
        Config.CNN_BATCH_SIZE = 64
        Config.TRANSFORMER_BATCH_SIZE = 8
        print("🚀 High-memory GPU configuration loaded")


if __name__ == "__main__":
    Config.setup()
    print_device_info()