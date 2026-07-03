import torch
import os

# =========================================================================
# 🎯 CONFIG — ViT-DistilGPT-2 Medical Report Generation (MTL)
# =========================================================================

class Config:

    # =========================================================================
    # 📂 DİZİN AYARLARI
    # =========================================================================
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR  = os.path.join(BASE_DIR, 'Data', 'processed')
    IMAGE_DIR = os.path.join(BASE_DIR, 'OriginalData', 'official_data_iccv_final')

    # Bölümleme Türü (Split Type): "subject_level" (hasta tabanlı) veya "image_level" (sızıntılı eski split)
    SPLIT_TYPE = "subject_level"

    # CSV dosyaları
    TRAIN_PROCESSED_CSV = os.path.join(DATA_DIR, "labeled_reports_train.csv")
    VAL_PROCESSED_CSV   = os.path.join(DATA_DIR, "labeled_reports_val.csv")
    TEST_PROCESSED_CSV  = os.path.join(DATA_DIR, "labeled_reports_test.csv")

    # Checkpoint dizinleri (Bölümleme türüne göre otomatik ayrılır)
    CHECKPOINT_DIR     = os.path.join(BASE_DIR, f'checkpoints_densenet_findings_{SPLIT_TYPE}')   # CNN-LSTM
    SWIN_CHECKPOINT_DIR = os.path.join(BASE_DIR, f'checkpoints_swin_distilgpt2_{SPLIT_TYPE}')     # Swin-B

    LOG_DIR    = os.path.join(BASE_DIR, f'logs_{SPLIT_TYPE}')
    VOCAB_PATH = os.path.join(BASE_DIR, 'Data', 'vocab', 'vocabulary.pkl')         # CNN-LSTM (eski)

    # =========================================================================
    # 🖥️ DONANIM
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        GPU_NAME   = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False
    else:
        GPU_NAME   = "CPU"
        GPU_MEMORY = None

    # =========================================================================
    # 🎛️ GENEL EĞİTİM PARAMETRELERİ
    # =========================================================================
    EPOCHS   = 100
    PATIENCE = 3                # EarlyStopping sabır eşiği
    NUM_WORKERS = 0               # Windows: 0 (multiprocessing kısıtı)

    CHECKPOINT_INTERVAL = 5       # Her N epoch'ta bir periyodik checkpoint
    SHUTDOWN_AFTER_TRAIN = False  # Eğitim bitince bilgisayarı kapat

    # =========================================================================
    # 🧠 ViT-DistilGPT-2 MODEL PARAMETRELERİ
    # =========================================================================
    VIT_ENCODER = "google/vit-base-patch16-224"
    VIT_DECODER = "distilgpt2"

    VIT_IMAGE_SIZE        = 224    # ViT-base-patch16-224 giriş boyutu
    VIT_BATCH_SIZE        = 16     # 12 GB VRAM için güvenli
    VIT_MAX_LENGTH        = 150    # Maks rapor token sayısı
    VIT_NUM_BEAMS         = 5      # Beam search genişliği
    VIT_NUM_PREFIX_TOKENS = 197    # Vizüel prefix: 1 CLS + 196 patch

    # Diferansiyel öğrenme hızları
    VIT_ENCODER_LR = 4e-6          # ViT encoder (pretrained → çok düşük)
    VIT_DECODER_LR = 2e-4          # DistilGPT-2 + Projeksiyon + Classifier

    VIT_WEIGHT_DECAY  = 0.01
    VIT_GRAD_CLIP     = 1.0
    VIT_GRAD_ACCUM    = 8          # Efektif batch = 16 × 8 = 128
    VIT_WARMUP_EPOCHS = 2          # İlk N epoch: sadece projeksiyon + sınıflandırıcı
    VIT_PROJ_DROPOUT  = 0.1        # Projeksiyon katmanı dropout

    # =========================================================================
    # 🦅 Swin-B + DistilGPT-2 MODEL PARAMETRELERİ  ← Yeni Mimari
    # =========================================================================
    SWIN_ENCODER = "microsoft/swin-base-patch4-window7-224"
    SWIN_DECODER = "distilgpt2"

    SWIN_IMAGE_SIZE        = 224   # Swin-B giriş boyutu (değişmez)
    SWIN_HIDDEN_DIM        = 1024  # Swin-B stage-4 çıktı boyutu
    SWIN_NUM_PREFIX_TOKENS = 49    # Vizüel prefix: 7×7 spatial grid (stage-4)

    # Diferansiyel öğrenme hızları — ViT'ten daha düşük (Swin daha büyük)
    SWIN_ENCODER_LR = 2e-6         # Swin-B encoder (pretrained → çok düşük)
    SWIN_DECODER_LR = 2e-4         # DistilGPT-2 + Projeksiyon + Classifier

    SWIN_WEIGHT_DECAY  = 0.01
    SWIN_GRAD_CLIP     = 1.0
    SWIN_GRAD_ACCUM    = 16        # Efektif batch = batch_size × 16
    SWIN_WARMUP_EPOCHS = 2         # İlk N epoch: sadece projeksiyon + sınıflandırıcı
    SWIN_PROJ_DROPOUT  = 0.1       # Projeksiyon katmanı dropout
    SWIN_NUM_BEAMS     = 5        # Beam search genişliği

    # =========================================================================
    # 🖼️ CNN-LSTM MODEL PARAMETRELERİ  ← Eski mimari (train.py hâlâ kullanıyor)
    # =========================================================================
    MODEL_TYPE   = 'cnn_lstm'
    CNN_IMAGE_SIZE = 224
    CNN_BATCH_SIZE = 32

    CNN_LEARNING_RATE      = 1e-4
    CNN_FINE_TUNE_LR       = 1e-5
    CNN_WEIGHT_DECAY       = 1e-5
    CNN_GRAD_CLIP          = 5.0
    CNN_DROPOUT            = 0.5
    CNN_FINE_TUNE_START_EPOCH = 3

    CNN_ENCODER_DIM        = 1024
    CNN_EMBED_SIZE         = 1024
    CNN_HIDDEN_SIZE        = 512
    CNN_ATTENTION_DIM      = 256
    CNN_ATTENTION_LOSS_WEIGHT = 0.5

    # CNN-LSTM Resume
    RESUME          = True
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "best_model.pth")


    # =========================================================================
    # 🛠️ SETUP
    # =========================================================================
    @staticmethod
    def vit_setup():
        """Sadece ViT eğitimi için gerekli klasörleri oluşturur."""
        os.makedirs(Config.SWIN_CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR,            exist_ok=True)

    @staticmethod
    def setup():
        """Tüm proje klasörlerini oluşturur (eski scriptler için)."""
        os.makedirs(Config.CHECKPOINT_DIR,             exist_ok=True)
        os.makedirs(Config.SWIN_CHECKPOINT_DIR,         exist_ok=True)
        os.makedirs(Config.LOG_DIR,                    exist_ok=True)

        print(f"\n{'='*70}")
        print(f"🎯 CONFIG INITIALIZED")
        print(f"{'='*70}")
        print(f"   Device     : {Config.DEVICE}  ({Config.GPU_NAME})")
        if Config.GPU_MEMORY:
            print(f"   GPU Memory : {Config.GPU_MEMORY:.1f} GB")
        print(f"   Base Dir   : {Config.BASE_DIR}")
        print(f"   Epochs     : {Config.EPOCHS}  |  Patience: {Config.PATIENCE}")
        print(f"{'='*70}\n")


# =========================================================================
# 📝 YARDIMCI FONKSİYONLAR
# =========================================================================
def print_device_info():
    print(f"\n{'='*70}")
    print(f"🖥️  DEVICE INFORMATION")
    print(f"{'='*70}")
    print(f"   PyTorch   : {torch.__version__}")
    print(f"   CUDA      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA ver  : {torch.version.cuda}")
        print(f"   GPU       : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  CPU eğitimi (yavaş!)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    Config.setup()
    print_device_info()