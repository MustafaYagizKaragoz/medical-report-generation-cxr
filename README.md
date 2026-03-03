# 🏥 Medical Report Generation from Chest X-Rays

Göğüs röntgeni (Chest X-Ray) görüntülerinden otomatik tıbbi rapor üretimi için derin öğrenme sistemi.

> **Bitirme Projesi** — MIMIC-CXR veri seti üzerinde eğitilmiş iki farklı mimari ile karşılaştırmalı çalışma.

---

## 📋 Proje Özeti

Bu proje, radyoloji uzmanlarının iş yükünü azaltmak amacıyla göğüs röntgenlerinden **otomatik "findings" raporu** üreten iki farklı derin öğrenme mimarisi geliştirmiştir:

| Model | Encoder | Decoder | Parametre | Durum |
|-------|---------|---------|-----------|-------|
| **CNN-LSTM** | DenseNet-121 | 2-Layer LSTM + Attention | ~25M | ✅ Eğitildi |
| **Transformer** | Swin Transformer Base | BioGPT | ~435M | 🔄 Hazır |

## 🧠 Mimariler

### 1. CNN-LSTM + Attention
- **Encoder**: ImageNet önceden eğitilmiş DenseNet-121
- **Decoder**: 2 katmanlı LSTM + Additive Attention + Context Gating
- **Özellikler**: Sinusoidal positional encoding, teacher forcing, beam search

### 2. Swin Transformer + BioGPT
- **Encoder**: `microsoft/swin-base-patch4-window12-384` (384×384 giriş)
- **Decoder**: `microsoft/BioGPT` (PubMed üzerinde eğitilmiş tıbbi dil modeli)
- **Özellikler**: HuggingFace VisionEncoderDecoderModel, mixed precision (AMP), gradient accumulation

## 📊 Veri Seti

| Split | Örnek Sayısı |
|-------|-------------|
| Train | 178,221 |
| Val | ~18,000 |
| Test | ~18,000 |

**Kaynak**: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) — Beth Israel Deaconess Medical Center

## 📁 Proje Yapısı

```
├── config.py                    # Tüm konfigürasyonlar (CNN-LSTM & Transformer)
├── train.py                     # CNN-LSTM eğitim scripti
├── test.py                      # CNN-LSTM test scripti
├── train_transformer.py         # Transformer eğitim scripti
├── test_transformer.py          # Transformer test scripti
├── requirements.txt             # Bağımlılıklar
│
├── src/
│   ├── models/
│   │   ├── cnn_lstm.py          # CNN-LSTM model mimarisi
│   │   └── transformer_model.py # Swin + BioGPT model mimarisi
│   │
│   ├── data_loader/
│   │   ├── dataset.py           # CNN-LSTM için Dataset & DataLoader
│   │   ├── data_transformer.py  # Transformer için Dataset & DataLoader
│   │   └── vocabulary.py        # Kelime haznesi yönetimi
│   │
│   └── utils/
│       ├── early_stopping.py    # Early stopping mekanizması
│       └── visualization.py     # Loss curve çizimi
│
├── Data/
│   ├── processed/               # İşlenmiş CSV dosyaları
│   └── vocab/                   # Vocabulary dosyaları
│
├── ProcessData/                 # Veri ön işleme scriptleri
├── checkpoints_densenet_findings/  # CNN-LSTM checkpoint'leri
└── checkpoints_transformer/     # Transformer checkpoint'leri
```

## 🚀 Kurulum ve Kullanım

### Gereksinimler
```bash
pip install -r requirements.txt
```

### CNN-LSTM Eğitimi
```bash
python train.py
python test.py
```

### Transformer Eğitimi
```bash
python train_transformer.py
# Test otomatik çalışır veya manuel:
python test_transformer.py
```

## 📐 Değerlendirme Metrikleri

Modeller aşağıdaki metriklerle değerlendirilmektedir:
- **BLEU** (1-4): N-gram eşleşme
- **ROUGE** (1, 2, L): Özetleme kalitesi
- **METEOR**: Eşanlamlı ve morfolojik eşleşme
- **CIDEr**: Konsensüs tabanlı görüntü açıklama değerlendirmesi

## ⚙️ Konfigürasyon

Tüm hiperparametreler `config.py` dosyasından yönetilir:
- Batch size, learning rate, epoch sayısı
- Encoder/Decoder boyutları
- Fine-tuning stratejisi
- Checkpoint & early stopping ayarları

## 📜 Lisans

Bu proje akademik amaçlı geliştirilmiştir.
