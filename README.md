# 🏥 Medical Report Generation from Chest X-Rays

Göğüs röntgeni (Chest X-Ray) görüntülerinden otomatik radyoloji raporu (combined findings-and-impression) üretimi için geliştirilmiş derin öğrenme sistemi.

> **Bitirme Projesi** — MIMIC-CXR veri seti üzerinde eğitilmiş iki farklı mimari ile karşılaştırmalı çalışma.

---

## 📋 Proje Özeti

Bu proje, radyoloji uzmanlarının iş yükünü azaltmak ve raporlama sürecine karar destek mekanizması sunmak amacıyla göğüs röntgenlerinden **otomatik radyoloji raporu** (combined findings-and-impression report generation) üreten iki farklı derin öğrenme mimarisini karşılaştırmalı olarak uygulamaktadır:

| Model | Encoder | Decoder | Parametre | Durum |
|-------|---------|---------|-----------|-------|
| **CNN-LSTM** | DenseNet-121 | 2-Layer LSTM + Attention | ~29.2M | ✅ Eğitildi |
| **Swin-B + DistilGPT-2** | Swin Transformer Base | DistilGPT-2 | ~171.4M | ✅ Eğitildi |

## 🧠 Mimariler

### 1. CNN-LSTM + Attention (Klasik Yaklaşım)
* **Encoder**: ImageNet üzerinde önceden eğitilmiş DenseNet-121. Görüntülerden öznitelik haritası (feature map) çıkarır.
* **Decoder**: 2 katmanlı LSTM + Additive Attention + Context Gating. Çıkarılan öznitelik haritalarına odaklanarak adım adım kelime tahmini yapar.
* **Özellikler**: Sinusoidal positional encoding, teacher forcing ve beam search algoritmaları entegre edilmiştir.

### 2. Swin-B + DistilGPT-2 (Modern Transformer Yaklaşımı)
* **Encoder**: `microsoft/swin-base-patch4-window7-224` (Swin Transformer Base)
* **Decoder**: `distilgpt2` (Damıtılmış GPT-2 mimarisi)
* **Özellikler**: HuggingFace `VisionEncoderDecoderModel` mimarisi tabanlıdır. Karışık hassasiyetli eğitim (mixed precision - AMP), gradyan biriktirme (gradient accumulation), ve diferansiyel öğrenme oranları (differential learning rates - Swin encoder için çok düşük, DistilGPT-2 decoder için standart öğrenme hızı) barındırır.

## 📊 Veri Seti

| Split | Örnek Sayısı |
|-------|-------------|
| Train | 178,221 |
| Val | ~18,000 |
| Test | ~18,000 |

**Kaynak**: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) — Beth Israel Deaconess Medical Center.

---

## 📁 Proje Yapısı

```
├── config.py                         # Tüm konfigürasyonlar (CNN-LSTM, ViT & Swin)
├── train_cnnlstm.py                  # CNN-LSTM eğitim scripti
├── test_cnnlstm.py                   # CNN-LSTM test scripti
├── cnn_lstm_predict.py               # Eğitilmiş CNN-LSTM modeli ile çıkarım ve görselleştirme
├── train_swin_distilgpt2.py          # Swin-B + DistilGPT-2 eğitim scripti
├── test_swin_distilgpt2.py           # Swin-B + DistilGPT-2 test scripti
├── predict_swin_distilgpt2.py        # Swin-B + DistilGPT-2 çıkarım ve doğruluk analizi
├── plot_loss.py                      # Eğitim kayıpları (loss) grafik çizim scripti
├── visual_grounding_check.py         # Görsel hizalama ve dikkat (grounding) doğrulaması
├── requirements.txt                  # Python bağımlılıkları ve kütüphaneler
│
├── src/                              # Kaynak kodlar
│   ├── models/
│   │   ├── cnn_lstm.py               # CNN-LSTM model mimarisi tanımı
│   │   └── swin_distilgpt2.py        # Swin-B + DistilGPT-2 model mimarisi tanımı
│   │
│   ├── data_loader/
│   │   ├── dataset_cnnlstm.py        # CNN-LSTM için Dataset & DataLoader
│   │   ├── dataset_swin.py           # Swin-B + DistilGPT-2 için özel Dataset ve DataLoader
│   │   └── vocabulary.py             # CNN-LSTM için kelime haznesi (vocab) yönetimi
│   │
│   └── utils/
│       ├── early_stopping.py         # Erken durdurma (Early stopping) mekanizması
│       └── visualization.py          # Sonuçların görselleştirilmesi
│
├── Data/
│   ├── processed/                    # Ön işlenmiş CSV verileri (train/val/test split'leri)
│   └── vocab/                        # Oluşturulan vocabulary.pkl dosyası
│
├── ProcessData/                      # Ham veriyi analiz eden ve bölümlere ayıran scriptler
├── checkpoints_densenet_findings/     # CNN-LSTM model ağırlıkları (checkpoint)
└── checkpoints_swin_distilgpt2/      # Swin-B + DistilGPT-2 model ağırlıkları (checkpoint)
```

---

## 🚀 Kurulum ve Kullanım

### Bağımlılıkların Yüklenmesi
Gerekli kütüphaneleri yüklemek için aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
```

### 1. CNN-LSTM Modeli Kullanımı
Modeli eğitmek ve test etmek için:
```bash
# Eğitimi başlatır
python train_cnnlstm.py

# Test seti üzerinde değerlendirir
python test_cnnlstm.py

# Görüntü bazlı rapor üretir ve görselleştirme yapar
python cnn_lstm_predict.py
```

### 2. Swin-B + DistilGPT-2 Modeli Kullanımı
Transformer modelini eğitmek ve test etmek için:
```bash
# Eğitimi başlatır
python train_swin_distilgpt2.py

# Test seti üzerinde metrik değerlendirmesi yapar
python test_swin_distilgpt2.py

# Rastgele görseller seçerek rapor üretimi yapar
python predict_swin_distilgpt2.py
```

### 3. Analiz ve Yardımcı Araçlar
Eğitim grafiklerini çizmek veya görsel grounding test etmek için:
```bash
# Eğitim loglarından loss grafiklerini oluşturur
python plot_loss.py

# Görsel dikkat / grounding mekanizmasını test eder
python visual_grounding_check.py
```

---

## 📐 Değerlendirme Metrikleri

Modellerin ürettiği raporlar klinik terimler ve dilbilgisi açısından aşağıdaki metriklerle değerlendirilmektedir:
- **BLEU (1-4)**: N-gram kelime eşleşmeleri
- **ROUGE (1, 2, L)**: Rapor özetleme başarısı ve kelime dizilimi doğruluğu
- **METEOR**: Eşanlamlı kelimeler ve morfolojik varyasyonları dikkate alan eşleşme kalitesi
- **CIDEr**: Konsensüs tabanlı görüntü açıklama değerlendirmesi (radyolojik terimlerin sıklığına göre ağırlıklandırılmış)

## ⚙️ Konfigürasyon Yönetimi

Tüm model mimarileri, hiperparametreler ve veri yolları merkezi olarak `config.py` dosyasından yönetilmektedir:
* Öğrenme oranları (encoder/decoder için diferansiyel oranlar)
* Batch size, epoch sayısı ve erken durdurma (early stopping) sabır eşiği
* Kayıt aralıkları ve log klasör tanımlamaları

---

## 📜 Lisans

Bu proje akademik ve bitirme projesi araştırma amaçlarıyla geliştirilmiştir.
