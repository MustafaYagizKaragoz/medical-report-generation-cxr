"""
🧪 Transformer (Swin + BioGPT) Test Scripti
==============================================
Eğitilmiş Transformer modelini test eder ve metriklerini hesaplar.

Kullanım:
    python test_transformer.py

Metrikler:
    - BLEU-1/2/3/4
    - ROUGE-1/2/L
    - METEOR
    - CIDEr
"""

import torch
from tqdm import tqdm
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Metrik kütüphaneleri
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

# Proje importları
from config import Config
from src.data_loader.data_transformer import get_transformer_dataloaders
from src.models.transformer_model import MedicalTransformer

# NLTK kaynakları
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)


# =========================================================================
# 🚀 TAHMİN ÜRETİMİ
# =========================================================================

def generate_predictions(model, loader, device, max_length=150, num_beams=5):
    """
    Test seti üzerinde rapor üret.
    
    HuggingFace'in dahili beam search'ünü kullanır.
    
    Args:
        model: MedicalTransformer modeli
        loader: Test DataLoader
        device: torch device
        max_length: Maksimum rapor uzunluğu
        num_beams: Beam search genişliği
    
    Returns:
        predictions: List[str] — Üretilen raporlar
        references: List[str] — Gerçek raporlar
    """
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"🔮 Rapor Üretimi (beam={num_beams})", ncols=120):
            images = batch['images'].to(device)
            raw_texts = batch['raw_texts']  # Referans metinler
            
            # HuggingFace generate — dahili beam search
            generated_texts = model.generate_report(
                pixel_values=images,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False
            )
            
            predictions.extend(generated_texts)
            references.extend(raw_texts)
    
    return predictions, references


# =========================================================================
# 📐 METRİK HESAPLAMALARI
# =========================================================================

def compute_bleu(predictions, references):
    """BLEU-1/2/3/4 hesapla"""
    smooth = SmoothingFunction().method1
    
    refs_tokenized = [[ref.split()] for ref in references]
    hyps_tokenized = [pred.split() for pred in predictions]
    
    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, 
                        weights=(1,0,0,0), smoothing_function=smooth)
    bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, 
                        weights=(0.5,0.5,0,0), smoothing_function=smooth)
    bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, 
                        weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, 
                        weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
    
    return {
        "BLEU-1": round(bleu1, 4),
        "BLEU-2": round(bleu2, 4),
        "BLEU-3": round(bleu3, 4),
        "BLEU-4": round(bleu4, 4),
    }


def compute_rouge(predictions, references):
    """ROUGE-1/2/L hesapla"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_list, r2_list, rl_list = [], [], []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores['rouge1'].fmeasure)
        r2_list.append(scores['rouge2'].fmeasure)
        rl_list.append(scores['rougeL'].fmeasure)
    
    return {
        "ROUGE-1": round(np.mean(r1_list), 4),
        "ROUGE-2": round(np.mean(r2_list), 4),
        "ROUGE-L": round(np.mean(rl_list), 4),
    }


def compute_meteor(predictions, references):
    """METEOR hesapla"""
    scores = []
    for pred, ref in tqdm(zip(predictions, references), 
                          desc="METEOR hesaplanıyor", 
                          total=len(predictions), ncols=100):
        score = meteor_score([ref.split()], pred.split())
        scores.append(score)
    
    return {"METEOR": round(np.mean(scores), 4)}


def compute_cider(predictions, references):
    """CIDEr hesapla"""
    gts = {i: [references[i]] for i in range(len(references))}
    res = {i: [predictions[i]] for i in range(len(predictions))}
    
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    
    return {"CIDEr": round(score, 4)}


# =========================================================================
# 📊 SONUÇ YAZICI
# =========================================================================

def print_results(metrics: dict, num_samples: int):
    """Sonuçları güzel formatta yazdır"""
    print(f"\n{'='*70}")
    print(f"📊 TRANSFORMER TEST SONUÇLARI  ({num_samples} örnek)")
    print(f"{'='*70}")
    
    sections = {
        "BLEU":   ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"],
        "ROUGE":  ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "METEOR": ["METEOR"],
        "CIDEr":  ["CIDEr"],
    }
    
    for section, keys in sections.items():
        print(f"\n  📌 {section}")
        for k in keys:
            if k in metrics:
                bar_len = int(metrics[k] * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"     {k:<10} {metrics[k]:.4f}  [{bar}]")
    
    print(f"\n{'='*70}\n")


def save_results(metrics, predictions, references, save_dir):
    """Sonuçları JSON ve TXT olarak kaydet"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Metrikler JSON
    metrics_path = save_dir / f"transformer_test_metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"💾 Metrikler: {metrics_path}")
    
    # Örnek tahminler TXT
    samples_path = save_dir / f"transformer_test_samples_{timestamp}.txt"
    with open(samples_path, 'w', encoding='utf-8') as f:
        f.write(f"TRANSFORMER TEST SONUÇLARI - {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Metrikleri de üste yaz
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        for i in range(min(50, len(predictions))):
            f.write(f"[{i+1}]\n")
            f.write(f"REFERANS : {references[i]}\n")
            f.write(f"TAHMİN   : {predictions[i]}\n")
            f.write("-" * 80 + "\n")
    print(f"💾 Örnek tahminler: {samples_path}")
    
    return metrics_path, samples_path


# =========================================================================
# 🎯 ANA FONKSİYON
# =========================================================================

def main():
    Config.setup()
    device = Config.DEVICE
    
    # Model yolu
    model_path = os.path.join(Config.TRANSFORMER_CHECKPOINT_DIR, "best_model")
    
    print(f"\n{'='*70}")
    print(f"🧪 TRANSFORMER MODEL TEST BAŞLIYOR")
    print(f"{'='*70}")
    print(f"📱 Cihaz:      {device}")
    print(f"📂 Model:      {model_path}")
    print(f"📊 Test CSV:   {Config.TEST_PROCESSED_CSV}")
    print(f"{'='*70}\n")
    
    # --- MODEL YÜKLEME ---
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        print(f"💡 Önce eğitimi tamamlayın: python train_transformer.py")
        return
    
    print("📥 Model yükleniyor...")
    model = MedicalTransformer.from_pretrained(model_path, freeze_encoder=False)
    model = model.to(device)
    model.eval()
    
    # --- TEST VERİSİ ---
    print("📥 Test verisi yükleniyor...")
    _, _, test_loader, tokenizer = get_transformer_dataloaders(
        train_csv=Config.TRAIN_PROCESSED_CSV,
        val_csv=Config.VAL_PROCESSED_CSV,
        test_csv=Config.TEST_PROCESSED_CSV,
        image_dir=Config.IMAGE_DIR,
        batch_size=Config.TRANSFORMER_BATCH_SIZE,
        max_length=Config.TRANSFORMER_MAX_LENGTH,
        image_size=Config.TRANSFORMER_IMAGE_SIZE,
        num_workers=0
    )
    print(f"✅ Test seti: {len(test_loader.dataset):,} örnek, {len(test_loader)} batch\n")
    
    # --- TAHMİN ÜRETİMİ ---
    BEAM_SIZE = getattr(Config, 'TRANSFORMER_NUM_BEAMS', 5)
    MAX_LENGTH = getattr(Config, 'TRANSFORMER_MAX_LENGTH', 150)
    
    print(f"🔮 Rapor üretimi başlıyor (beam_size={BEAM_SIZE})...")
    print(f"   HuggingFace dahili beam search kullanılıyor.\n")
    
    predictions, references = generate_predictions(
        model, test_loader, device,
        max_length=MAX_LENGTH, num_beams=BEAM_SIZE
    )
    print(f"✅ {len(predictions):,} tahmin üretildi\n")
    
    # --- METRİKLER ---
    print("📐 Metrikler hesaplanıyor...\n")
    all_metrics = {}
    
    print("  ▶ BLEU hesaplanıyor...")
    all_metrics.update(compute_bleu(predictions, references))
    
    print("  ▶ ROUGE hesaplanıyor...")
    all_metrics.update(compute_rouge(predictions, references))
    
    print("  ▶ METEOR hesaplanıyor...")
    all_metrics.update(compute_meteor(predictions, references))
    
    print("  ▶ CIDEr hesaplanıyor...")
    all_metrics.update(compute_cider(predictions, references))
    
    # --- SONUÇLAR ---
    print_results(all_metrics, len(predictions))
    
    # --- KAYDET ---
    save_results(
        all_metrics, predictions, references, 
        Config.TRANSFORMER_CHECKPOINT_DIR
    )
    
    # --- ÖRNEK TAHMİNLER ---
    print(f"\n{'='*70}")
    print("🔍 ÖRNEK TAHMİNLER (İlk 5)")
    print(f"{'='*70}")
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"  REFERANS : {references[i]}")
        print(f"  TAHMİN   : {predictions[i]}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
