import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Metrik kütüphaneleri
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

# Proje importları
from config import Config
from src.data_loader.dataset import create_vocabulary_and_dataloaders
from src.models.cnn_lstm import ImageCaptioningModel

# NLTK gerekli kaynaklar
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)


# =========================================================================
# 🔧 YARDIMCI FONKSİYONLAR
# =========================================================================

def ids_to_text(token_ids, vocab, skip_special=True):
    """Token ID listesini metne çevirir"""
    special = {vocab.word2idx["<PAD>"], vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"]}
    words = []
    for idx in token_ids:
        if skip_special and idx in special:
            continue
        word = vocab.idx2word.get(idx, "<UNK>")
        words.append(word)
    return " ".join(words)


def truncate_at_eos(token_ids, eos_id):
    """EOS token'dan sonrasını kes"""
    if eos_id in token_ids:
        return token_ids[:token_ids.index(eos_id)]
    return token_ids


def load_model(checkpoint_path, vocab_size, device):
    """Checkpoint'ten modeli yükle"""
    print(f"📥 Model yükleniyor: {checkpoint_path}")

    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=Config.CNN_EMBED_SIZE,
        hidden_size=Config.CNN_HIDDEN_SIZE,
        attention_dim=Config.CNN_ATTENTION_DIM,
        num_layers=2,
        dropout=0.0,          # Test sırasında dropout kapalı
        freeze_backbone=False
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    loss  = checkpoint.get("loss", "?")
    print(f"✅ Model yüklendi! (Epoch: {epoch}, Val Loss: {loss:.4f})")
    return model


# =========================================================================
# 🚀 BEAM SEARCH
# =========================================================================

def beam_search(model, features, vocab, beam_size=5, max_len=150):
    """
    Tek bir görüntü için Beam Search ile caption üret.

    Beam Search mantığı:
    - Her adımda beam_size kadar en iyi hipotezi tut
    - Her hipotez için vocab_size kadar olasılık hesapla
    - En yüksek log-prob'a sahip beam_size tanesini seç
    - EOS üretilen beam'i tamamlanmış olarak işaretle
    - En sonunda en yüksek skora sahip tamamlanmış beam'i döndür

    Args:
        model:      ImageCaptioningModel (eval modunda)
        features:   [1, seq_len, embed_size]  ← tek görüntü
        vocab:      Vocabulary nesnesi
        beam_size:  Kaç beam tutulacak (5 önerilir)
        max_len:    Maksimum token sayısı

    Returns:
        best_seq: list of int  ← en iyi token ID dizisi
    """
    device   = features.device
    sos_id   = vocab.word2idx["<SOS>"]
    eos_id   = vocab.word2idx["<EOS>"]
    vocab_sz = len(vocab)

    decoder = model.decoder

    # --- Başlangıç ---
    # Her beam: (log_prob, token_ids, h, c)
    h, c = decoder.init_hidden(features)  # [num_layers, 1, hidden_size]

    # İlk token: SOS
    init_token = torch.tensor([sos_id], device=device)  # [1]

    beams = [(0.0, [sos_id], h, c)]   # (score, sequence, h, c)
    completed = []                     # Tamamlanan beam'ler

    for t in range(max_len):
        new_beams = []

        for score, seq, h, c in beams:
            # Son token
            last_token = torch.tensor([seq[-1]], device=device)  # [1]

            # Embedding + Attention + LSTM
            emb     = decoder.embedding(last_token)               # [1, embed_size]
            context, _ = decoder.attention(features, h[-1])      # [1, embed_size]

            lstm_input = torch.cat([emb, context], dim=1).unsqueeze(1)  # [1, 1, 2*embed]
            out, (h_new, c_new) = decoder.lstm(lstm_input, (h, c))
            out = out.squeeze(1)                                  # [1, hidden_size]

            combined = torch.cat([out, context], dim=1)           # [1, hidden+embed]
            logits   = decoder.fc(combined)                       # [1, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)         # [1, vocab_size]

            # Top beam_size token seç
            topk_log_probs, topk_ids = log_probs[0].topk(beam_size)

            for log_p, tok_id in zip(topk_log_probs, topk_ids):
                tok_id = tok_id.item()
                new_score = score + log_p.item()
                new_seq   = seq + [tok_id]

                if tok_id == eos_id:
                    # EOS → tamamlandı, uzunluğa göre normalize et
                    normalized = new_score / len(new_seq)
                    completed.append((normalized, new_seq))
                else:
                    new_beams.append((new_score, new_seq, h_new, c_new))

        if not new_beams:
            break

        # En iyi beam_size tanesini tut
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        # Yeterince tamamlanan beam varsa dur
        if len(completed) >= beam_size:
            break

    # Tamamlananlar arasından en iyiyi seç
    # Hiç tamamlanmadıysa aktif beam'lerin en iyisini al
    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        best_seq = completed[0][1]
    else:
        beams.sort(key=lambda x: x[0], reverse=True)
        best_seq = beams[0][1]

    # SOS'u çıkar, EOS'tan sonrasını kes
    best_seq = best_seq[1:]  # SOS'u at
    best_seq = truncate_at_eos(best_seq, eos_id)

    return best_seq


def generate_predictions(model, loader, vocab, device,
                          beam_size=5, max_len=150):
    """
    Tüm test seti için Beam Search ile tahmin üret.

    Args:
        beam_size: Beam genişliği (5 önerilir, artırınca kalite↑ hız↓)

    Returns:
        predictions: list of str
        references:  list of str
    """
    model.eval()
    predictions = []
    references  = []

    encoder  = model.encoder

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"🔮 Beam Search (k={beam_size})", ncols=100):
            imgs     = batch['images'].to(device)   # [B, 3, H, W]
            captions = batch['captions']             # CPU

            # Encoder: tüm batch'i bir seferde çalıştır (hız için)
            features_batch = encoder(imgs)           # [B, seq_len, embed_size]

            B = imgs.size(0)
            for i in range(B):
                # Her görüntü için ayrı beam search
                features_i = features_batch[i].unsqueeze(0)  # [1, seq_len, embed_size]

                pred_ids  = beam_search(model, features_i, vocab,
                                        beam_size=beam_size, max_len=max_len)
                pred_text = ids_to_text(pred_ids, vocab)

                ref_ids   = captions[i].tolist()
                ref_text  = ids_to_text(ref_ids, vocab)

                predictions.append(pred_text)
                references.append(ref_text)

    return predictions, references


# =========================================================================
# 📐 METRİK HESAPLAMALARI
# =========================================================================

def compute_bleu(predictions, references):
    """BLEU-1, BLEU-2, BLEU-3, BLEU-4 hesapla"""
    smooth = SmoothingFunction().method1

    # Token listelerine çevir
    refs_tokenized  = [[ref.split()] for ref in references]
    hyps_tokenized  = [pred.split() for pred in predictions]

    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1,0,0,0), smoothing_function=smooth)
    bleu2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5,0.5,0,0), smoothing_function=smooth)
    bleu3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.33,0.33,0.33,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)

    return {
        "BLEU-1": round(bleu1, 4),
        "BLEU-2": round(bleu2, 4),
        "BLEU-3": round(bleu3, 4),
        "BLEU-4": round(bleu4, 4),
    }


def compute_rouge(predictions, references):
    """ROUGE-1, ROUGE-2, ROUGE-L hesapla"""
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
                          total=len(predictions), 
                          ncols=100):
        # meteor_score token listesi bekliyor
        score = meteor_score([ref.split()], pred.split())
        scores.append(score)

    return {"METEOR": round(np.mean(scores), 4)}


def compute_cider(predictions, references):
    """CIDEr hesapla (pycocoevalcap formatı)"""
    # CIDEr dict formatı: {id: [caption]}
    gts  = {i: [references[i]]  for i in range(len(references))}
    res  = {i: [predictions[i]] for i in range(len(predictions))}

    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)

    return {"CIDEr": round(score, 4)}


# =========================================================================
# 📊 RAPOR YAZICI
# =========================================================================

def print_results(metrics: dict, num_samples: int):
    """Sonuçları güzel formatta yazdır"""
    print(f"\n{'='*70}")
    print(f"📊 TEST SONUÇLARI  ({num_samples} örnek)")
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
    metrics_path = save_dir / f"test_metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"💾 Metrikler kaydedildi: {metrics_path}")

    # Örnek tahminler TXT
    samples_path = save_dir / f"test_samples_{timestamp}.txt"
    with open(samples_path, 'w', encoding='utf-8') as f:
        f.write(f"TEST SONUÇLARI - {timestamp}\n")
        f.write("="*80 + "\n\n")
        for i in range(min(50, len(predictions))):
            f.write(f"[{i+1}]\n")
            f.write(f"REFERANS : {references[i]}\n")
            f.write(f"TAHMİN   : {predictions[i]}\n")
            f.write("-"*80 + "\n")
    print(f"💾 Örnek tahminler kaydedildi: {samples_path}")

    return metrics_path, samples_path


# =========================================================================
# 🎯 ANA FONKSİYON
# =========================================================================

def main():
    Config.setup()
    device = Config.DEVICE

    print(f"\n{'='*70}")
    print(f"🧪 MODEL TEST BAŞLIYOR")
    print(f"{'='*70}")
    print(f"📱 Cihaz:      {device}")
    print(f"📂 Checkpoint: {Config.CHECKPOINT_FILE}")
    print(f"{'='*70}\n")

    # --- VERİ ---
    print("📥 Test verisi yükleniyor...")
    _, _, test_loader, vocab = create_vocabulary_and_dataloaders(
        data_dir=Config.DATA_DIR,
        image_root=Config.IMAGE_DIR,
        vocab_path=Config.VOCAB_PATH,
        batch_size=32,           # Test için daha büyük batch olabilir
        num_workers=0,
        build_vocab=False,       # Vocab zaten hazır
        freq_threshold=2,
    )
    print(f"✅ Test seti: {len(test_loader.dataset):,} örnek, {len(test_loader)} batch\n")

    # --- MODEL ---
    model = load_model(Config.CHECKPOINT_FILE, len(vocab), device)

    # --- TAHMİN ---
    # Beam size: 5 önerilir. Artırınca kalite artar ama yavaşlar.
    # 3 → hızlı,  5 → dengeli,  10 → yavaş ama en iyi kalite
    BEAM_SIZE = 5

    print(f"\n🔮 Beam Search başlıyor (beam_size={BEAM_SIZE})...")
    print(f"⚠️  Greedy'ye göre ~{BEAM_SIZE}x daha yavaş, kalite daha iyi.\n")
    predictions, references = generate_predictions(
        model, test_loader, vocab, device,
        beam_size=BEAM_SIZE, max_len=150
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
    save_results(all_metrics, predictions, references, Config.CHECKPOINT_DIR)

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