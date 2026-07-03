import pandas as pd
import numpy as np
import sys
import re
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None

# Ensure console supports UTF-8 on Windows
if sys.platform.startswith("win"):
    import io
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Separate punctuation characters with spaces (Standard NLP Tokenization)
    text = re.sub(r'([.,!?();:])', r' \1 ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def compute_bleu(predictions, references):
    smooth = SmoothingFunction().method1
    refs_tokenized = [[r.split()] for r in references]
    hyps_tokenized = [p.split() for p in predictions]

    print("  .. Calculating BLEU-1")
    b1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1, 0, 0, 0), smoothing_function=smooth)
    print("  .. Calculating BLEU-2")
    b2 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    print("  .. Calculating BLEU-3")
    b3 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    print("  .. Calculating BLEU-4")
    b4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "BLEU-1": round(b1, 4),
        "BLEU-2": round(b2, 4),
        "BLEU-3": round(b3, 4),
        "BLEU-4": round(b4, 4),
    }

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_list, r2_list, rl_list = [], [], []

    for p, r in tqdm(zip(predictions, references), total=len(predictions), desc="  .. Calculating ROUGE-1/2/L"):
        scores = scorer.score(r, p)
        r1_list.append(scores['rouge1'].fmeasure)
        r2_list.append(scores['rouge2'].fmeasure)
        rl_list.append(scores['rougeL'].fmeasure)

    return {
        "ROUGE-1": round(np.mean(r1_list), 4),
        "ROUGE-2": round(np.mean(r2_list), 4),
        "ROUGE-L": round(np.mean(rl_list), 4),
    }

def compute_sbert_similarity(predictions, references):
    if SentenceTransformer is None:
        print("⚠️  sentence-transformers is not installed")
        return {"SBERT": 0.0}
    print("  .. Calculating SBERT Cosine Similarity")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    scores = []
    bs = 64
    for i in tqdm(range(0, len(predictions), bs), desc="  .. SBERT", ncols=100):
        p_b = predictions[i:i+bs]
        r_b = references[i:i+bs]
        e1  = sbert.encode(p_b, convert_to_tensor=True)
        e2  = sbert.encode(r_b, convert_to_tensor=True)
        cos = st_util.cos_sim(e1, e2)
        for j in range(len(p_b)):
            scores.append(cos[j][j].item())
    del sbert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"SBERT": round(float(np.mean(scores)), 4)}

def main():
    csv_path = "test_predictions_combined.csv"
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing values in reference or predictions
    df = df.dropna(subset=["Reference Report", "CNN-LSTM Prediction", "Swin Prediction"])
    
    # Clean text to make sure spacing is normalized
    refs = [clean_text(r) for r in df["Reference Report"].tolist()]
    cnn_preds = [clean_text(p) for p in df["CNN-LSTM Prediction"].tolist()]
    swin_preds = [clean_text(p) for p in df["Swin Prediction"].tolist()]
    
    print(f"Loaded {len(refs)} valid predictions for evaluation.")
    
    # ── CNN-LSTM Metrics ──────────────────────────────────────────────────
    print("\n📊 Evaluating CNN-LSTM...")
    cnn_bleu = compute_bleu(cnn_preds, refs)
    cnn_rouge = compute_rouge(cnn_preds, refs)
    cnn_sbert = compute_sbert_similarity(cnn_preds, refs)
    
    # ── Swin-GPT Metrics ──────────────────────────────────────────────────
    print("\n📊 Evaluating Swin-GPT (MTL)...")
    swin_bleu = compute_bleu(swin_preds, refs)
    swin_rouge = compute_rouge(swin_preds, refs)
    swin_sbert = compute_sbert_similarity(swin_preds, refs)
    
    # ── Print Results ────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("📈 FINAL BENCHMARK NLP METRICS COMPARISON")
    print("="*50)
    print(f"{'Metric':<10} | {'CNN-LSTM':<12} | {'Swin-GPT (MTL)':<15}")
    print("-"*50)
    
    metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "SBERT"]
    for m in metrics:
        if m in cnn_bleu:
            c_val = cnn_bleu[m]
            s_val = swin_bleu[m]
        elif m in cnn_rouge:
            c_val = cnn_rouge[m]
            s_val = swin_rouge[m]
        else:
            c_val = cnn_sbert[m]
            s_val = swin_sbert[m]
        print(f"{m:<10} | {c_val:<12.4f} | {s_val:<15.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
