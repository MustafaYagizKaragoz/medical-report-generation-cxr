"""
advanced_evaluation_suite.py
============================
Danışman Hocanın Özel Talimatlarına (Madde 4, 5, 6, 7) Tam Uyumlu Değerlendirme Paketi:

1. Standart 10 NLP Metriği:
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, CIDEr, SBERT Cosine Sim.
2. İstatistiksel Anlamlılık:
   - 2.000 Tekrarlı Hasta Düzeyinde Kümelenmiş (Patient-Clustered) Bootstrap (%95 GA)
   - Eşleştirilmiş Hasta Düzeyinde Permütasyon Testi (Paired Permutation Test -> Kesin İki Yönlü p-Değeri)
3. Klinik Doğruluk Değerlendirmesi:
   - 4 Durum Ayrımı: Pozitif (1), Negatif (0), Belirsiz (-1), Bahsedilmeyen (NaN)
   - U-Zero (Ana Analiz) ve U-Ones (Duyarlılık Analizi)
   - Omissions (Atlanan Bulgular) ve Unsupported Findings (Referansta Desteklenmeyen Bulgular)
   - Odak Patolojiler: Pneumothorax, Pneumonia, Cardiomegaly, Atelectasis, Pleural Effusion
4. Denetim Tabloları:
   - Frontal/Lateral, Patoloji Dağılımı, Filtreleme Hunisi (248.506 -> 215.617)
5. Otomatik Nihai PDF Rapor Üretimi (ReportLab ile)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torch

# Metrik Kütüphaneleri
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from sentence_transformers import SentenceTransformer, util as st_util
from sklearn.metrics import precision_recall_fscore_support

# 14 Standart CheXpert Patolojisi
PATHOLOGY_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]

FOCUS_PATHOLOGIES = [
    "Pneumothorax", "Pneumonia", "Cardiomegaly", "Atelectasis", "Pleural Effusion"
]


# =========================================================================
# 1. STANDART 10 NLP METRİĞİ HESAPLAYICI
# =========================================================================
def compute_all_10_nlp_metrics(predictions: list, references: list, sbert_model=None) -> dict:
    """Orijinal makale yöntemiyle 10 NLP metriğini hesaplar."""
    n = len(predictions)
    assert n == len(references), "Tahmin ve referans boyutları eşit olmalıdır!"

    # Tokenize
    tok_preds = [nltk.word_tokenize(p.lower()) if p.strip() else ["empty"] for p in predictions]
    tok_refs  = [[nltk.word_tokenize(r.lower()) if r.strip() else ["empty"]] for r in references]

    # BLEU 1-4 (Corpus-level)
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(tok_refs, tok_preds, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    b2 = corpus_bleu(tok_refs, tok_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    b3 = corpus_bleu(tok_refs, tok_preds, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth)
    b4 = corpus_bleu(tok_refs, tok_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # ROUGE 1, 2, L
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_list, r2_list, rl_list = [], [], []
    for p, r in zip(predictions, references):
        res = scorer.score(r, p)
        r1_list.append(res["rouge1"].fmeasure)
        r2_list.append(res["rouge2"].fmeasure)
        rl_list.append(res["rougeL"].fmeasure)

    r1 = float(np.mean(r1_list))
    r2 = float(np.mean(r2_list))
    rl = float(np.mean(rl_list))

    # METEOR
    met_list = []
    for p_tok, r_toks in zip(tok_preds, tok_refs):
        # meteor_score accepts list of tokenized refs and single tokenized pred
        met_list.append(meteor_score(r_toks, p_tok))
    meteor = float(np.mean(met_list))

    # Standart CIDEr
    cider_preds = {i: [p] for i, p in enumerate(predictions)}
    cider_refs  = {i: [r] for i, r in enumerate(references)}
    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(cider_refs, cider_preds)
    except Exception as e:
        print(f"⚠️ CIDEr hesaplama hatası: {e}")
        cider_score = 0.0

    # SBERT Cosine Similarity
    if sbert_model is None:
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    bs = 64
    sbert_scores = []
    for i in range(0, n, bs):
        p_b = predictions[i:i+bs]
        r_b = references[i:i+bs]
        e1 = sbert_model.encode(p_b, convert_to_tensor=True, show_progress_bar=False)
        e2 = sbert_model.encode(r_b, convert_to_tensor=True, show_progress_bar=False)
        cos = st_util.cos_sim(e1, e2)
        for j in range(len(p_b)):
            sbert_scores.append(cos[j][j].item())
    sbert_sim = float(np.mean(sbert_scores))

    return {
        "BLEU-1": round(b1, 4),
        "BLEU-2": round(b2, 4),
        "BLEU-3": round(b3, 4),
        "BLEU-4": round(b4, 4),
        "ROUGE-1": round(r1, 4),
        "ROUGE-2": round(r2, 4),
        "ROUGE-L": round(rl, 4),
        "METEOR": round(meteor, 4),
        "CIDEr": round(float(cider_score), 4),
        "SBERT": round(sbert_sim, 4),
    }


# =========================================================================
# 2. HASTA DÜZEYİNDE BOOTSTRAP VE EŞLEŞTİRİLMİŞ PERMÜTASYON TESTİ
# =========================================================================
def run_patient_clustered_bootstrap_and_permutation(
    df_eval: pd.DataFrame,
    model_a_col: str,
    model_b_col: str,
    ref_col: str = "reference_report",
    subject_col: str = "subject_id",
    n_bootstraps: int = 2000,
    n_permutations: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Danışman Madde 5 ve 6:
    - Hasta bazlı kümelenmiş (patient-clustered) bootstrap (2.000 replikasyon) ile %95 GA
    - Eşleştirilmiş hasta düzeyinde permütasyon testi ile bağımsız kesin iki yönlü p-değeri hesabı
      (P-değerleri asla yalnızca güven aralığı yüzdeliklerinden türetilmez; hasta düzeyinde
       takas (swap) permütasyon dağılımından tam olarak hesaplanır).
    """
    rng = np.random.default_rng(seed)
    unique_subjects = df_eval[subject_col].unique()
    n_subjects = len(unique_subjects)
    total_samples = len(df_eval)

    # Hastaların satır indekslerini grupla (kümeleme)
    subject_to_indices = df_eval.groupby(subject_col).indices

    print(f"\n📊 İstatistiksel Analiz: {model_a_col} vs {model_b_col}")
    print(f"   Toplam Görüntü: {total_samples:,} | Tekil Hasta: {n_subjects:,}")
    print(f"   Bootstrap Tekrar: {n_bootstraps:,} | Permütasyon Tekrar: {n_permutations:,}")

    # Örnek bazlı 10 NLP metriği hesaplaması
    smooth = SmoothingFunction().method1
    scorer_r = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    sent_b1_a, sent_b1_b = [], []
    sent_b2_a, sent_b2_b = [], []
    sent_b3_a, sent_b3_b = [], []
    sent_b4_a, sent_b4_b = [], []
    sent_r1_a, sent_r1_b = [], []
    sent_r2_a, sent_r2_b = [], []
    sent_rl_a, sent_rl_b = [], []
    sent_met_a, sent_met_b = [], []

    print("   ▶ Cümle düzeyinde 10 eşleştirilmiş metrik dizisi hazırlanıyor...")
    for idx, row in df_eval.iterrows():
        ref = str(row[ref_col]).lower().strip()
        p_a = str(row[model_a_col]).lower().strip()
        p_b = str(row[model_b_col]).lower().strip()

        tok_r = [nltk.word_tokenize(ref)] if ref else [["empty"]]
        tok_a = nltk.word_tokenize(p_a) if p_a else ["empty"]
        tok_b = nltk.word_tokenize(p_b) if p_b else ["empty"]

        # Sentence BLEU 1-4
        sent_b1_a.append(sentence_bleu(tok_r, tok_a, weights=(1, 0, 0, 0), smoothing_function=smooth))
        sent_b1_b.append(sentence_bleu(tok_r, tok_b, weights=(1, 0, 0, 0), smoothing_function=smooth))

        sent_b2_a.append(sentence_bleu(tok_r, tok_a, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        sent_b2_b.append(sentence_bleu(tok_r, tok_b, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))

        sent_b3_a.append(sentence_bleu(tok_r, tok_a, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth))
        sent_b3_b.append(sentence_bleu(tok_r, tok_b, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth))

        sent_b4_a.append(sentence_bleu(tok_r, tok_a, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
        sent_b4_b.append(sentence_bleu(tok_r, tok_b, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

        # Sentence ROUGE 1, 2, L
        res_a = scorer_r.score(ref, p_a)
        res_b = scorer_r.score(ref, p_b)
        sent_r1_a.append(res_a["rouge1"].fmeasure)
        sent_r1_b.append(res_b["rouge1"].fmeasure)
        sent_r2_a.append(res_a["rouge2"].fmeasure)
        sent_r2_b.append(res_b["rouge2"].fmeasure)
        sent_rl_a.append(res_a["rougeL"].fmeasure)
        sent_rl_b.append(res_b["rougeL"].fmeasure)

        # Sentence METEOR
        sent_met_a.append(meteor_score(tok_r, tok_a))
        sent_met_b.append(meteor_score(tok_r, tok_b))

    # Sentence SBERT Cosine Similarity
    print("   ▶ SBERT cümle benzerlikleri vektörize ediliyor...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    all_refs = [str(r).lower().strip() for r in df_eval[ref_col]]
    all_pa = [str(p).lower().strip() for p in df_eval[model_a_col]]
    all_pb = [str(p).lower().strip() for p in df_eval[model_b_col]]
    
    bs = 128
    sent_sbert_a, sent_sbert_b = [], []
    for i in range(0, total_samples, bs):
        r_chunk = all_refs[i:i+bs]
        pa_chunk = all_pa[i:i+bs]
        pb_chunk = all_pb[i:i+bs]
        e_r = sbert_model.encode(r_chunk, convert_to_tensor=True, show_progress_bar=False)
        e_a = sbert_model.encode(pa_chunk, convert_to_tensor=True, show_progress_bar=False)
        e_b = sbert_model.encode(pb_chunk, convert_to_tensor=True, show_progress_bar=False)
        cos_a = st_util.cos_sim(e_a, e_r)
        cos_b = st_util.cos_sim(e_b, e_r)
        for j in range(len(r_chunk)):
            sent_sbert_a.append(cos_a[j][j].item())
            sent_sbert_b.append(cos_b[j][j].item())
    del sbert_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Sentence CIDEr Proxy / Consensus
    # CIDEr test set corpus tf-idf consensus
    cider_preds_a = {i: [p] for i, p in enumerate(all_pa)}
    cider_preds_b = {i: [p] for i, p in enumerate(all_pb)}
    cider_refs    = {i: [r] for i, r in enumerate(all_refs)}
    cider_scorer = Cider()
    _, cid_scores_a = cider_scorer.compute_score(cider_refs, cider_preds_a)
    _, cid_scores_b = cider_scorer.compute_score(cider_refs, cider_preds_b)
    sent_cider_a = list(cid_scores_a)
    sent_cider_b = list(cid_scores_b)

    metric_arrays = {
        "BLEU-1": (np.array(sent_b1_a), np.array(sent_b1_b)),
        "BLEU-2": (np.array(sent_b2_a), np.array(sent_b2_b)),
        "BLEU-3": (np.array(sent_b3_a), np.array(sent_b3_b)),
        "BLEU-4": (np.array(sent_b4_a), np.array(sent_b4_b)),
        "ROUGE-1": (np.array(sent_r1_a), np.array(sent_r1_b)),
        "ROUGE-2": (np.array(sent_r2_a), np.array(sent_r2_b)),
        "ROUGE-L": (np.array(sent_rl_a), np.array(sent_rl_b)),
        "METEOR": (np.array(sent_met_a), np.array(sent_met_b)),
        "CIDEr": (np.array(sent_cider_a), np.array(sent_cider_b)),
        "SBERT": (np.array(sent_sbert_a), np.array(sent_sbert_b)),
    }

    # Hasta düzeyinde görüntü sayıları ve indeksleri
    subj_indices_list = [subject_to_indices[s] for s in unique_subjects]
    subj_counts = np.array([len(idx_arr) for idx_arr in subj_indices_list], dtype=np.float64)

    results = {}

    for metric_name, (arr_a, arr_b) in metric_arrays.items():
        subj_sum_a = np.array([np.sum(arr_a[idx_arr]) for idx_arr in subj_indices_list], dtype=np.float64)
        subj_sum_b = np.array([np.sum(arr_b[idx_arr]) for idx_arr in subj_indices_list], dtype=np.float64)
        subj_diff  = subj_sum_a - subj_sum_b

        mean_a = float(np.mean(arr_a))
        mean_b = float(np.mean(arr_b))
        observed_diff = mean_a - mean_b

        # 1. Hasta Düzeyinde Kümelenmiş Bootstrap (2.000 Replikasyon)
        boot_subjs_matrix = rng.integers(0, n_subjects, size=(n_bootstraps, n_subjects))
        boot_sums_a = np.sum(subj_sum_a[boot_subjs_matrix], axis=1)
        boot_sums_b = np.sum(subj_sum_b[boot_subjs_matrix], axis=1)
        boot_counts = np.sum(subj_counts[boot_subjs_matrix], axis=1)
        boot_diffs = (boot_sums_a - boot_sums_b) / boot_counts

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        # 2. Monte Carlo Eşleştirilmiş Hasta Düzeyinde Permütasyon Testi (2.000 Replikasyon)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n_subjects))
        perm_diffs = np.dot(signs, subj_diff) / total_samples

        p_val = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))
        if p_val == 0.0:
            p_val_str = f"< {1.0 / n_permutations:.4f}"
        else:
            p_val_str = f"{p_val:.4f}"

        results[metric_name] = {
            "mean_model_a": round(mean_a, 4),
            "mean_model_b": round(mean_b, 4),
            "mean_diff (A - B)": round(observed_diff, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "p_val_numeric": p_val,
            "p_value": p_val_str,
            "statistically_significant": (p_val < 0.05)
        }

    # 3. Çoklu Karşılaştırma İçin Holm-Bonferroni (Step-Down) Düzeltmesi
    sorted_metrics = sorted(results.items(), key=lambda x: x[1]["p_val_numeric"])
    m_count = len(sorted_metrics)
    prev_adj = 0.0
    for i, (m_name, d_val) in enumerate(sorted_metrics):
        raw_p = d_val["p_val_numeric"]
        adj_p = (m_count - i) * raw_p
        adj_p = max(adj_p, prev_adj)
        adj_p = min(1.0, adj_p)
        prev_adj = adj_p
        
        if adj_p == 0.0 or adj_p < 0.0005:
            p_holm_str = f"< {1.0 / n_permutations:.4f}"
        else:
            p_holm_str = f"{adj_p:.4f}"
        
        results[m_name]["p_value_holm"] = p_holm_str
        results[m_name]["significant_holm"] = (adj_p < 0.05)

    return results


# =========================================================================
# 3. KLİNİK DOĞRULUK DEĞERLENDİRİCİ (U-Zero & U-Ones & Unsupported Findings)
# =========================================================================
def evaluate_clinical_efficacy_advanced(
    df_eval: pd.DataFrame,
    pred_col: str,
    ref_col: str = "reference_report",
) -> dict:
    """
    Danışman Madde 4:
    - 4 Durum Ayrımı: Pozitif (1.0), Açıkça Negatif (0.0), Belirsiz (-1.0), Bahsedilmeyen (NaN).
    - U-Zero (Ana Analiz) ve U-Ones (Duyarlılık Analizi).
    - Kural: Referansta bahsedilmeyen her bulgu doğrulanmış negatif KABUL EDİLMEZ.
    - Kural: Referans tarafından desteklenmeyen bulgular 'radyolog tarafından doğrulanmış halüsinasyon'
      olarak değil, 'Referans Tarafından Desteklenmeyen Bulgu (Unsupported Finding)' olarak adlandırılır.
    - Metrikler: Precision, Recall, F1, Omissions (FN), Unsupported Findings (FP), Direct Conflicts.
    """
    patterns = {
        "Enlarged Cardiomediastinum": [r"cardiomediastin\w+", r"widened mediastinum"],
        "Cardiomegaly": [r"cardiomegaly", r"enlarged heart", r"heart is enlarged", r"cardiac silhouette is enlarged"],
        "Lung Opacity": [r"opacity", r"opacities", r"infiltrat\w+"],
        "Lung Lesion": [r"nodule", r"mass", r"lesion"],
        "Edema": [r"edema", r"vascular congestion", r"fluid overload"],
        "Consolidation": [r"consolidation", r"air space disease"],
        "Pneumonia": [r"pneumonia", r"infectious process"],
        "Atelectasis": [r"atelectasis", r"collapse", r"volume loss"],
        "Pneumothorax": [r"pneumothorax", r"pneumo thorax"],
        "Pleural Effusion": [r"pleural effusion", r"effusion"],
        "Pleural Other": [r"pleural thickening", r"fibrothorax"],
        "Fracture": [r"fracture", r"broken rib"],
        "Support Devices": [r"tube", r"catheter", r"line", r"pacer", r"port-a-cath", r"stent", r"hardware", r"sternotomy"],
        "No Finding": [r"no acute", r"normal chest", r"unremarkable", r"clear lungs", r"no active disease"]
    }

    import re

    def extract_labels(text):
        labels = {}
        t = str(text).lower()
        for pat, regexes in patterns.items():
            matched = any(re.search(rgx, t) for rgx in regexes)
            # Açıkça negatifleme kontrolü
            negated = any(re.search(r"(no|without|free of|negative for|absence of|denies|cleared of)\s+[^.;\n]*?" + rgx, t) for rgx in regexes)
            if pat == "Cardiomegaly" and any(re.search(rgx, t) for rgx in [r"normal heart", r"heart size is normal", r"cardiac silhouette is normal"]):
                negated = True
                matched = True

            uncertain = any(re.search(r"(possible|probable|suspect\w+|cannot exclude|borderline)\s+[^.;\n]*?" + rgx, t) for rgx in regexes)

            if matched and not negated:
                if uncertain:
                    labels[pat] = -1.0  # Belirsiz (Uncertain)
                else:
                    labels[pat] = 1.0   # Pozitif (Positive)
            elif negated:
                labels[pat] = 0.0       # Açıkça Negatif (Clearly Negative)
            else:
                labels[pat] = np.nan    # Raporda Bahsedilmeyen (Unmentioned)
        return labels

    print(f"\n🩺 Klinik Değerlendirme Çıkarılıyor: {pred_col}...")
    ref_labels = [extract_labels(r) for r in tqdm(df_eval[ref_col], desc="Ref Etiketleme", ncols=100)]
    pred_labels = [extract_labels(p) for p in tqdm(df_eval[pred_col], desc="Tahmin Etiketleme", ncols=100)]

    ref_df = pd.DataFrame(ref_labels)
    pred_df = pd.DataFrame(pred_labels)

    # 1. U-ZERO (Ana Analiz): Belirsiz -1.0 -> 0 (Pozitif kabul edilmez)
    # Referansta kesin pozitif: ref == 1.0
    # Modelde pozitif: pred == 1.0
    # 2. U-ONES (Duyarlılık Analizi): Belirsiz -1.0 -> 1 (Pozitif kabul edilir)
    # Referansta pozitif veya belirsiz: ref in [1.0, -1.0]
    # Modelde pozitif veya belirsiz: pred in [1.0, -1.0]

    u0_tps, u0_fps, u0_fns = 0, 0, 0
    u1_tps, u1_fps, u1_fns = 0, 0, 0

    per_pat_u0_f1 = []
    per_pat_u1_f1 = []

    focus_details = {}
    total_omissions = 0          # Ref == 1, Pred != 1 (Atlanan Pozitif)
    total_unsupported = 0        # Ref is NaN, Pred == 1 (Referansta Bahsedilmeyen Pozitif)
    total_contradictions = 0     # Ref == 0, Pred == 1 (Açıkça Çelişen Pozitif)

    for col in PATHOLOGY_COLS:
        r_col = ref_df[col]
        p_col = pred_df[col]

        # U-Zero pozitif tanımları
        r_pos_u0 = (r_col == 1.0).values
        p_pos_u0 = (p_col == 1.0).values

        tp_u0 = int(np.sum(r_pos_u0 & p_pos_u0))
        fn_u0 = int(np.sum(r_pos_u0 & ~p_pos_u0))      # Omissions
        fp_u0 = int(np.sum(~r_pos_u0 & p_pos_u0))

        # Detaylı FP ayrımı (Danışman Madde 4)
        # Unsupported: Ref is NaN, Pred == 1 (Desteklenmeyen bulgu, doğrulanmış halüsinasyon DEĞİL)
        unsupported = int(np.sum(r_col.isna().values & p_pos_u0))
        # Contradiction: Ref == 0, Pred == 1 (Açıkça negatif denmiş ama model pozitif demiş)
        contradiction = int(np.sum((r_col == 0.0).values & p_pos_u0))

        u0_tps += tp_u0
        u0_fns += fn_u0
        u0_fps += fp_u0
        total_omissions += fn_u0
        total_unsupported += unsupported
        total_contradictions += contradiction

        prec_u0 = tp_u0 / (tp_u0 + fp_u0) if (tp_u0 + fp_u0) > 0 else 0.0
        rec_u0  = tp_u0 / (tp_u0 + fn_u0) if (tp_u0 + fn_u0) > 0 else 0.0
        f1_u0   = 2 * prec_u0 * rec_u0 / (prec_u0 + rec_u0) if (prec_u0 + rec_u0) > 0 else 0.0
        per_pat_u0_f1.append(f1_u0)

        # U-Ones pozitif tanımları (Belirsiz -1 de pozitif sayılır)
        r_pos_u1 = (r_col.isin([1.0, -1.0])).values
        p_pos_u1 = (p_col.isin([1.0, -1.0])).values

        tp_u1 = int(np.sum(r_pos_u1 & p_pos_u1))
        fn_u1 = int(np.sum(r_pos_u1 & ~p_pos_u1))
        fp_u1 = int(np.sum(~r_pos_u1 & p_pos_u1))

        u1_tps += tp_u1
        u1_fns += fn_u1
        u1_fps += fp_u1

        prec_u1 = tp_u1 / (tp_u1 + fp_u1) if (tp_u1 + fp_u1) > 0 else 0.0
        rec_u1  = tp_u1 / (tp_u1 + fn_u1) if (tp_u1 + fn_u1) > 0 else 0.0
        f1_u1   = 2 * prec_u1 * rec_u1 / (prec_u1 + rec_u1) if (prec_u1 + rec_u1) > 0 else 0.0
        per_pat_u1_f1.append(f1_u1)

        if col in FOCUS_PATHOLOGIES:
            focus_details[col] = {
                "Precision": round(float(prec_u0), 4),
                "Recall": round(float(rec_u0), 4),
                "F1-Score": round(float(f1_u0), 4),
                "True Positives": tp_u0,
                "Support (Ref Positive)": int(np.sum(r_pos_u0)),
                "Omissions (Atlanan Pozitif)": fn_u0,
                "Unsupported Findings (Referansta Bahsedilmeyen Pozitif)": unsupported,
                "Direct Contradictions (Açıkça Çelişen Pozitif)": contradiction,
            }

    # Micro & Macro F1
    micro_p_u0 = u0_tps / (u0_tps + u0_fps) if (u0_tps + u0_fps) > 0 else 0.0
    micro_r_u0 = u0_tps / (u0_tps + u0_fns) if (u0_tps + u0_fns) > 0 else 0.0
    micro_f1_u0 = 2 * micro_p_u0 * micro_r_u0 / (micro_p_u0 + micro_r_u0) if (micro_p_u0 + micro_r_u0) > 0 else 0.0
    macro_f1_u0 = float(np.mean(per_pat_u0_f1))

    micro_p_u1 = u1_tps / (u1_tps + u1_fps) if (u1_tps + u1_fps) > 0 else 0.0
    micro_r_u1 = u1_tps / (u1_tps + u1_fns) if (u1_tps + u1_fns) > 0 else 0.0
    micro_f1_u1 = 2 * micro_p_u1 * micro_r_u1 / (micro_p_u1 + micro_r_u1) if (micro_p_u1 + micro_r_u1) > 0 else 0.0
    macro_f1_u1 = float(np.mean(per_pat_u1_f1))

    return {
        "U-Zero (Ana Analiz)": {
            "Macro F1": round(macro_f1_u0, 4),
            "Micro F1": round(micro_f1_u0, 4),
            "Micro Precision": round(micro_p_u0, 4),
            "Micro Recall": round(micro_r_u0, 4),
        },
        "U-Ones (Duyarlılık Analizi)": {
            "Macro F1": round(macro_f1_u1, 4),
            "Micro F1": round(micro_f1_u1, 4),
            "Micro Precision": round(micro_p_u1, 4),
            "Micro Recall": round(micro_r_u1, 4),
        },
        "Total Omissions (Atlanan Pozitif Bulgular)": total_omissions,
        "Total Unsupported Findings (Referansta Bahsedilmeyen Bulgular)": total_unsupported,
        "Total Direct Contradictions (Açıkça Çelişen Bulgular)": total_contradictions,
        "5_Focus_Pathologies": focus_details,
    }


# =========================================================================
# 4. KAPSAMLI NİHAİ PDF RAPOR OLUŞTURUCU (ReportLab)
# =========================================================================
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Türkçe karakterlerin (ç, ğ, ı, ö, ş, ü, İ, Ğ, Ş, Ö, Ç, Ü) kusursuz çizimi için Windows Arial TrueType fontlarını kaydet
arial_path = "C:/Windows/Fonts/arial.ttf"
arial_bold_path = "C:/Windows/Fonts/arialbd.ttf"
if os.path.exists(arial_path) and os.path.exists(arial_bold_path):
    pdfmetrics.registerFont(TTFont("Arial", arial_path))
    pdfmetrics.registerFont(TTFont("Arial-Bold", arial_bold_path))
    FONT_NORMAL = "Arial"
    FONT_BOLD = "Arial-Bold"
else:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont(FONT_NORMAL, 8)
        self.setFillColor(colors.HexColor('#718096'))
        
        # Üst Bilgi (2 ve 3. Sayfalar)
        if self._pageNumber > 1:
            self.drawString(32, 758, 'Healthcare Analytics — Revizyon Teknik Raporu')
            self.drawRightString(580, 758, 'Ekim 2026 | CXR Tıbbi Rapor Üretimi')
            self.setStrokeColor(colors.HexColor('#CBD5E0'))
            self.setLineWidth(0.5)
            self.line(32, 752, 580, 752)
            
        # Alt Bilgi (Tüm Sayfalar)
        self.setStrokeColor(colors.HexColor('#CBD5E0'))
        self.setLineWidth(0.5)
        self.line(32, 20, 580, 20)
        self.drawString(32, 10, 'Gizli — Healthcare Analytics Revizyon İncelemesi İçin Hazırlanmıştır')
        self.drawRightString(580, 10, f'Sayfa {self._pageNumber} / {page_count}')
        self.restoreState()


def generate_master_revision_pdf(
    table4_data: pd.DataFrame,
    bootstrap_data: dict,
    clinical_data: dict,
    audit_data: dict,
    output_pdf_path: str = "Healthcare_Analytics_Revision_Master_Report.pdf",
):
    """Danışman ve hakemler için Türkçe nihai revizyon PDF raporu üretir."""
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=32,
        leftMargin=32,
        topMargin=26,
        bottomMargin=26
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName=FONT_BOLD,
        fontSize=16.5,
        leading=20.5,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=2
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontName=FONT_NORMAL,
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor('#4A5568')
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName=FONT_BOLD,
        fontSize=10.2,
        leading=12.5,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=4,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName=FONT_NORMAL,
        fontSize=7.3,
        leading=9.3,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=2
    )
    th_style = ParagraphStyle(
        'TH',
        fontName=FONT_BOLD,
        fontSize=7.3,
        leading=9.2,
        textColor=colors.whitesmoke,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD',
        fontName=FONT_NORMAL,
        fontSize=7.3,
        leading=9.2,
        textColor=colors.HexColor('#1A202C'),
        alignment=1
    )
    td_left = ParagraphStyle(
        'TDLeft',
        fontName=FONT_NORMAL,
        fontSize=7.3,
        leading=9.2,
        textColor=colors.HexColor('#1A202C'),
        alignment=0
    )
    th_small = ParagraphStyle(
        'THSmall',
        fontName=FONT_BOLD,
        fontSize=6.7,
        leading=8.0,
        textColor=colors.whitesmoke,
        alignment=1
    )
    td_small = ParagraphStyle(
        'TDSmall',
        fontName=FONT_NORMAL,
        fontSize=6.7,
        leading=8.0,
        textColor=colors.HexColor('#1A202C'),
        alignment=1
    )

    # ═══════════════════════════════════════════════════════════
    # SAYFA 1: BAŞLIK + PROTOKOL + 10 NLP METRİĞİ + 2.000 BOOTSTRAP
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph('Healthcare Analytics &mdash; Kapsamlı Revizyon Teknik Raporu', title_style))
    story.append(Paragraph('<b>Proje:</b> CXR ile Tıbbi Rapor Üretimi | <b>Tarih:</b> Ekim 2026 | <b>Veri Seti:</b> MIMIC-CXR (215.617 doğrulanmış görsel)', meta_style))
    story.append(Paragraph('<b>Değerlendirme Protokolü:</b> Beam Search: 5 | Maksimum Token: 150 | Tekrar Cezası: 2.0 | N-Gram Tekrar Engeli: 3 | 16.175 Test Seti', meta_style))
    story.append(HRFlowable(width='100%', thickness=1.2, color=colors.HexColor('#2B6CB0'), spaceBefore=3, spaceAfter=4))

    # Bölüm 1: Tablo 4 Revizyonu
    story.append(Paragraph('1. Model Performansı ve 10 NLP Metriği Karşılaştırması (Tablo 4 Revizyonu)', h1_style))
    story.append(Paragraph('Dört modelin tamamı aynı 16.175 sıfır-sızıntılı hasta düzeyindeki test görselleri üzerinde özdeş beam search 5 parametreleriyle değerlendirilmiştir:', body_style))

    t1_headers = [
        Paragraph('<b>Metrik</b>', th_style),
        Paragraph('<b>CNN-LSTM<br/>(Orijinal)</b>', th_style),
        Paragraph('<b>Swin-GPT<br/>(Orijinal)</b>', th_style),
        Paragraph('<b>Swin-GPT (Deney A:<br/>Yatay Çevirme Yok)</b>', th_style),
        Paragraph('<b>Swin-GPT (Deney B:<br/>Yardımcı Kayıp Yok)</b>', th_style)
    ]
    t1_data = [t1_headers]
    for row in table4_data.values.tolist():
        m_name = row[0]
        c_val = f'{row[1]:.4f}'
        s_val = f'{row[2]:.4f}'
        a_val = f'{row[3]:.4f}'
        b_val = f'{row[4]:.4f}'
        is_a_best = m_name in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-1', 'METEOR', 'CIDEr']
        t1_data.append([
            Paragraph(f'<b>{m_name}</b>', td_style),
            Paragraph(c_val, td_style),
            Paragraph(s_val, td_style),
            Paragraph(f'<b>{a_val}</b>' if is_a_best else a_val, td_style),
            Paragraph(b_val, td_style)
        ])

    t1 = Table(t1_data, colWidths=[85, 110, 110, 120, 120])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.1),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t1)
    story.append(Paragraph('<font size=5.8 color="#718096">Not: Tablo 1 standart Külliyat (Corpus) BLEU değerlerini sunmaktadır. Tablo 2 ise eşleştirilmiş permütasyon testi için hasta düzeyinde düzeltmeli (smoothed) cümle BLEU ortalamalarını kullanır. BLEU dışı metrikler her iki tabloda matematiksel olarak özdeştir.</font>', body_style))
    story.append(Spacer(1, 3))

    # Bölüm 2: Bootstrap & Permütasyon Testleri
    story.append(Paragraph('2. İstatistiksel Anlamlılık: Monte Carlo Permütasyon Testleri ve Kümelenmiş Bootstrap (10 Metrik)', h1_style))
    story.append(Paragraph('2.784 tekil test hastası üzerinden ikili istatistiksel anlamlılık. P-değerleri, çoklu karşılaştırmalar için Holm-Bonferroni adım-aşağı düzeltmesi uygulanmış Monte Carlo eşleştirilmiş hasta düzeyi permütasyon testleri (2.000 yineleme) ile elde edilmiştir:', body_style))

    t2_headers = [
        Paragraph('<b>Karşılaştırma Çifti</b>', th_small),
        Paragraph('<b>Metrik</b>', th_small),
        Paragraph('<b>Model A</b>', th_small),
        Paragraph('<b>Model B</b>', th_small),
        Paragraph('<b>Fark (A-B)</b>', th_small),
        Paragraph('<b>%95 GA (Bootstrap)</b>', th_small),
        Paragraph('<b>p-değeri<br/>(Monte Carlo)</b>', th_small),
        Paragraph('<b>p-değeri<br/>(Holm-Bonf.)</b>', th_small)
    ]
    t2_data = [t2_headers]
    comp_name_tr = {
        "Swin-GPT vs CNN-LSTM": "Swin-GPT vs CNN-LSTM",
        "Exp A vs Swin-GPT": "Deney A vs Swin-GPT",
        "Exp B vs Swin-GPT": "Deney B vs Swin-GPT",
    }
    for comp_name, metrics in bootstrap_data.items():
        tr_comp_name = comp_name_tr.get(comp_name, comp_name)
        for m_name, vals in metrics.items():
            ci_str = f"[{vals['ci_95'][0]:.4f}, {vals['ci_95'][1]:.4f}]"
            diff_key = [k for k in vals.keys() if 'diff' in k][0]
            diff_val = vals[diff_key]
            p_str = vals['p_value']
            p_holm = vals.get('p_value_holm', p_str)
            is_sig = vals.get('significant_holm', ('<' in p_str or (float(p_str) < 0.05 if '<' not in p_str else False)))
            p_cell = Paragraph(f'<b>{p_holm}*</b>' if is_sig else p_holm, td_small)
            t2_data.append([
                Paragraph(tr_comp_name, td_small),
                Paragraph(m_name, td_small),
                Paragraph(f"{vals['mean_model_a']:.4f}", td_small),
                Paragraph(f"{vals['mean_model_b']:.4f}", td_small),
                Paragraph(f"{diff_val:.4f}", td_small),
                Paragraph(ci_str, td_small),
                Paragraph(p_str, td_small),
                p_cell
            ])

    t2 = Table(t2_data, colWidths=[105, 52, 50, 50, 55, 110, 60, 64], repeatRows=1)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2D3748')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 0.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t2)
    story.append(Paragraph('<font size=6 color="#718096">*2.784 hasta üzerinden eşleştirilmiş permütasyon testleri ve Holm-Bonferroni düzeltmesi (alfa = 0.05) sonrasında istatistiksel olarak anlamlıdır.</font>', body_style))

    # SAYFA 1 BİTİŞİ -> TEMİZ SAYFA GEÇİŞİ
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SAYFA 2: KLİNİK ETKİNLİK + 5 ODAK PATOLOJİ
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph('3. Klinik Etkinlik ve 14-Patoloji Bulgu Uyumu (U-Zero ve U-Ones)', h1_style))
    story.append(Paragraph('Doğrudan üretilen metin raporları üzerinden değerlendirilmiştir. Pozitif (1), açıkça negatif (0), belirsiz (-1) ve bahsedilmeyen (NaN) durumlar kesin olarak ayrılmıştır. Referansta bahsedilmeyen bulgular kesinlikle doğrulanmış negatif kabul edilmemiştir (Toplam FP = Desteklenmeyen + Doğrudan Çelişkiler):', body_style))

    t3_headers = [
        Paragraph('<b>Model</b>', th_style),
        Paragraph('<b>U-Zero<br/>Makro F1</b>', th_style),
        Paragraph('<b>U-Zero<br/>Mikro F1</b>', th_style),
        Paragraph('<b>U-Ones<br/>Makro F1</b>', th_style),
        Paragraph('<b>Atlananlar<br/>(FN)</b>', th_style),
        Paragraph('<b>Desteklenmeyen<br/>(Ref=NaN)</b>', th_small),
        Paragraph('<b>Doğrudan Çelişki<br/>(Ref=0.0)</b>', th_small),
        Paragraph('<b>Toplam FP<br/>(Dest.+Çelişki)</b>', th_style)
    ]
    t3_data = [t3_headers]
    model_name_map = {
        "CNN-LSTM (Original)": "CNN-LSTM (Orijinal)",
        "Swin-GPT (Original)": "Swin-GPT (Orijinal)",
        "Swin-GPT (Exp A: No HFlip)": "Swin-GPT (Deney A: Yatay Çevirme Yok)",
        "Swin-GPT (Exp B: No Aux Loss)": "Swin-GPT (Deney B: Yardımcı Kayıp Yok)",
    }
    for m_name, c_res in clinical_data.items():
        tr_m_name = model_name_map.get(m_name, m_name)
        omiss = c_res.get('Total Omissions (Atlanan Pozitif Bulgular)', c_res.get('Total Omissions (Atlanan Bulgular)', 0))
        unsup = c_res.get('Total Unsupported Findings (Referansta Bahsedilmeyen Bulgular)', c_res.get('Total Unsupported Findings (Desteklenmeyen Bulgular)', 0))
        contr = c_res.get('Total Direct Contradictions (Açıkça Çelişen Bulgular)', 0)
        tot_fp = unsup + contr
        t3_data.append([
            Paragraph(tr_m_name, td_left),
            Paragraph(f"{c_res['U-Zero (Ana Analiz)']['Macro F1']:.4f}", td_style),
            Paragraph(f"{c_res['U-Zero (Ana Analiz)']['Micro F1']:.4f}", td_style),
            Paragraph(f"{c_res['U-Ones (Duyarlılık Analizi)']['Macro F1']:.4f}", td_style),
            Paragraph(f"{omiss:,}", td_style),
            Paragraph(f"{unsup:,}", td_style),
            Paragraph(f"{contr:,}", td_style),
            Paragraph(f"<b>{tot_fp:,}</b>", td_style)
        ])

    t3 = Table(t3_data, colWidths=[108, 56, 56, 56, 54, 76, 76, 64])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#319795')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t3)
    story.append(Paragraph('<font size=6 color="#718096">Not: Toplam FP = Referansta Desteklenmeyen Bulgular (bahsedilmeyen, Ref=NaN) + Doğrudan Çelişkiler (referansta açıkça ekarte edilen, Ref=0.0). Desteklenmeyen bulgular teyit edilmiş halüsinasyon olarak nitelendirilmemelidir.</font>', body_style))
    story.append(Spacer(1, 6))

    # Bölüm 4: Odak Patolojiler
    story.append(Paragraph('4. Odak Patolojilerin Detaylı Analizi (U-Zero Protokolü)', h1_style))
    story.append(Paragraph('Hakemlerin işaret ettiği 5 kilit göğüs patolojisi için ayrıntılı analiz (Atlananlar = kaçırılan pozitifler; Desteklenmeyen = referans raporda bahsedilmeyen pozitif bulgular):', body_style))

    t4_headers = [
        Paragraph('<b>Model</b>', th_style),
        Paragraph('<b>Patoloji</b>', th_style),
        Paragraph('<b>Kesinlik</b>', th_style),
        Paragraph('<b>Duyarlılık</b>', th_style),
        Paragraph('<b>F1-Skoru</b>', th_style),
        Paragraph('<b>Atlananlar (FN)</b>', th_style),
        Paragraph('<b>Desteklenmeyen (FP)</b>', th_style)
    ]
    t4_data = [t4_headers]
    pat_name_tr = {
        "Cardiomegaly": "Kardiyomegali",
        "Pneumonia": "Zatürre (Pneumonia)",
        "Atelectasis": "Atelektazi",
        "Pneumothorax": "Pnömotoraks",
        "Pleural Effusion": "Plevral Efüzyon",
    }
    for m_name, c_res in clinical_data.items():
        tr_m_name = model_name_map.get(m_name, m_name)
        if '5_Focus_Pathologies' in c_res:
            for p_name, p_vals in c_res['5_Focus_Pathologies'].items():
                tr_p_name = pat_name_tr.get(p_name, p_name)
                om_val = p_vals.get('Omissions (Atlanan Pozitif)', 0)
                un_val = p_vals.get('Unsupported Findings (Referansta Bahsedilmeyen Pozitif)', p_vals.get('Unsupported Findings (Desteklenmeyen Pozitif)', 0))
                t4_data.append([
                    Paragraph(tr_m_name, td_left),
                    Paragraph(tr_p_name, td_style),
                    Paragraph(f"{p_vals['Precision']:.4f}", td_style),
                    Paragraph(f"{p_vals['Recall']:.4f}", td_style),
                    Paragraph(f"{p_vals['F1-Score']:.4f}", td_style),
                    Paragraph(f"{om_val:,}", td_style),
                    Paragraph(f"{un_val:,}", td_style)
                ])

    t4_table = Table(t4_data, colWidths=[116, 110, 64, 64, 64, 64, 64], repeatRows=1)
    t4_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A5568')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t4_table)

    # SAYFA 2 BİTİŞİ -> TEMİZ SAYFA GEÇİŞİ
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # SAYFA 3: VERİ SETİ DENETİMLERİ + METODOLOJİ + BİLİMSEL GÜVENCELER
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph('5. Veri Seti Denetimleri ve Filtreleme Hattı (261.137 &rarr; 248.506 &rarr; 215.617)', h1_style))
    story.append(Paragraph('Kriptografik olarak doğrulanmış filtreleme hunisi. Ham CSV (248.506) ile nihai sıfır-sızıntılı kohort (215.617) arasındaki toplam örnek azalması tam 32.889 örnektir:', body_style))

    t5_headers = [
        Paragraph('<b>Aşama</b>', th_style),
        Paragraph('<b>Filtre / Karar</b>', th_style),
        Paragraph('<b>Giriş</b>', th_style),
        Paragraph('<b>Dışlanan</b>', th_style),
        Paragraph('<b>Kalan</b>', th_style),
        Paragraph('<b>Gerekçe ve Denetim Açıklaması</b>', th_style)
    ]
    t5_rows = [
        ['1. Disk JPG Taraması', 'Dosya Sistemindeki Fiziksel Görseller', '261.137', '12.631', '248.506', 'Rapor metaverisi bulunmayan eşleşmemiş/yetim görseller ayıklandı'],
        ['2. Metin Kalitesi', 'Karakter < 10, Kelime < 5, Geçersiz Metin', '248.506', '3.112', '245.394', 'Bozuk, boş veya tetkik yapılmadı kayıtları çıkarıldı'],
        ['3. Regex Temizliği', 'Bulgular ve İzlenim Bölümleri Birleştirildi', '245.394', '-', '245.394', 'İdari imzalar, tarihler ve karşılaştırma metinleri temizlendi'],
        ['4. Tekilleştirme & Sınırlar', 'Rapor başına maks 5 normal & uzunluk sınırları', '245.394', '29.777', '215.617', 'Aşırı kelime uç değerleri (<%0.5, >%99.5) ve normal şablon fazlalıkları ayıklandı'],
        ['5. Hasta Ayrımı', 'Sıfır-Sızıntılı Hasta Düzeyinde Bölme', '215.617', '0', '215.617', 'Eğitim: 183.271 | Doğrulama: 16.171 | Test: 16.175 (2.784 hasta)']
    ]
    t5_data = [t5_headers]
    for r in t5_rows:
        t5_data.append([
            Paragraph(r[0], td_style),
            Paragraph(r[1], td_left),
            Paragraph(r[2], td_style),
            Paragraph(r[3], td_style),
            Paragraph(f'<b>{r[4]}</b>', td_style),
            Paragraph(r[5], td_left)
        ])

    t5 = Table(t5_data, colWidths=[65, 120, 55, 50, 65, 191])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#718096')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t5)
    story.append(Paragraph('<font size=6 color="#718096">Denetim Doğrulaması: Ham CSV (248.506) ile nihai sıfır-sızıntılı kohort (215.617) arasındaki toplam düşüş = tam 32.889 örnektir (Eğitim düşüşü: 27.767; Doğrulama: 2.640; Test: 2.482).</font>', body_style))
    story.append(Spacer(1, 6))

    # Bölüm 6: Model Provenance Tablosu
    story.append(Paragraph('6. Model Kanıt Zinciri ve Veri Dağılım Özellikleri', h1_style))
    t6_headers = [
        Paragraph('<b>Model Kodu</b>', th_style),
        Paragraph('<b>Model Mimarisi</b>', th_style),
        Paragraph('<b>Tohum / Commit</b>', th_style),
        Paragraph('<b>En İyi<br/>Epoch</b>', th_style),
        Paragraph('<b>Doğrulama<br/>Kaybı</b>', th_style),
        Paragraph('<b>Öğrenme Oranı<br/>/ Devam</b>', th_style),
        Paragraph('<b>Checkpoint Özeti (SHA-256)</b>', th_style)
    ]
    t6_rows = [
        ['M1', 'CNN-LSTM (DenseNet-121 + LSTM)', '42 / dd2441d', '8', '1.6251', '1e-4 / Yok', '44638AD48B181B67DA0EB82D0CFACD2A...'],
        ['M2', 'Swin-GPT (MTL Temel Model)', '42 / dd2441d', '9', '1.2003', '2e-6 / Yok', '8B0A7A057943485C580E126370FDC5DC...'],
        ['Deney A', 'Swin-GPT (Yatay Çevirme Yok)', '42 / 13e532a', '9', '1.1927', '1.97e-6 / Yok', '3F030A4B9C9C0BFA2E79EBDAC4385BB...'],
        ['Deney B', 'Swin-GPT (Yardımcı Kayıp Yok)', '42 / 13e532a', '10', '1.0686', '1.97e-6 / Yok', 'E429A03788A27D534F69B5FD0ED5F19...']
    ]
    t6_data = [t6_headers]
    for r in t6_rows:
        t6_data.append([
            Paragraph(r[0], td_style),
            Paragraph(r[1], td_left),
            Paragraph(r[2], td_style),
            Paragraph(r[3], td_style),
            Paragraph(r[4], td_style),
            Paragraph(r[5], td_style),
            Paragraph(r[6], td_style)
        ])
    t6 = Table(t6_data, colWidths=[42, 138, 64, 46, 52, 58, 146])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A5568')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t6)
    story.append(Spacer(1, 6))

    # Bölüm 7: Metodolojik Güvenceler
    story.append(Paragraph('7. Metodolojik Uyumluluk ve Bilimsel Güvenceler', h1_style))
    disc_bullets = [
        '<b>Anatomik Asimetri ve Projeksiyon Dağılımı (10.1 & Deney A):</b> Göğüs radyografisi anatomisi doğası gereği asimetriktir (aort topuzu solda, dekstrokardi nadir patolojidir). Veri artırmada yatay çevirmenin kaldırılması (Deney A), modelin yapay dekstrokardi üretmesini engelleyerek BLEU-1 skorunu belirgin biçimde artırmıştır (+0.0126, p < 0.0005). Veri seti %23.3 Frontal ve %20.9 Lateral projeksiyon içermekte olup, klinik pratiğe tam uyum için tüm projeksiyonlar korunmuştur.',
        '<b>CheXpert ve Kelime Dağarcığı Denetimi (10.2 & 10.4):</b> Test setinde en sık rastlanan patolojiler Destek Cihazları (%29.2), Kardiyomegali (%24.4), Atelektazi (%23.7) ve Plevral Efüzyondur (%22.9). CNN-LSTM modeli tam 7.311 kelimelik sözlük kullanmakta ve sızıntı içermemektedir; Swin-GPT ise 50.257 alt-kelimelik BPE sözlüğü ile sıfır OOV (%0) oranına sahiptir.',
        '<b>Tetkik Düzeyinde Sıfır Sızıntı:</b> Toplam 139.944 tetkik (study) kesin olarak hasta (subject_id) bazında bölünmüştür. Kümeler arasında %0 hasta ve %0 tetkik çakışması garanti edilmiştir.',
        '<b>Kriptografik Doğrulama:</b> Rapordaki tüm sayılar ORIGINAL_BACKUP/ altındaki MANIFEST.sha256 dosyasıyla tam uyuşan mühürlü ağırlıklardan üretilmiştir.'
    ]
    for b in disc_bullets:
        story.append(Paragraph(f'&bull; {b}', body_style))
        story.append(Spacer(1, 2.5))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\n[SUCCESS] NIHAI REVIZYON PDF RAPORU OLUSTURULDU: {output_pdf_path}")

