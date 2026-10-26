"""
run_full_revision_analysis.py
=============================
4 modelin tüm tahminlerini birleştirir, 10 NLP metriğini hesaplar,
2.000 tekrarlı hasta bazlı bootstrap ve eşleştirilmiş permütasyon testlerini icra eder,
U-Zero ve U-Ones klinik doğruluk analizlerini yapar ve nihai PDF raporunu oluşturur.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Windows konsolunda UTF-8 çıktı desteği sağla
if sys.platform.startswith("win"):
    import io
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from advanced_evaluation_suite import (
    compute_all_10_nlp_metrics,
    run_patient_clustered_bootstrap_and_permutation,
    evaluate_clinical_efficacy_advanced,
    generate_master_revision_pdf,
    PATHOLOGY_COLS,
    FOCUS_PATHOLOGIES
)

def main():
    print("=" * 80)
    print("🚀 NİHAİ REVİZYON ANALİZ VE RAPORLAMA PİPELİNE'I BAŞLIYOR")
    print("=" * 80)

    # 1. DOSYALARI YÜKLE VE DOĞRULA
    paths = {
        "CNN-LSTM (Original)": "experiments/baseline_cnnlstm/predictions/test_predictions_cnnlstm.csv",
        "Swin-GPT (MTL Original)": "experiments/baseline_swin_gpt/predictions/test_predictions_baseline_swin.csv",
        "Swin-GPT (No HFlip / Deney A)": "experiments/exp_A_no_hflip/predictions/test_predictions_exp_A.csv",
        "Swin-GPT (No Aux Loss / Deney B)": "experiments/exp_B_no_aux_loss/predictions/test_predictions_exp_B.csv",
    }

    dfs = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Eksik dosya: {p}")
        df = pd.read_csv(p)
        assert len(df) == 16175, f"{name} satır sayısı 16,175 olmalıdır, bulunan: {len(df)}"
        dfs[name] = df
        print(f"✅ Yüklendi: {name:<32} ({len(df):,} örnek)")

    # Master DataFrame
    master_df = pd.DataFrame({
        "image_id": dfs["CNN-LSTM (Original)"]["image_id"],
        "subject_id": dfs["CNN-LSTM (Original)"]["subject_id"],
        "study_id": dfs["CNN-LSTM (Original)"]["study_id"],
        "reference_report": dfs["CNN-LSTM (Original)"]["reference_report"],
        "pred_cnnlstm": dfs["CNN-LSTM (Original)"]["generated_report"],
        "pred_swin_original": dfs["Swin-GPT (MTL Original)"]["generated_report"],
        "pred_exp_a": dfs["Swin-GPT (No HFlip / Deney A)"]["generated_report"],
        "pred_exp_b": dfs["Swin-GPT (No Aux Loss / Deney B)"]["generated_report"],
    })

    master_csv_path = "experiments/master_test_predictions_all_models.csv"
    os.makedirs("experiments", exist_ok=True)
    master_df.to_csv(master_csv_path, index=False, encoding="utf-8")
    print(f"\n💾 4 modelin birleşik tahminleri kaydedildi: {master_csv_path}")

    # 2. STANDART 10 NLP METRİĞİ HESAPLAMA (TABLO 4)
    print("\n" + "=" * 80)
    print("📊 1. STANDART 10 NLP METRİĞİ HESAPLANIYOR (TABLO 4 REVİZYONU)")
    print("=" * 80)

    table4_csv = "experiments/table4_nlp_metrics.csv"
    cols_map = {
        "CNN-LSTM (Original)": "pred_cnnlstm",
        "Swin-GPT (Original)": "pred_swin_original",
        "Swin-GPT (Exp A: No HFlip)": "pred_exp_a",
        "Swin-GPT (Exp B: No Aux Loss)": "pred_exp_b",
    }
    metrics_list = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "CIDEr", "SBERT"]

    if os.path.exists(table4_csv):
        print(f"💡 Önceden hesaplanmış Tablo 4 bulundu, doğrudan yükleniyor: {table4_csv}")
        table4_df = pd.read_csv(table4_csv)
    else:
        from sentence_transformers import SentenceTransformer
        print("📦 SentenceTransformer (all-MiniLM-L6-v2) yükleniyor...")
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

        nlp_results = {}
        for model_name, col_name in cols_map.items():
            print(f"\n▶ Hesaplanıyor: {model_name}...")
            preds = master_df[col_name].fillna("empty").astype(str).tolist()
            refs  = master_df["reference_report"].fillna("empty").astype(str).tolist()
            res = compute_all_10_nlp_metrics(preds, refs, sbert_model=sbert_model)
            nlp_results[model_name] = res
            print(f"   Sonuçlar: {res}")

        table4_df = pd.DataFrame({"Metric": metrics_list})
        for model_name in cols_map.keys():
            table4_df[model_name] = [nlp_results[model_name][m] for m in metrics_list]
        table4_df.to_csv(table4_csv, index=False)

    print("\n" + "─" * 80)
    print("📋 TABLO 4: 10 NLP METRİĞİ ÖZET KARŞILAŞTIRMASI")
    print("─" * 80)
    print(table4_df.to_string(index=False))

    import argparse
    parser = argparse.ArgumentParser(description="Master Evaluation and Reporting Pipeline")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputing 2000 bootstraps & permutations")
    parser.add_argument("--skip_eval", action="store_true", help="Skip recomputing metrics if outputs exist")
    parser.add_argument("--show_stats", action="store_true", help="Print 10-metric Holm statistical significance tables")
    args = parser.parse_args()

    # 3. İSTATİSTİKSEL ANALİZ: 2.000 HASTA BAZLI BOOTSTRAP VE PERMÜTASYON TESTİ (10 METRİK)
    print("\n" + "=" * 80)
    print("📐 2. İSTATİSTİKSEL ANLAMLILIK: 2.000 BOOTSTRAP & EŞLEŞTİRİLMİŞ PERMÜTASYON TESTİ (10 METRİK)")
    print("=" * 80)

    comparisons = {
        "Exp A vs Swin-GPT": ("pred_exp_a", "pred_swin_original"),
        "Exp B vs Swin-GPT": ("pred_exp_b", "pred_swin_original"),
        "Swin-GPT vs CNN-LSTM": ("pred_swin_original", "pred_cnnlstm"),
    }

    final_output_json = "experiments/final_revision_results_all.json"
    bootstrap_results = {}

    if not args.force_recompute and os.path.exists(final_output_json):
        try:
            with open(final_output_json, "r", encoding="utf-8") as f_prev:
                prev_json_data = json.load(f_prev)
            if "bootstrap_permutation_stats" in prev_json_data:
                bootstrap_results = prev_json_data["bootstrap_permutation_stats"]
                print(f"💡 Önceden hesaplanmış 2.000 replikasyonlu istatistiksel sonuçlar yüklendi: {final_output_json}")
        except Exception as e:
            print(f"⚠️ Önceki sonuçlar yüklenemedi ({e}), yeniden hesaplanıyor...")

    if not bootstrap_results or args.force_recompute:
        for comp_name, (col_a, col_b) in comparisons.items():
            res_comp = run_patient_clustered_bootstrap_and_permutation(
                master_df,
                model_a_col=col_a,
                model_b_col=col_b,
                ref_col="reference_report",
                subject_col="subject_id",
                n_bootstraps=2000,
                n_permutations=2000,
                seed=42,
            )
            bootstrap_results[comp_name] = res_comp

    # 10 Metriğin Tamamını Holm-Bonferroni Düzeltmesiyle Konsola ve CSV'ye Aktar
    table5_rows = []
    for comp_name, metrics_dict in bootstrap_results.items():
        print("\n" + "─" * 80)
        print(f"📌 {comp_name.upper()} (10 METRİK İÇİN HASTA DÜZEYİNDE PERMÜTASYON & HOLM DÜZELTMESİ)")
        print("─" * 80)
        comp_rows = []
        for m_name, vals in metrics_dict.items():
            ci_str = f"[{vals['ci_95'][0]:.4f}, {vals['ci_95'][1]:.4f}]"
            sig_holm = "Evet" if vals.get("significant_holm", vals.get("statistically_significant", False)) else "Hayır"
            p_holm = vals.get("p_value_holm", vals.get("p_value", "N/A"))
            row_data = {
                "Karşılaştırma": comp_name,
                "Metrik": m_name,
                "Model A Ort": vals.get("mean_model_a", 0.0),
                "Model B Ort": vals.get("mean_model_b", 0.0),
                "Fark (A - B)": vals.get("mean_diff (A - B)", 0.0),
                "%95 GA": ci_str,
                "p (Perm)": vals.get("p_value", "N/A"),
                "p (Holm)": p_holm,
                "Anlamlı (α=0.05)": sig_holm
            }
            comp_rows.append(row_data)
            table5_rows.append(row_data)
        comp_df = pd.DataFrame(comp_rows)
        print(comp_df.to_string(index=False))

    table5_csv = "experiments/table5_statistical_significance_holm.csv"
    pd.DataFrame(table5_rows).to_csv(table5_csv, index=False, encoding="utf-8")
    print(f"\n💾 10 metriklik Holm düzeltmeli istatistik tablosu kaydedildi: {table5_csv}")

    # 4. KLİNİK DOĞRULUK DEĞERLENDİRMESİ (U-Zero, U-Ones, Omissions, Unsupported)
    print("\n" + "=" * 80)
    print("🩺 3. KLİNİK DOĞRULUK DEĞERLENDİRMESİ (U-ZERO & U-ONES & 14 PATOLOJİ)")
    print("=" * 80)

    clinical_results = {}
    if not args.force_recompute and os.path.exists(final_output_json):
        try:
            with open(final_output_json, "r", encoding="utf-8") as f_prev:
                prev_json_data = json.load(f_prev)
            if "clinical_efficacy" in prev_json_data:
                clinical_results = prev_json_data["clinical_efficacy"]
                print(f"💡 Önceden hesaplanmış klinik doğruluk sonuçları yüklendi: {final_output_json}")
        except Exception:
            pass

    if not clinical_results or args.force_recompute:
        for model_name, col_name in cols_map.items():
            c_res = evaluate_clinical_efficacy_advanced(master_df, pred_col=col_name, ref_col="reference_report")
            clinical_results[model_name] = c_res
            omiss = c_res.get("Total Omissions (Atlanan Pozitif Bulgular)", c_res.get("Total Omissions (Atlanan Bulgular)", 0))
            unsup = c_res.get("Total Unsupported Findings (Referansta Bahsedilmeyen Bulgular)", c_res.get("Total Unsupported Findings (Desteklenmeyen Bulgular)", 0))
            print(f"      {model_name:<32} -> Omissions (FN): {omiss:,} | Unsupported (FP): {unsup:,}")

    # 5. DENETİM BİLGİLERİ DERLEME
    audit_data = {
        "total_images": 215617,
        "test_images": 16175,
        "patients": 2784,
    }

    # 6. JSON DOSYASI OLARAK TÜM SONUÇLARI KAYDET
    full_data_to_save = {
        "timestamp": datetime.now().isoformat(),
        "table4_nlp_metrics": table4_df.to_dict(orient="records"),
        "bootstrap_permutation_stats": bootstrap_results,
        "clinical_efficacy": clinical_results,
        "audit_metadata": audit_data,
    }
    with open(final_output_json, "w", encoding="utf-8") as f:
        json.dump(full_data_to_save, f, indent=2)
    print(f"\n💾 Tüm sayısal analiz sonuçları kaydedildi: {final_output_json}")

    # 7. NİHAİ PDF RAPORU OLUŞTUR
    pdf_out_path = "Healthcare_Analytics_Revision_Master_Report.pdf"
    print("\n" + "=" * 80)
    print(f"📄 4. DANIŞMAN VE HAKEMLER İÇİN MASTER PDF RAPORU ÜRETİLİYOR: {pdf_out_path}")
    print("=" * 80)
    generate_master_revision_pdf(
        table4_data=table4_df,
        bootstrap_data=bootstrap_results,
        clinical_data=clinical_results,
        audit_data=audit_data,
        output_pdf_path=pdf_out_path,
    )
    import shutil
    shutil.copyfile(pdf_out_path, "Rapor.pdf")
    print(f"📋 'Rapor.pdf' dosyası da güncellendi: {os.path.abspath('Rapor.pdf')}")

    print("\n" + "=" * 80)
    print("🎉 TÜM ANALİZLER VE MASTER PDF RAPORU BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)

if __name__ == "__main__":
    main()
