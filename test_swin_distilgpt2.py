"""
test_swin_distilgpt2.py
=======================
Swin-B + DistilGPT-2 MTL modelini test setinde değerlendirir.

Metrikler:
  NLP   : BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore, SBERT
  Klinik: 14 patoloji için F1, Precision, Recall (BCE sınıflandırma kafası)
"""

# ── HF Offline Koruması ───────────────────────────────────────────────────
import os as _os
def _hf_cache_ok(ids):
    root = _os.environ.get("HF_HOME", _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    return all(_os.path.isdir(_os.path.join(root, "models--" + m.replace("/", "--"))) for m in ids)
if _hf_cache_ok(["microsoft/swin-base-patch4-window7-224", "distilgpt2", "sentence-transformers/all-MiniLM-L6-v2"]):
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# ─────────────────────────────────────────────────────────────────────────

import os
import sys
import gc
import json

# Support emoji/Unicode prints in Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

from sklearn.metrics import classification_report

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None

from config import Config
from src.data_loader.dataset_swin import get_swin_dataloaders
from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL

nltk.download("wordnet", quiet=True)
nltk.download("punkt",   quiet=True)
nltk.download("omw-1.4", quiet=True)


# =========================================================================
# SABİTLER
# =========================================================================
PATHOLOGY_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]

SWIN_MODEL_NAME    = "microsoft/swin-base-patch4-window7-224"
GPT_MODEL_NAME     = "distilgpt2"
CHECKPOINT_DIRNAME = "checkpoints_swin_distilgpt2"


# =========================================================================
# 1. INFERENCE
# =========================================================================
def run_inference(
    model: SwinDistilGPT2ForMTL,
    loader,
    tokenizer,
    device: torch.device,
    max_new_tokens:       int   = 120,
    num_beams:            int   = 4,
    repetition_penalty:   float = 2.0,
    no_repeat_ngram_size: int   = 3,
    temperature:          float = 1.0,
    top_p:                float = 1.0,
    do_sample:            bool  = False,
    early_stopping:       bool  = True,
):
    model.eval()

    predictions    = []
    references     = []
    all_logits_cls = []
    all_labels_cls = []

    with torch.no_grad():
        loop = tqdm(loader, desc="🔮 Inference", ncols=120)
        for batch in loop:
            images         = batch["images"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_cls     = batch.get("labels_cls")

            # Sınıflandırma logitleri
            outputs    = model(
                pixel_values=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits_cls = outputs["logits_cls"].cpu().numpy()
            all_logits_cls.extend(logits_cls)

            if labels_cls is not None:
                all_labels_cls.extend(labels_cls.cpu().numpy())

            # Metin üretimi
            gen_ids = model.generate(
                pixel_values=images,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                early_stopping=early_stopping,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            predictions.extend(pred_texts)

            if "raw_texts" in batch:
                references.extend(batch["raw_texts"])
            else:
                for ids in input_ids:
                    ref = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    references.append(ref)

            del images, input_ids, attention_mask, outputs, gen_ids
            torch.cuda.empty_cache()

    return predictions, references, np.array(all_logits_cls), np.array(all_labels_cls)


# =========================================================================
# 2. METRİKLER
# =========================================================================

def compute_clinical_efficacy(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    if len(labels) == 0:
        return {}
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return classification_report(
        labels, preds,
        target_names=PATHOLOGY_COLS,
        output_dict=True,
        zero_division=0,
    )


def compute_standard_nlp(predictions: list, references: list) -> dict:
    smooth    = SmoothingFunction().method1
    refs_tok  = [[r.split()] for r in references]
    hyps_tok  = [p.split()   for p in predictions]

    b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    b4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    rl_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl_list   = [rl_scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]

    met = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]

    cider_scorer = Cider()
    gts = {i: [references[i]]  for i in range(len(references))}
    res = {i: [predictions[i]] for i in range(len(predictions))}
    cid, _ = cider_scorer.compute_score(gts, res)

    return {
        "BLEU-1":  round(b1,           4),
        "BLEU-4":  round(b4,           4),
        "ROUGE-L": round(np.mean(rl_list), 4),
        "METEOR":  round(np.mean(met),     4),
        "CIDEr":   round(cid,           4),
    }


def compute_bert_score(predictions: list, references: list) -> dict:
    if bert_score_fn is None:
        print("⚠️  bert-score yüklü değil (pip install bert-score)")
        return {}
    print("  ▶ BERTScore hesaplanıyor...")
    P, R, F1 = bert_score_fn(predictions, references, lang="en", rescale_with_baseline=True)
    return {
        "BERTScore-P":  round(P.mean().item(),  4),
        "BERTScore-R":  round(R.mean().item(),  4),
        "BERTScore-F1": round(F1.mean().item(), 4),
    }


def compute_sbert_similarity(predictions: list, references: list) -> dict:
    if SentenceTransformer is None:
        print("⚠️  sentence-transformers yüklü değil")
        return {}
    print("  ▶ SBERT benzerliği hesaplanıyor...")
    sbert  = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    scores = []
    bs     = 64
    for i in tqdm(range(0, len(predictions), bs), desc="SBERT", ncols=100):
        p_b = predictions[i:i+bs]
        r_b = references[i:i+bs]
        e1  = sbert.encode(p_b, convert_to_tensor=True)
        e2  = sbert.encode(r_b, convert_to_tensor=True)
        cos = st_util.cos_sim(e1, e2)
        for j in range(len(p_b)):
            scores.append(cos[j][j].item())
    del sbert
    torch.cuda.empty_cache()
    return {"SBERT-Cosine-Sim": round(float(np.mean(scores)), 4)}


# =========================================================================
# 3. YAZDIRMA / KAYIT
# =========================================================================

def print_results(nlp_metrics: dict, ce_report: dict):
    print(f"\n{'='*70}")
    print("📊 Swin-B + DistilGPT-2 (MTL) TEST SONUÇLARI")
    print(f"{'='*70}")

    print("\n[ NLP & SEMANTIC METRİKLER ]")
    for k, v in nlp_metrics.items():
        bar = "█" * int(v * 40) + "░" * (40 - int(v * 40))
        print(f"   {k:<20} {v:.4f}  [{bar}]")

    print("\n[ KLİNİK ETKİNLİK (14 Patoloji MTL Kafası) ]")
    if "macro avg" in ce_report:
        mac = ce_report["macro avg"]
        print(f"   Macro Precision : {mac['precision']:.4f}")
        print(f"   Macro Recall    : {mac['recall']:.4f}")
        print(f"   Macro F1-Score  : {mac['f1-score']:.4f}")

    print(f"\n{'='*70}\n")


def save_results(nlp_metrics: dict, ce_report: dict, predictions: list, references: list, save_dir: str):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = save_path / f"swin_distilgpt2_metrics_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"NLP_Metrics": nlp_metrics, "Clinical_Efficacy": ce_report}, f, indent=2, ensure_ascii=False)

    txt_path = save_path / f"swin_distilgpt2_summary_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Swin-B + DistilGPT-2 (MTL) TEST ÖZETİ — {ts}\n")
        f.write("=" * 80 + "\n\n")
        f.write("--- NLP METRİKLERİ ---\n")
        for k, v in nlp_metrics.items():
            f.write(f"{k}: {v}\n")
        if "macro avg" in ce_report:
            mac = ce_report["macro avg"]
            f.write("\n--- KLİNİK ETKİNLİK (MACRO) ---\n")
            f.write(f"Precision : {mac['precision']:.4f}\n")
            f.write(f"Recall    : {mac['recall']:.4f}\n")
            f.write(f"F1-Score  : {mac['f1-score']:.4f}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("--- 14 HASTALIK DETAYI ---\n")
        for cls_name in PATHOLOGY_COLS:
            if cls_name in ce_report:
                cr = ce_report[cls_name]
                f.write(f"\n{cls_name}:\n")
                f.write(f"  P : {cr['precision']:.4f}\n")
                f.write(f"  R : {cr['recall']:.4f}\n")
                f.write(f"  F1: {cr['f1-score']:.4f}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write("--- ÖRNEK TAHMİNLER (ilk 30) ---\n")
        for i in range(min(30, len(predictions))):
            f.write(f"[{i+1}]\nGERÇEK : {references[i]}\nTAHMİN : {predictions[i]}\n{'-'*60}\n")

    print(f"💾 Sonuçlar kaydedildi:\n   ➡ {json_path}\n   ➡ {txt_path}")


# =========================================================================
# ANA FONKSİYON
# =========================================================================
def main():
    Config.vit_setup()
    device = Config.DEVICE

    SHUTDOWN_AFTER_TEST = True

    checkpoint_dir = Config.SWIN_CHECKPOINT_DIR
    model_path     = os.path.join(checkpoint_dir, "best_model_swin_distilgpt2.pth")

    if not os.path.exists(model_path):
        # SWA modeli de kabul et
        swa_path = os.path.join(checkpoint_dir, "swa_model_final.pth")
        if os.path.exists(swa_path):
            model_path = swa_path
            print(f"ℹ️  best_model bulunamadı; SWA modeli kullanılıyor: {swa_path}")
        else:
            print(f"❌ Model bulunamadı: {model_path}")
            return

    print(f"\n{'='*70}")
    print("🧪 Swin-B + DistilGPT-2 (MTL) TEST BAŞLIYOR")
    print(f"{'='*70}")
    print(f"   Model: {model_path}")

    batch_size  = getattr(Config, "VIT_BATCH_SIZE",  4)
    max_length  = getattr(Config, "VIT_MAX_LENGTH",  150)
    num_workers = getattr(Config, "NUM_WORKERS",     0)

    # ── Veri ──────────────────────────────────────────────────────────────
    print("\n📥 Test verisi yükleniyor...")
    _, _, test_loader, tokenizer = get_swin_dataloaders(
        train_csv=Config.TRAIN_PROCESSED_CSV,
        val_csv=Config.VAL_PROCESSED_CSV,
        image_dir=Config.IMAGE_DIR,
        test_csv=Config.TEST_PROCESSED_CSV,
        batch_size=batch_size,
        max_length=max_length,
        image_size=224,
        num_workers=num_workers,
        tokenizer_name=GPT_MODEL_NAME,
    )    # ── Backup Check ──────────────────────────────────────────────────────
    import pickle
    backup_path = Path(Config.LOG_DIR) / "temp_test_predictions.pkl"
    loaded_backup = False
    
    if backup_path.exists():
        print(f"\n💡 Bulunan yedek tahmin dosyası: {backup_path}")
        print("   Önceki inference sonuçları yükleniyor, 38 dakikalık görsel tahmin adımı atlanıyor!")
        try:
            with open(backup_path, "rb") as f:
                backup_data = pickle.load(f)
            preds = backup_data["preds"]
            refs = backup_data["refs"]
            logits = backup_data["logits"]
            labels = backup_data["labels"]
            loaded_backup = True
            print("   ✅ Tahminler başarıyla yüklendi!")
        except Exception as e:
            print(f"   ⚠️ Yedek yüklenirken hata oluştu (silinip baştan başlanıyor): {e}")

    if not loaded_backup:
        # ── Model ─────────────────────────────────────────────────────────────
        print("\n📥 Model ağırlıkları yükleniyor...")
        model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
            encoder_name=SWIN_MODEL_NAME,
            decoder_name=GPT_MODEL_NAME,
            enable_gradient_checkpointing=False,
        ).to(device)

        ckpt  = torch.load(model_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print("✅ Ağırlıklar yüklendi.")

        # ── Inference ─────────────────────────────────────────────────────────
        beam_size    = getattr(Config, "VIT_NUM_BEAMS",   4)
        max_new_toks = getattr(Config, "VIT_MAX_LENGTH",  150)

        preds, refs, logits, labels = run_inference(
            model, test_loader, tokenizer, device,
            max_new_tokens=max_new_toks,
            num_beams=beam_size,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            early_stopping=True,
        )

        preds = [p.strip() or "empty" for p in preds]
        refs  = [r.strip() or "empty" for r in refs]

        print("\n🧹 Model ve tokenizer VRAM'den temizleniyor...")
        del model, tokenizer, test_loader
        gc.collect()
        torch.cuda.empty_cache()

        # Save predictions immediately to backup
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_path, "wb") as f:
                pickle.dump({
                    "preds": preds,
                    "refs": refs,
                    "logits": logits,
                    "labels": labels
                }, f)
            print(f"💾 Tahminler yedek dosyasına kaydedildi: {backup_path}")
        except Exception as e:
            print(f"⚠️ Tahminler yedeklenirken hata oluştu: {e}")

    # ── Metrikler ─────────────────────────────────────────────────────────
    print("\n📐 Metrikler hesaplanıyor...")

    ce_report   = compute_clinical_efficacy(logits, labels)
    nlp_metrics = {}

    print("  ▶ Standart NLP (BLEU, ROUGE, METEOR, CIDEr)...")
    nlp_metrics.update(compute_standard_nlp(preds, refs))
    
    # Temporarily restore online mode for metric models to download from HuggingFace
    for env_var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        if env_var in os.environ:
            del os.environ[env_var]

    print("  ▶ BERTScore hesaplanıyor...")
    try:
        nlp_metrics.update(compute_bert_score(preds, refs))
        print("  ✅ BERTScore başarıyla hesaplandı.")
    except Exception as e:
        print(f"  ⚠️ BERTScore hesaplanırken hata oluştu: {e}")
        nlp_metrics.update({
            "BERTScore-P": 0.0,
            "BERTScore-R": 0.0,
            "BERTScore-F1": 0.0
        })

    print("  ▶ SBERT benzerliği hesaplanıyor...")
    try:
        nlp_metrics.update(compute_sbert_similarity(preds, refs))
        print("  ✅ SBERT benzerliği başarıyla hesaplandı.")
    except Exception as e:
        print(f"  ⚠️ SBERT benzerliği hesaplanırken hata oluştu: {e}")
        nlp_metrics.update({
            "SBERT-Cosine-Sim": 0.0
        })

    print_results(nlp_metrics, ce_report)
    save_results(nlp_metrics, ce_report, preds, refs, Config.LOG_DIR)

    # Clean up the backup file upon successful completion
    if backup_path.exists():
        try:
            os.remove(backup_path)
            print("🧹 Geçici yedek tahmin dosyası silindi.")
        except Exception:
            pass

    if SHUTDOWN_AFTER_TEST:
        os.system("shutdown /s /t 15")


if __name__ == "__main__":
    main()
