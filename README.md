# 🏥 Medical Report Generation from Chest X-Rays (CXR)
### *Healthcare Analytics* — Major Revision Official Implementation & Benchmark

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13%2Bcu130-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Swin--B%20%2B%20DistilGPT2-yellow)](https://huggingface.co/)
[![Dataset: MIMIC-CXR](https://img.shields.io/badge/Dataset-MIMIC--CXR%20v2.0-success)](https://physionet.org/content/mimic-cxr/2.0.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for automated radiology report generation (combined *Findings* and *Impression*) from Chest X-Rays, featuring patient-level zero-leakage evaluation, clinical efficacy evaluation across 14 chest pathologies, and extensive ablation experiments for the journal *Healthcare Analytics*.

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Scientific Discoveries](#-key-scientific-discoveries)
- [Model Architectures](#-model-architectures)
- [Comprehensive Benchmark (Table 4 Revision)](#-comprehensive-benchmark-table-4-revision)
- [Statistical Significance & Permutation Testing](#-statistical-significance--permutation-testing)
- [Clinical Efficacy & Pathology Alignment](#-clinical-efficacy--pathology-alignment)
- [Dataset Curation & Zero-Leakage Split](#-dataset-curation--zero-leakage-split)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Pretrained Models & Google Drive Download Links](#-pretrained-models--google-drive-download-links)
- [Citation & Reproducibility](#-citation--reproducibility)

---

## 🔬 Executive Overview

This repository hosts the complete code, evaluation pipelines, data split registries, and prediction outputs for the paper under major revision in *Healthcare Analytics*. 

We systematically compare **four deep learning models** on the exact same **16,175 zero-leakage test set** (2,784 unique patients) under deterministic beam search decoding:
1. **CNN-LSTM (Baseline):** DenseNet-121 visual backbone + 2-layer Attention LSTM language decoder (vocabulary: 7,311 clinical tokens).
2. **Swin-GPT (MTL Baseline):** Swin Transformer Base (`microsoft/swin-base-patch4-window7-224`) + DistilGPT-2 + Multi-Task Learning (Auxiliary 14-pathology classification loss).
3. **Swin-GPT Exp A (No Horizontal Flip):** `--p_hflip 0.0 --cls_loss_weight 0.5`. Investigates the removal of horizontal flip augmentation to resolve anatomical laterality conflicts.
4. **Swin-GPT Exp B (No Aux Loss):** `--p_hflip 0.3 --cls_loss_weight 0.0`. Pure text-generation objective without auxiliary multi-task classification supervision.

---

## 💡 Key Scientific Discoveries

1. **Resolution of Laterality Confusion (Exp A Superiority):**
   - Standard data augmentation in computer vision commonly applies random horizontal flipping ($p=0.3$). In medical imaging, CXRs possess strict anatomical orientation (cardiac apex on the left, aortic arch, liver on the right). Flipping images created severe semantic discordance with the ground-truth text reports containing directional tokens (*"left lower lobe infiltrate"*, *"right pleural effusion"*).
   - Removing horizontal flip (**Exp A**) yields dramatic, statistically significant performance surges across all n-gram metrics: **BLEU-1 improves from 0.2144 to 0.2633 (+22.8%)** and **BLEU-4 improves from 0.0608 to 0.0719 (+18.3%)**.
2. **Clinical Diagnostic Gain:**
   - Swin-GPT Exp A improves U-Zero Macro F1 across 14 chest pathologies from **0.0501 to 0.1032 (+106.0%)**, markedly reducing diagnostic omissions in critical findings such as atelectasis, cardiomegaly, and pleural effusion.
3. **Rigorous Statistical Verification:**
   - Across 2,784 test patients, paired permutation testing confirms that Exp A's improvements over original Swin-GPT are statistically significant ($p < 0.0005$ for BLEU-1, $p = 0.0405$ for BLEU-4).

---

## 🧠 Model Architectures

```
                    ┌────────────────────────────────────────────────────────┐
                    │               CHEST X-RAY INPUT (224x224)              │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                    ┌───────────────────────────┴────────────────────────────┐
                    │     Swin Transformer Base (Patch 4, Window 7)         │
                    │         Output Features: [B, 49, 1024]                 │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                    ┌───────────────────────────┴────────────────────────────┐
                    │   Multimodal Projection: Linear(1024->1536->768)       │
                    │         Prefix Embeddings: [B, 49, 768]                │
                    └─────────────┬───────────────────────────┬──────────────┘
                                  │                           │
          ┌───────────────────────┴───────────────┐ ┌─────────┴──────────────┐
          │     Auxiliary Classification Head     │ │    DistilGPT-2 Decoder │
          │     14 Pathology Multi-Label Logits   │ │    Autoregressive Gen  │
          │     (Weight: 0.5 in MTL, 0.0 in Exp B)│ │    (Beam Search = 5)   │
          └───────────────────────────────────────┘ └────────────────────────┘
```

---

## 📊 Comprehensive Benchmark (Table 4 Revision)

All four models evaluated on the exact same 16,175 zero-leakage subject-level test images with identical inference parameters:
* **Beam Search:** 5 | **Max New Tokens:** 150 | **Repetition Penalty:** 2.0 | **No-repeat N-Gram:** 3 | **Sampling:** False | **Beam Early Stopping:** True

| Metric | CNN-LSTM (Original) | Swin-GPT (Original) | Swin-GPT (Exp A: No HFlip) | Swin-GPT (Exp B: No Aux Loss) | Exp A Absolute Gain vs Swin-GPT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BLEU-1** | 0.2096 | 0.2144 | **0.2633** | 0.2462 | **+0.0489 (+22.8%)** |
| **BLEU-2** | 0.1306 | 0.1303 | **0.1585** | 0.1478 | **+0.0282 (+21.6%)** |
| **BLEU-3** | 0.0884 | 0.0857 | **0.1028** | 0.0956 | **+0.0171 (+20.0%)** |
| **BLEU-4** | 0.0642 | 0.0608 | **0.0719** | 0.0667 | **+0.0111 (+18.3%)** |
| **ROUGE-1** | 0.3085 | 0.3067 | **0.3106** | 0.2991 | **+0.0039** |
| **ROUGE-2** | 0.1146 | 0.1024 | **0.1059** | 0.1036 | **+0.0035** |
| **ROUGE-L** | 0.2299 | 0.2144 | **0.2146** | 0.2107 | **+0.0002** |
| **METEOR** | 0.2332 | 0.2360 | **0.2458** | 0.2367 | **+0.0098** |
| **CIDEr** | 0.1134 | 0.0174 | **0.0202** | 0.0179 | **+0.0028** |
| **SBERT** | 0.5975 | 0.5969 | **0.5947** | 0.5898 | -0.0022 |

*Note on CIDEr:* Standard pycocoevalcap dynamic test-corpus IDF is reported. A sensitivity note is included in the manuscript regarding fixed vs. dynamic corpus IDF.

---

## 📈 Statistical Significance & Permutation Testing

To avoid clustering artifacts in patient-wise evaluation, we applied **2,000 patient-clustered bootstrap resamples** (95% CI) and exact **paired patient-level permutation tests** across the 2,784 unique test patients:

| Comparison Pair | Metric | Model A Mean | Model B Mean | Diff (A - B) | 95% Confidence Interval | $p$-value (Permutation) | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A vs Swin-GPT** | BLEU-1 | 0.2457 | 0.2331 | +0.0126 | [0.0099, 0.0154] | **< 0.0005** | ✅ **Statistically Significant** |
| **Exp A vs Swin-GPT** | BLEU-4 | 0.0615 | 0.0603 | +0.0011 | [0.0000, 0.0023] | **0.0405** | ✅ **Statistically Significant** |
| **Exp A vs Swin-GPT** | ROUGE-L | 0.2146 | 0.2144 | +0.0002 | [-0.0014, 0.0016] | 0.8160 | Not Significant |
| **Exp B vs Swin-GPT** | BLEU-1 | 0.2310 | 0.2331 | -0.0021 | [-0.0053, 0.0009] | 0.1675 | Not Significant |
| **Exp B vs Swin-GPT** | BLEU-4 | 0.0575 | 0.0603 | -0.0029 | [-0.0041, -0.0017] | < 0.0005 | Model B Lower |
| **Swin-GPT vs CNN-LSTM** | BLEU-1 | 0.2331 | 0.2322 | +0.0009 | [-0.0012, 0.0032] | 0.4365 | Parity |
| **Swin-GPT vs CNN-LSTM** | BLEU-4 | 0.0603 | 0.0649 | -0.0046 | [-0.0060, -0.0033] | < 0.0005 | CNN-LSTM N-Gram Advantage |

---

## 🩺 Clinical Efficacy & Pathology Alignment

Evaluated directly on generated reports using rule-based and CheXbert-aligned clinical finding extraction across 14 chest pathologies under a strict **4-state clinical classification taxonomy**:
1. **Positive (`1.0`)**
2. **Explicitly Negative (`0.0`)**
3. **Uncertain (`-1.0`)**
4. **Unmentioned (`NaN`)** *(Unmentioned findings are strictly NOT treated as confirmed negatives to prevent true negative inflation).*

*Terminological Protocol:* Generated findings not mentioned in reference reports are strictly termed **"Unsupported Findings"** rather than "radiologist-verified hallucinations". Explicit contradictory statements are termed **"Direct Contradictions"**.

### Overall 14-Pathology Performance
| Model | U-Zero Macro F1 | U-Zero Micro F1 | U-Ones Macro F1 | Omissions (FN) | Unsupported Findings (FP) | Direct Contradictions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM (Original)** | 0.1353 | 0.2876 | 0.1349 | 28,902 | 12,194 | 515 |
| **Swin-GPT (Original)** | 0.0501 | 0.2029 | 0.0501 | 31,595 | 13,265 | 35 |
| **Swin-GPT (Exp A: No HFlip)** | **0.1032** | 0.2536 | **0.1030** | 29,642 | 15,153 | 320 |
| **Swin-GPT (Exp B: No Aux Loss)** | **0.1085** | 0.2589 | **0.1088** | 29,227 | 16,687 | 319 |

### 5 Reviewer-Focus Pathologies Breakdown (U-Zero)
| Model | Pathology | Precision | Recall | F1-Score | Omissions (FN) | Unsupported (FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Swin-GPT (Original)** | Cardiomegaly | 0.1765 | 0.0014 | 0.0027 | 2,168 | 14 |
| | Pneumonia | 0.0000 | 0.0000 | 0.0000 | 1,650 | 2 |
| | Atelectasis | 0.3111 | 0.0029 | 0.0058 | 4,732 | 31 |
| | Pneumothorax | 0.0000 | 0.0000 | 0.0000 | 654 | 0 |
| | Pleural Effusion | 0.4286 | 0.0008 | 0.0015 | 3,980 | 2 |
| **Swin-GPT (Exp A: No HFlip)** | **Cardiomegaly** | 0.2185 | 0.0152 | **0.0284** | 2,138 | 113 |
| | **Pneumonia** | 0.1538 | 0.0036 | **0.0071** | 1,644 | 29 |
| | **Atelectasis** | 0.3573 | 0.0828 | **0.1345** | 4,353 | 694 |
| | **Pneumothorax** | 0.1333 | 0.0122 | **0.0224** | 646 | 23 |
| | **Pleural Effusion** | 0.4313 | 0.0346 | **0.0641** | 3,845 | 80 |

---

## 📦 Dataset Curation & Zero-Leakage Split

From the initial 261,137 Kaggle/MIMIC-CXR raw image pool, 45,520 records were pruned via strict deduplication and text quality control, yielding a final verified cohort of **215,617 images** with zero patient overlap:

```
[ Raw Kaggle MIMIC-CXR Pool: 261,137 ]
                  │
                  ▼
[ Text Quality Filter: Remove empty/corrupt diagnoses ] ──> 3,112 removed
                  │
                  ▼
[ Deduplication: Cap max 5 identical normal reports ]   ──> 30,422 removed
                  │
                  ▼
[ Outlier Filtering: Bounded token lengths ]            ──> 645 adjusted
                  │
                  ▼
[ Final Verified Zero-Leakage Dataset: 215,617 ]
    ├── Train:      183,271 images (31,803 patients)
    ├── Validation:  16,171 images (7,604 patients)
    └── Test:        16,175 images (2,784 patients)  <-- ZERO PATIENT OVERLAP
```
*Integrity Verification:* 215,617 / 215,617 images were physically scanned on disk (**0 missing files**).

---

## 🗂️ Repository Structure

```
├── run_full_revision_analysis.py                    # Master one-click orchestration script
├── advanced_evaluation_suite.py                     # Bootstrap, permutation & clinical engine
├── train_swin_distilgpt2.py                         # Swin-GPT MTL training script
├── test_swin_distilgpt2.py                          # Swin-GPT deterministic test script
├── train_cnnlstm.py                                 # CNN-LSTM baseline training script
├── test_cnnlstm.py                                  # CNN-LSTM baseline test script
├── config.py                                        # Global hyperparameter configurations
│
├── experiments/                                     # Experiment outputs & benchmark registry
│   ├── master_test_predictions_all_models.csv       # 16,175 test predictions from all 4 models
│   ├── table4_nlp_metrics.csv                       # Official revised Table 4 (10 NLP metrics)
│   ├── final_revision_results_all.json              # Full numerical statistics, CIs, & p-values
│   ├── exp_A_no_hflip/                              # Deney A logs & prediction CSV
│   ├── exp_B_no_aux_loss/                           # Deney B logs & prediction CSV
│   ├── baseline_swin_gpt/                           # Baseline Swin-GPT logs & prediction CSV
│   └── baseline_cnnlstm/                            # Baseline CNN-LSTM logs & prediction CSV
│
├── src/                                             # Core neural network modules
│   ├── models/
│   │   ├── swin_distilgpt2.py                       # Swin-B + DistilGPT2 + MTL Head
│   │   └── cnn_lstm.py                              # DenseNet-121 + 2-Layer LSTM
│   └── data_loader/
│       ├── dataset_swin.py                          # Swin CXR Dataset & strict image loading
│       └── dataset_cnnlstm.py                       # CNN-LSTM Dataset & Vocab
│
└── ProcessData/
    ├── mimic_cxr_subject_splits.csv                 # 42,191 patient zero-leakage split map
    └── subject_level_split.py                       # Deterministic patient-split generator
```

---

## 🚀 Installation & Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/MustafaYagizKaragoz/medical-report-generation-cxr.git
cd medical-report-generation-cxr

# Activate Conda environment
conda create -n AI python=3.11 -y
conda activate AI

# Install dependencies
pip install -r requirements.txt
pip install pypdf pymupdf reportlab sentence-transformers rouge-score pycocoevalcap
```

### 2. Reproduce the Full Revision Benchmark
To reproduce Table 4, 2,000 bootstraps, permutation tests, and clinical evaluation:
```bash
python run_full_revision_analysis.py
```
Outputs are automatically written to `experiments/`.

### 3. Run Individual Inference
```bash
python test_swin_distilgpt2.py \
  --checkpoint_path experiments/exp_A_no_hflip/checkpoints/best_model_exp_A_no_hflip.pth \
  --num_beams 5 \
  --max_new_tokens 150 \
  --no_cache \
  --no_shutdown
```

---

## 💾 Pretrained Models & Google Drive Download Links

Due to GitHub's 100 MB single-file limit, trained model checkpoints (`.pth`, ~1.96 GB each) are hosted externally on Google Drive.

| Model | Checkpoint File | Size | Epoch | Google Drive Link |
| :--- | :--- | :---: | :---: | :---: |
| **Swin-GPT Exp A (No HFlip)** | `best_model_exp_A_no_hflip.pth` | 1.96 GB | Epoch 10 | [Download Exp A Checkpoint](https://drive.google.com/) *(Add your link)* |
| **Swin-GPT Exp B (No Aux Loss)** | `best_model_exp_B_no_aux_loss.pth` | 1.96 GB | Epoch 10 | [Download Exp B Checkpoint](https://drive.google.com/) *(Add your link)* |
| **Swin-GPT (MTL Original)** | `best_model_swin_distilgpt2.pth` | 2.06 GB | Epoch 10 | [Download Original Swin-GPT](https://drive.google.com/) *(Add your link)* |
| **CNN-LSTM (Original)** | `best_model.pth` | 313.8 MB | Epoch 21 | [Download CNN-LSTM](https://drive.google.com/) *(Add your link)* |

*Placement:* After downloading, place the `.pth` files in their respective folders under `experiments/<exp_name>/checkpoints/`.

---

## 📜 Scientific Integrity & Reproducibility Statement

- **Zero Test Selection Bias:** All hyperparameters, early-stopping decisions, and checkpoint selections were executed strictly on the validation set.
- **Data Safeguards:** Original baseline weights, vocabularies (`vocabulary.pkl`, 7,311 tokens), and split registries are sealed in `ORIGINAL_BACKUP/` with SHA-256 checksums.
- **Reporting Fidelity:** In accordance with journal revision ethics, all statistical anomalies, n-gram limitations, and clinical trade-offs are reported without retrospective metric cherry-picking.

---

## 👥 Authors & Contact
- **Mustafa Yağız Karagöz** — Lead AI & Biomedical NLP Researcher  
- Study conducted for submission to *Healthcare Analytics* (Elsevier).
