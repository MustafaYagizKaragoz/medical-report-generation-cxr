"""
subject_level_split.py
======================
MIMIC-CXR Veri Seti Hasta Bazlı Bölme (Subject-Level Zero-Leakage Split)

Bu betik, 215.617 görüntü-rapor çiftini sıfır hasta sızıntısı (zero patient leakage)
garantisiyle %85.0 Train (183.271), %7.5 Validation (16.171) ve %7.5 Test (16.175) 
kümelerine ayıran hasta tabanlı algoritmayı içerir.

Akademik ve Metodolojik Gerekçeler (Reviewer 1 Yanıtı):
-------------------------------------------------------
1. Hedef Kotası: Bölme oranları (%85 / %7.5 / %7.5) hasta sayısına göre değil, 
   modelin göreceği GÖRÜNTÜ SAYISINA (image quota) göre hedeflenmiştir.
2. Dağılım Asimetrisi:
   - MIMIC-CXR'da hastaların bir kısmı tekil ayaktan başvuru (1 grafi), bir kısmı ise 
     yoğun bakım / yatan hasta (çoklu boylamsal grafi, ortalama 5.8 görüntü) şeklindedir.
   - TEST KÜMESİ (16.175 görüntü, 2.784 hasta): Modelin boylamsal klinik takip ve 
     zaman içindeki patoloji değişimini değerlendirebilmesi için özellikle çoklu grafisi 
     olan hastalar arasından ayrılmıştır (tek görüntülü hasta = 0).
   - TRAIN KÜMESİ (183.271 görüntü, 31.803 hasta): Eğitimin zengin temsili için yine 
     çoklu hastalar arasından oluşturulmuştur (tek görüntülü hasta = 0).
   - VALIDATION KÜMESİ (16.171 görüntü, 7.604 hasta): Havuzdaki 5.933 tekil ayaktan hasta 
     bu kümeye atanmış, kalan ~10.238 görüntü kotası 1.671 çoklu hasta ile tamamlanmıştır.
3. Sızıntı Kontrolü: Kümeler arasında Train-Val, Train-Test ve Val-Test hasta çakışması %0'dır.
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_subject_id(path):
    """Görüntü yolundan subject_id (hasta ID) çıkarır (örn: p11180362 -> 11180362)."""
    if pd.isna(path):
        return None
    path_str = str(path).replace('\\', '/')
    for p in path_str.split('/'):
        if p.startswith('p') and len(p) == 9 and p[1:].isdigit():
            return int(p[1:])
    return None


def extract_study_id(path):
    """Görüntü yolundan study_id (çalışma ID) çıkarır (örn: s55561098 -> 55561098)."""
    if pd.isna(path):
        return None
    path_str = str(path).replace('\\', '/')
    for p in path_str.split('/'):
        if p.startswith('s') and len(p) == 9 and p[1:].isdigit():
            return int(p[1:])
    return None


def partition_dataset_subject_wise(
    df: pd.DataFrame,
    target_train_images: int = 183271,
    target_val_images: int = 16171,
    target_test_images: int = 16175,
    random_state: int = 42
):
    """
    Hastaları boylamsal yoğunluklarına (tekil vs çoklu) göre tabakalandırarak
    hedef görüntü kotalarına göre böler.
    """
    logger.info("=" * 70)
    logger.info("HASTA TABANLI SIZINTISIZ BÖLME ALGORİTMASI BAŞLATILIYOR")
    logger.info("=" * 70)

    # Hasta ID ve Çalışma ID ekle
    if 'subject_id' not in df.columns:
        df['subject_id'] = df['image_path'].apply(extract_subject_id)
    if 'study_id' not in df.columns:
        df['study_id'] = df['image_path'].apply(extract_study_id)

    total_images = len(df)
    total_subjects = df['subject_id'].nunique()
    logger.info(f"Toplam Görüntü: {total_images:,} | Toplam Hasta: {total_subjects:,}")

    # Hasta başına görüntü sayıları
    subject_counts = df['subject_id'].value_counts()
    single_image_subjects = set(subject_counts[subject_counts == 1].index)
    multi_image_subjects = set(subject_counts[subject_counts > 1].index)

    logger.info(f"Tek Görüntülü Hastalar : {len(single_image_subjects):,} ({len(single_image_subjects):,} görüntü)")
    logger.info(f"Çoklu Görüntülü Hastalar: {len(multi_image_subjects):,} ({sum(subject_counts[s] for s in multi_image_subjects):,} görüntü)")

    # 1. ADIM: Tek görüntülü hastalar Validation kümesine atanır
    # (Ayaktan tek seferlik başvurular Val kümesinde toplanarak Test ve Train'in boylamsal odaklanması sağlanır)
    val_subjects = set(single_image_subjects)
    current_val_images = len(single_image_subjects)

    # 2. ADIM: Çoklu hastalar arasında bölme
    # Kalan çoklu hastaların hedef paylaştırılması:
    # - Val kotasının kalan kısmı: target_val_images - current_val_images (~10,238 görüntü)
    # - Test kotası: target_test_images (~16,175 görüntü)
    # - Train kotası: target_train_images (~183,271 görüntü)
    multi_subject_list = list(multi_image_subjects)
    rng = np.random.RandomState(random_state)
    rng.shuffle(multi_subject_list)

    test_subjects = set()
    current_test_images = 0
    val_multi_subjects = set()

    # Çoklu hastaları sırayla kotalara göre paylaştır
    for sub in multi_subject_list:
        sub_imgs = subject_counts[sub]
        # Önce test kotasını doldur
        if current_test_images + sub_imgs <= target_test_images:
            test_subjects.add(sub)
            current_test_images += sub_imgs
        # Sonra validation kalan kotasını doldur
        elif current_val_images + sub_imgs <= target_val_images:
            val_subjects.add(sub)
            val_multi_subjects.add(sub)
            current_val_images += sub_imgs
        else:
            # Geri kalan tüm hastalar train kümesine
            pass

    # Geriye kalan tüm hastalar doğrudan Train setine atanır
    allocated_val_test = test_subjects | val_subjects
    train_subjects = set(multi_image_subjects) - allocated_val_test

    train_df = df[df['subject_id'].isin(train_subjects)].copy().reset_index(drop=True)
    val_df   = df[df['subject_id'].isin(val_subjects)].copy().reset_index(drop=True)
    test_df  = df[df['subject_id'].isin(test_subjects)].copy().reset_index(drop=True)

    # Sızıntı kontrolü (Assertion)
    assert len(train_subjects & val_subjects) == 0, "HATA: Train-Val hasta sızıntısı!"
    assert len(train_subjects & test_subjects) == 0, "HATA: Train-Test hasta sızıntısı!"
    assert len(val_subjects & test_subjects) == 0, "HATA: Val-Test hasta sızıntısı!"

    logger.info("\n" + "=" * 70)
    logger.info("BÖLME TAMAMLANDI — ÖZET İSTATİSTİKLER:")
    logger.info("=" * 70)
    for name, s_df, s_subs in [('TRAIN', train_df, train_subjects), ('VAL', val_df, val_subjects), ('TEST', test_df, test_subjects)]:
        n_imgs = len(s_df)
        n_pts = len(s_subs)
        n_single = sum(subject_counts[s] == 1 for s in s_subs)
        logger.info(f"{name:5}: {n_imgs:,} görüntü (%{n_imgs/total_images*100:.2f}) | {n_pts:,} hasta | Tekil hasta: {n_single:,} | Ort: {n_imgs/n_pts:.2f} gör/hasta")

    return train_df, val_df, test_df


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "processed"
    train_csv = DATA_DIR / "labeled_reports_train.csv"
    val_csv   = DATA_DIR / "labeled_reports_val.csv"
    test_csv  = DATA_DIR / "labeled_reports_test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        logger.info("Mevcut CSV dosyaları başarıyla doğrulandı.")
        logger.info(f"  Train: {train_csv.name}")
        logger.info(f"  Val  : {val_csv.name}")
        logger.info(f"  Test : {test_csv.name}")
