# final_validation.py
import pandas as pd
from pathlib import Path
import os

def extract_subject_id(path):
    if pd.isna(path):
        return None
    path = str(path).replace('\\', '/')
    parts = path.split('/')
    for p in parts:
        if p.startswith('p') and len(p) == 9 and p[1:].isdigit():
            return int(p[1:])
    return None

def extract_study_id(path):
    if pd.isna(path):
        return None
    path = str(path).replace('\\', '/')
    parts = path.split('/')
    for p in parts:
        if p.startswith('s') and len(p) == 9 and p[1:].isdigit():
            return int(p[1:])
    return None

def validate_split_group(train_path, val_path, test_path, group_name):
    print("\n" + "="*70)
    print(f"VALİDASYON GRUBU: {group_name.upper()}")
    print("="*70)
    
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        print(f"UYARI: {group_name} grubu için gerekli CSV dosyaları bulunamadı.")
        return
        
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Kayıt Sayıları:")
    print(f"  Train: {len(df_train):,}")
    print(f"  Val:   {len(df_val):,}")
    print(f"  Test:  {len(df_test):,}")
    
    # Extract IDs
    for df in [df_train, df_val, df_test]:
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['image_path'].apply(extract_subject_id)
        if 'study_id' not in df.columns:
            df['study_id'] = df['image_path'].apply(extract_study_id)
            
    # Boş değerler kontrolü
    for name, df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        nulls = df.isnull().sum()
        null_cols = [f"{col}: {count}" for col, count in nulls.items() if count > 0]
        if null_cols:
            print(f"  [UYARI] {name} kümesinde boş değerli kolonlar: {null_cols}")
            
    # Sets for overlap checks
    train_subjects = set(df_train['subject_id'].dropna().unique())
    val_subjects = set(df_val['subject_id'].dropna().unique())
    test_subjects = set(df_test['subject_id'].dropna().unique())
    
    train_studies = set(df_train['study_id'].dropna().unique())
    val_studies = set(df_val['study_id'].dropna().unique())
    test_studies = set(df_test['study_id'].dropna().unique())
    
    train_images = set(df_train['image_path'].dropna().unique())
    val_images = set(df_val['image_path'].dropna().unique())
    test_images = set(df_test['image_path'].dropna().unique())
    
    print(f"\nBenzersiz Varlık Sayıları:")
    print(f"  Hastalar (Patients) -> Train: {len(train_subjects):,}, Val: {len(val_subjects):,}, Test: {len(test_subjects):,}")
    print(f"  Çalışmalar (Studies) -> Train: {len(train_studies):,}, Val: {len(val_studies):,}, Test: {len(test_studies):,}")
    print(f"  Görüntüler (Images)  -> Train: {len(train_images):,}, Val: {len(val_images):,}, Test: {len(test_images):,}")
    
    # Overlaps
    # 1. Subject level
    s_tr_val = train_subjects & val_subjects
    s_tr_te = train_subjects & test_subjects
    s_val_te = val_subjects & test_subjects
    
    # 2. Study level
    st_tr_val = train_studies & val_studies
    st_tr_te = train_studies & test_studies
    st_val_te = val_studies & test_studies
    
    # 3. Image level
    img_tr_val = train_images & val_images
    img_tr_te = train_images & test_images
    img_val_te = val_images & test_images
    
    print(f"\nÇakışma (Overlap) Analizi:")
    print(f"  1. Hasta (Subject ID) Seviyesi:")
    print(f"     Train & Val  çakışma: {len(s_tr_val):,} / {len(val_subjects):,} ({len(s_tr_val)/max(1,len(val_subjects))*100:.2f}%)")
    print(f"     Train & Test çakışma: {len(s_tr_te):,} / {len(test_subjects):,} ({len(s_tr_te)/max(1,len(test_subjects))*100:.2f}%)")
    print(f"     Val & Test   çakışma: {len(s_val_te):,} / {len(test_subjects):,} ({len(s_val_te)/max(1,len(test_subjects))*100:.2f}%)")
    
    print(f"  2. Çalışma (Study ID) Seviyesi:")
    print(f"     Train & Val  çakışma: {len(st_tr_val):,} ({len(st_tr_val)/max(1,len(val_studies))*100:.2f}%)")
    print(f"     Train & Test çakışma: {len(st_tr_te):,} ({len(st_tr_te)/max(1,len(test_studies))*100:.2f}%)")
    print(f"     Val & Test   çakışma: {len(st_val_te):,} ({len(st_val_te)/max(1,len(test_studies))*100:.2f}%)")
    
    print(f"  3. Görüntü (Image Path) Seviyesi:")
    print(f"     Train & Val  çakışma: {len(img_tr_val):,}")
    print(f"     Train & Test çakışma: {len(img_tr_te):,}")
    print(f"     Val & Test   çakışma: {len(img_val_te):,}")
    
    # Leakage verdict
    if len(img_tr_val) > 0 or len(img_tr_te) > 0 or len(img_val_te) > 0:
        print("\n  [DURUM] CRITICAL WARNING: Image path level leakage detected! The same image exists in multiple splits.")
    elif len(st_tr_val) > 0 or len(st_tr_te) > 0 or len(st_val_te) > 0:
        print("\n  [DURUM] WARNING: Study level leakage detected. The same study folder belongs to multiple splits.")
    elif len(s_tr_val) > 0 or len(s_tr_te) > 0 or len(s_val_te) > 0:
        print("\n  [DURUM] INFO: Subject level leakage exists (Image-level split).")
    else:
        print("\n  [DURUM] SUCCESS: Zero patient leakage. Split is perfectly patient-wise!")

def validate_all_data():
    OUTPUT_DIR = Path(r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed')
    
    # Group A: Active training files
    validate_split_group(
        train_path=OUTPUT_DIR / "labeled_reports_train.csv",
        val_path=OUTPUT_DIR / "labeled_reports_val.csv",
        test_path=OUTPUT_DIR / "labeled_reports_test.csv",
        group_name="labeled_reports (aktif model girdileri)"
    )
    
    # Group B: Alternative processed files
    validate_split_group(
        train_path=OUTPUT_DIR / "train_processed.csv",
        val_path=OUTPUT_DIR / "val_processed.csv",
        test_path=OUTPUT_DIR / "test_processed.csv",
        group_name="processed_files (alternatif ön işlenmişler)"
    )

if __name__ == "__main__":
    validate_all_data()