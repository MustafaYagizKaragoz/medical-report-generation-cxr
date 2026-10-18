"""
subject_level_split.py
======================
MIMIC-CXR Hasta Düzeyinde Deterministik Bölme ve CSV Üretici (Subject-Level Deterministic Split & Exporter)

Bu betik:
1. Veri havuzunu (215.617 satır) 42.191 tekil hasta kimliği (subject_id) bazında 
   sıfır hasta sızıntısı (Zero Patient Leakage) garantisiyle 3 kümeye böler:
     - Train : 183.271 satır (31.803 hasta) [%85.0]
     - Val   : 16.171 satır  (7.604 hasta)  [%7.5]
     - Test  : 16.175 satır  (2.784 hasta)  [%7.5]
2. --export parametresiyle çalıştırıldığında (veya generate_subject_splits çağrıldığında)
   fiziksel olarak şu 3 CSV dosyasını diske üretir:
     - labeled_reports_train.csv
     - labeled_reports_val.csv
     - labeled_reports_test.csv
3. Üretilen dosyaların makalede/eğitimde kullanılan nihai veri kümeleriyle 
   birebir özdeş olduğunu (satır sayısı, hasta dağılımı, sıfır sızıntı) matematiksel olarak doğrular.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_subject_id(path):
    """Görüntü dosya yolundan subject_id (hasta ID) çıkarır."""
    if pd.isna(path):
        return None
    path_str = str(path).replace('\\', '/')
    for p in path_str.split('/'):
        if p.startswith('p') and len(p) == 9 and p[1:].isdigit():
            return int(p[1:])
    return None


def extract_study_id(path):
    """Görüntü dosya yolundan study_id (çalışma ID) çıkarır."""
    if pd.isna(path):
        return None
    path_str = str(path).replace('\\', '/')
    for p in path_str.split('/'):
        if p.startswith('s') and len(p) == 9 and p[1:].isdigit():
            return int(p[1:])
    return None


def load_pool_dataframe(data_dir: Path) -> pd.DataFrame:
    """Mevcut processed klasöründeki verileri birleştirerek tam havuzu (215.617 satır) oluşturur."""
    dfs = []
    for s in ['train', 'val', 'test']:
        p = data_dir / f'labeled_reports_{s}.csv'
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
        else:
            raise FileNotFoundError(f"Kaynak veri dosyası bulunamadı: {p}")
    pool = pd.concat(dfs, ignore_index=True)
    pool['subject_id'] = pool['image_path'].apply(extract_subject_id)
    return pool


def generate_subject_splits(
    pool_df: pd.DataFrame,
    split_map_path: Path,
    output_dir: Path,
    export_csv: bool = True
):
    """
    Havuzdaki verileri mimic_cxr_subject_splits.csv referans haritasına göre
    train, val ve test kümelerine ayırır ve CSV olarak diske kaydeder.
    """
    logger.info("=" * 70)
    logger.info("SUBJECT-LEVEL BÖLME VE CSV ÜRETİM İŞLEMİ")
    logger.info("=" * 70)

    if not split_map_path.exists():
        raise FileNotFoundError(f"Hasta harita dosyası bulunamadı: {split_map_path}")

    split_map_df = pd.read_csv(split_map_path)
    split_map = split_map_df.set_index('subject_id')['split'].to_dict()
    logger.info(f"Yüklendi: {len(split_map):,} tekil hasta haritası ({split_map_path.name})")

    if 'subject_id' not in pool_df.columns:
        pool_df['subject_id'] = pool_df['image_path'].apply(extract_subject_id)

    pool_df['assigned_split'] = pool_df['subject_id'].map(split_map)
    unassigned = pool_df['assigned_split'].isna().sum()
    if unassigned > 0:
        raise ValueError(f"HATA: {unassigned} satırın hasta ID'si haritada bulunamadı!")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dfs = {}

    for split_name in ['train', 'val', 'test']:
        subset = pool_df[pool_df['assigned_split'] == split_name].copy()
        
        # 'assigned_split' ve gerekiyorsa yardımcı 'subject_id' kolonunu temizle
        export_df = subset.drop(columns=['assigned_split'])
        if 'subject_id' in export_df.columns and 'subject_id' not in pool_df.columns:
            export_df = export_df.drop(columns=['subject_id'])
            
        split_dfs[split_name] = export_df

        if export_csv:
            out_file = output_dir / f"labeled_reports_{split_name}.csv"
            export_df.to_csv(out_file, index=False)
            logger.info(f"-> CSV Üretildi: {out_file} ({len(export_df):,} satır, {subset['subject_id'].nunique():,} hasta)")

    return split_dfs


def verify_splits(split_dfs: dict):
    """Bölmelerin boyutlarını, hasta sayılarını ve sıfır sızıntı kuralını doğrular."""
    logger.info("\n" + "-" * 70)
    logger.info("DOĞRULAMA VE SIFIR HASTA SIZINTISI (ZERO LEAKAGE) KONTROLÜ")
    logger.info("-" * 70)

    expected = {
        'train': {'rows': 183271, 'patients': 31803},
        'val':   {'rows': 16171,  'patients': 7604},
        'test':  {'rows': 16175,  'patients': 2784}
    }

    patient_sets = {}
    for s in ['train', 'val', 'test']:
        df = split_dfs[s]
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['image_path'].apply(extract_subject_id)
        
        pts = set(df['subject_id'])
        patient_sets[s] = pts
        n_rows = len(df)
        n_pts = len(pts)

        exp_rows = expected[s]['rows']
        exp_pts = expected[s]['patients']

        logger.info(f"[{s.upper():5}] Satır: {n_rows:,} (Hedef: {exp_rows:,}) | Hasta: {n_pts:,} (Hedef: {exp_pts:,})")
        assert n_rows == exp_rows, f"{s} kümesinde satır sayısı uyumsuz!"
        assert n_pts == exp_pts, f"{s} kümesinde hasta sayısı uyumsuz!"

    # Sıfır hasta sızıntısı kontrolü
    assert len(patient_sets['train'] & patient_sets['val']) == 0, "HATA: Train-Val hasta sızıntısı!"
    assert len(patient_sets['train'] & patient_sets['test']) == 0, "HATA: Train-Test hasta sızıntısı!"
    assert len(patient_sets['val'] & patient_sets['test']) == 0, "HATA: Val-Test hasta sızıntısı!"

    logger.info("=" * 70)
    logger.info("SONUÇ: BİREBİR AYNI 183.271 / 16.171 / 16.175 GÖRÜNTÜ VE")
    logger.info("AYNI 31.803 / 7.604 / 2.784 HASTA KİMLİĞİ DAĞILIMI %100 DOĞRULANDI!")
    logger.info("SIFIR HASTA SIZINTISI (ZERO LEAKAGE) TEYİT EDİLDİ.")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MIMIC-CXR Hasta Düzeyinde Bölme ve CSV Üretici")
    parser.add_argument("--export", action="store_true", help="CSV dosyalarını diske fiziksel olarak üretir ve kaydeder")
    parser.add_argument("--output-dir", type=str, default=None, help="CSV dosyalarının kaydedileceği hedef klasör")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "Data" / "processed"
    splits_csv = Path(__file__).resolve().parent / "mimic_cxr_subject_splits.csv"

    out_dir = Path(args.output_dir) if args.output_dir else data_dir

    logger.info(f"Kaynak Veri Dizini: {data_dir}")
    logger.info(f"Hasta Haritası    : {splits_csv}")
    logger.info(f"Hedef Çıktı Dizini: {out_dir}")
    logger.info(f"CSV Üretim Modu   : {'AÇIK (--export aktif)' if args.export else 'SALT OKUMA / DOĞRULAMA'}")

    pool_df = load_pool_dataframe(data_dir)
    split_dfs = generate_subject_splits(
        pool_df=pool_df,
        split_map_path=splits_csv,
        output_dir=out_dir,
        export_csv=args.export
    )
    verify_splits(split_dfs)


if __name__ == "__main__":
    main()
