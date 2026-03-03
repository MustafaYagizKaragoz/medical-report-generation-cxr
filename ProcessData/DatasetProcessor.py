# train_val_cleaner.py - Bu dosyayı çalıştırın

import pandas as pd
import numpy as np
import json
import ast
import re
from pathlib import Path
from collections import Counter
import logging
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MimicCXRCleaner:
    """MIMIC-CXR Train ve Val için temizleme sınıfı"""
    
    def __init__(self, csv_path, image_root, output_path, split_name):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name  # 'train' veya 'val'
        
        self.stats = {
            'initial_subjects': 0,
            'initial_records': 0,
            'after_cleaning': 0,
            'after_quality_filter': 0,
            'final_records': 0
        }
    
    def _safe_eval(self, val):
        """String listeleri parse et"""
        if pd.isna(val):
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                try:
                    return json.loads(val.replace("'", '"'))
                except:
                    return []
        return []
    
    def clean_report_text(self, text):
        """Rapor temizleme"""
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
            return None
            
        text = text.strip()
        
        # Temizleme kuralları
        text = re.sub(r'___+', ' ', text)
        text = re.sub(r'\[\*\*.*?\*\*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(?i)findings\s*:', 'Findings:', text)
        text = re.sub(r'(?i)impression\s*:', 'Impression:', text)
        
        # Geçersiz rapor kontrolü
        invalid_patterns = [
            r'^no (report|findings|impression)',
            r'^report not available',
            r'^exam not performed',
            r'^\s*$',
        ]
        
        text_lower = text.lower()
        for pattern in invalid_patterns:
            if re.match(pattern, text_lower):
                return None
        
        if len(text.split()) < 5:
            return None
            
        return text.strip()
    
    def extract_sections(self, text):
        """Findings ve Impression ayır"""
        if not text:
            return "", "", text
            
        findings = ""
        impression = ""
        
        # Findings bul
        findings_match = re.search(r'Findings:(.+?)(?=Impression:|$)', text, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings = findings_match.group(1).strip()
        
        # Impression bul
        impression_match = re.search(r'Impression:(.+?)$', text, re.DOTALL | re.IGNORECASE)
        if impression_match:
            impression = impression_match.group(1).strip()
        
        # Eğer bölüm yoksa
        if not findings and not impression:
            impression = text
            
        return findings, impression, text
    
    def verify_image(self, image_path):
        """Görüntü kontrolü"""
        full_path = self.image_root / image_path
        if not full_path.exists():
            return False
        try:
            with Image.open(full_path) as img:
                if img.size[0] < 100 or img.size[1] < 100:
                    return False
                img.verify()
            return True
        except:
            return False
    
    def process_dataset(self):
        """Ana işlem fonksiyonu"""
        logger.info(f"\n{'='*60}")
        logger.info(f"{self.split_name.upper()} SET TEMİZLENİYOR")
        logger.info(f"{'='*60}")
        
        # 1. CSV yükle
        logger.info(f"Yükleniyor: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        self.stats['initial_subjects'] = len(df)
        logger.info(f"Başlangıç hasta sayısı: {len(df)}")
        
        # 2. Her satırı işle
        records = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{self.split_name} işleniyor"):
            subject_id = row['subject_id']
            
            # Listeleri parse et
            images = self._safe_eval(row['image'])
            views = self._safe_eval(row['view'])
            texts = self._safe_eval(row['text'])
            texts_aug = self._safe_eval(row.get('text_augment', []))
            
            # Her görüntü için kayıt oluştur
            for i, img_path in enumerate(images):
                view = views[i] if i < len(views) else 'unknown'
                
                # Raporu al
                text_idx = min(i, len(texts) - 1) if texts else -1
                text_aug_idx = min(i, len(texts_aug) - 1) if texts_aug else -1
                
                if text_idx < 0:
                    continue
                    
                original_text = texts[text_idx]
                aug_text = texts_aug[text_aug_idx] if text_aug_idx >= 0 else None
                
                # Temizle
                cleaned_original = self.clean_report_text(original_text)
                cleaned_aug = self.clean_report_text(aug_text) if aug_text else None
                
                if not cleaned_original:
                    continue
                
                findings, impression, full_report = self.extract_sections(cleaned_original)
                
                records.append({
                    'subject_id': subject_id,
                    'study_id': f"{subject_id}_{i}",  # Unique study ID
                    'image_path': str(img_path),
                    'view': view,
                    'report': full_report,
                    'findings': findings,
                    'impression': impression,
                    'report_augmented': cleaned_aug,
                    'word_count': len(full_report.split()),
                    'has_augmentation': cleaned_aug is not None,
                    'original_split': self.split_name  # Hangi split'ten geldiğini kaydet
                })
        
        self.stats['initial_records'] = len(records)
        logger.info(f"İlk kayıt sayısı: {len(records)}")
        
        # 3. DataFrame oluştur
        clean_df = pd.DataFrame(records)
        
        # 4. Kalite filtreleme
        # Outlier'ları kaldır (çok kısa/uzun raporlar)
        q_low = clean_df['word_count'].quantile(0.005)
        q_high = clean_df['word_count'].quantile(0.995)
        clean_df = clean_df[(clean_df['word_count'] >= q_low) & 
                           (clean_df['word_count'] <= q_high)]
        
        self.stats['after_quality_filter'] = len(clean_df)
        logger.info(f"Kalite filtresi sonrası: {len(clean_df)}")
        
        # 5. Yinelenen görüntüleri kaldır
        clean_df = clean_df.drop_duplicates(subset=['image_path'], keep='first')
        
        # 6. Görüntü kontrolü (opsiyonel)
        # logger.info("Görüntüler kontrol ediliyor...")
        # valid_mask = clean_df['image_path'].apply(self.verify_image)
        # clean_df = clean_df[valid_mask]
        
        self.stats['final_records'] = len(clean_df)
        logger.info(f"Final kayıt sayısı: {len(clean_df)}")
        
        # 7. Kaydet
        self.save_and_analyze(clean_df)
        
        return clean_df
    
    def save_and_analyze(self, df):
        """Kaydet ve analiz et"""
        
        # CSV olarak kaydet
        output_csv = self.output_path / f"{self.split_name}_cleaned.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Kaydedildi: {output_csv}")
        
        # JSON olarak kaydet
        output_json = self.output_path / f"{self.split_name}_cleaned.json"
        df.to_json(output_json, orient='records', indent=2)
        
        # İstatistikleri kaydet
        stats_file = self.output_path / f"{self.split_name}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Özet yazdır
        logger.info(f"\n{self.split_name.upper()} ÖZET:")
        logger.info(f"  View dağılımı:\n{df['view'].value_counts()}")
        logger.info(f"  Ortalama kelime sayısı: {df['word_count'].mean():.1f}")
        logger.info(f"  Augmented rapor oranı: {df['has_augmentation'].mean()*100:.1f}%")


class DatasetMerger:
    """Train ve Val'i birleştirip final split oluştur"""
    
    def __init__(self, cleaned_path, output_path):
        self.cleaned_path = Path(cleaned_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_cleaned_data(self):
        """Temizlenmiş train ve val'i yükle"""
        train_df = pd.read_csv(self.cleaned_path / 'train_cleaned.csv')
        val_df = pd.read_csv(self.cleaned_path / 'val_cleaned.csv')
        
        logger.info(f"Train yüklendi: {len(train_df)} kayıt")
        logger.info(f"Val yüklendi: {len(val_df)} kayıt")
        
        return train_df, val_df
    
    def create_stratified_split(self, df, train_ratio=0.85, val_ratio=0.075, test_ratio=0.075, random_state=42):
        """
        Stratified split oluştur (rapor uzunluğuna göre)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Oranlar toplamı 1 olmalı"
        
        # Rapor uzunluğuna göre kategoriler
        df['length_cat'] = pd.qcut(df['word_count'], q=5, labels=['vshort', 'short', 'med', 'long', 'vlong'])
        
        # Önce train ve temp (val+test) ayır
        train, temp = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio),
            stratify=df['length_cat'],
            random_state=random_state
        )
        
        # Sonra val ve test ayır
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            temp,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp['length_cat'],
            random_state=random_state
        )
        
        # Kategorileri kaldır
        train = train.drop('length_cat', axis=1)
        val = val.drop('length_cat', axis=1)
        test = test.drop('length_cat', axis=1)
        
        return train, val, test
    
    def merge_and_create_final_split(self, use_original_split=True):
        """
        Train ve Val'i birleştir ve final split oluştur
        
        use_original_split=True: Orijinal train/val ayrımını koru
        use_original_split=False: Hepsini karıştır ve yeniden böl
        """
        train_df, val_df = self.load_cleaned_data()
        
        if use_original_split:
            logger.info("\nOrijinal train/val ayrımı korunuyor...")
            # Val'den bir kısmını test olarak ayır
            val, test = train_test_split(
                val_df, 
                test_size=0.5,  # Val'in yarısını test yap
                stratify=pd.qcut(val_df['word_count'], q=3, labels=['short', 'med', 'long']),
                random_state=42
            )
            
            final_train = train_df
            final_val = val
            final_test = test
            
        else:
            logger.info("\nTüm veri karıştırılıp yeniden bölünüyor...")
            # Hepsini birleştir
            combined = pd.concat([train_df, val_df], ignore_index=True)
            
            # Yinelenen görüntüleri kaldır
            combined = combined.drop_duplicates(subset=['image_path'], keep='first')
            
            # Stratified split
            final_train, final_val, final_test = self.create_stratified_split(combined)
        
        logger.info(f"\nFinal split:")
        logger.info(f"  Train: {len(final_train)}")
        logger.info(f"  Val: {len(final_val)}")
        logger.info(f"  Test: {len(final_test)}")
        
        # Kaydet
        splits = {
            'train': final_train,
            'val': final_val,
            'test': final_test
        }
        
        for name, split_df in splits.items():
            # CSV
            split_df.to_csv(self.output_path / f'final_{name}.csv', index=False)
            # JSON
            split_df.to_json(self.output_path / f'final_{name}.json', orient='records', indent=2)
            
            # İstatistikler
            stats = {
                'total_samples': len(split_df),
                'avg_word_count': split_df['word_count'].mean(),
                'view_distribution': split_df['view'].value_counts().to_dict(),
                'augmented_ratio': split_df['has_augmentation'].mean()
            }
            
            with open(self.output_path / f'{name}_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
        
        # Data leakage kontrolü
        self.check_leakage(final_train, final_val, final_test)
        
        return splits
    
    def check_leakage(self, train, val, test):
        """Data leakage kontrolü"""
        train_images = set(train['image_path'])
        val_images = set(val['image_path'])
        test_images = set(test['image_path'])
        
        train_val_overlap = train_images & val_images
        train_test_overlap = train_images & test_images
        val_test_overlap = val_images & test_images
        
        logger.info(f"\nData Leakage Kontrolü:")
        logger.info(f"  Train-Val overlap: {len(train_val_overlap)}")
        logger.info(f"  Train-Test overlap: {len(train_test_overlap)}")
        logger.info(f"  Val-Test overlap: {len(val_test_overlap)}")
        
        if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
            logger.warning("⚠️ UYARI: Data leakage tespit edildi!")
        else:
            logger.info("✅ Data leakage yok.")


# ============================================================
# ANA ÇALIŞTIRMA KODU
# ============================================================

def main():
    # PATH'LERİNİZİ BURAYA GİRİN
    PATHS = {
    'train_csv': r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\mimic_cxr_aug_train.csv',
    'val_csv': r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\mimic_cxr_aug_validate.csv',
    'image_root': r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\official_data_iccv_final\files',
    'cleaned_output': './cleaned',
    'final_output': './final'
    }
    
    # =========================================================
    # ADIM 1: TRAIN TEMİZLE
    # =========================================================
    logger.info("\n" + "="*70)
    logger.info("ADIM 1: TRAIN SET TEMİZLENİYOR")
    logger.info("="*70)
    
    train_cleaner = MimicCXRCleaner(
        csv_path=PATHS['train_csv'],
        image_root=PATHS['image_root'],
        output_path=PATHS['cleaned_output'],
        split_name='train'
    )
    train_cleaned = train_cleaner.process_dataset()
    
    # =========================================================
    # ADIM 2: VAL TEMİZLE
    # =========================================================
    logger.info("\n" + "="*70)
    logger.info("ADIM 2: VAL SET TEMİZLENİYOR")
    logger.info("="*70)
    
    val_cleaner = MimicCXRCleaner(
        csv_path=PATHS['val_csv'],
        image_root=PATHS['image_root'],
        output_path=PATHS['cleaned_output'],
        split_name='val'
    )
    val_cleaned = val_cleaner.process_dataset()
    
    # =========================================================
    # ADIM 3: BİRLEŞTİR VE FİNAL SPLIT OLUŞTUR
    # =========================================================
    logger.info("\n" + "="*70)
    logger.info("ADIM 3: FİNAL SPLIT OLUŞTURULUYOR")
    logger.info("="*70)
    
    merger = DatasetMerger(
        cleaned_path=PATHS['cleaned_output'],
        output_path=PATHS['final_output']
    )
    
    # use_original_split=True: Orijinal train/val ayrımını koru, val'in yarısını test yap
    # use_original_split=False: Hepsini karıştır ve %85/7.5/7.5 böl
    final_splits = merger.merge_and_create_final_split(use_original_split=True)
    
    # =========================================================
    # ÖZET
    # =========================================================
    logger.info("\n" + "="*70)
    logger.info("TÜM İŞLEMLER TAMAMLANDI!")
    logger.info("="*70)
    
    for split_name, split_df in final_splits.items():
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Kayıt sayısı: {len(split_df)}")
        logger.info(f"  Ortalama kelime: {split_df['word_count'].mean():.1f}")
        logger.info(f"  Augmented oranı: {split_df['has_augmentation'].mean()*100:.1f}%")
        logger.info(f"  View dağılımı: {dict(split_df['view'].value_counts())}")


if __name__ == "__main__":
    main()