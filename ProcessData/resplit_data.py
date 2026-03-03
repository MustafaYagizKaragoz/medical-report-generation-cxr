# resplit_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resplit_dataset(cleaned_path, output_path, train_ratio=0.85, val_ratio=0.075, test_ratio=0.075):
    """
    Temizlenmiş veriyi yeniden böl (85/7.5/7.5)
    """
    
    logger.info("📁 Temizlenmiş veri yükleniyor...")
    
    # Temizlenmiş train ve val'i yükle
    train_df = pd.read_csv(Path(cleaned_path) / 'train_cleaned.csv')
    val_df = pd.read_csv(Path(cleaned_path) / 'val_cleaned.csv')
    
    logger.info(f"Train yüklendi: {len(train_df):,}")
    logger.info(f"Val yüklendi: {len(val_df):,}")
    
    # Hepsini birleştir
    combined = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"Birleştirildi: {len(combined):,}")
    
    # Yinelenen image_path'leri kaldır
    combined = combined.drop_duplicates(subset=['image_path'], keep='first')
    logger.info(f"Duplicate kaldırıldı: {len(combined):,}")
    
    # Rapor uzunluğuna göre kategori oluştur (stratified split için)
    combined['length_cat'] = pd.qcut(
        combined['word_count'], 
        q=5, 
        labels=['very_short', 'short', 'medium', 'long', 'very_long'],
        duplicates='drop'
    )
    
    logger.info("\n🔀 Stratified split yapılıyor...")
    
    # Önce train ve temp (val+test) ayır
    train, temp = train_test_split(
        combined,
        test_size=(val_ratio + test_ratio),
        stratify=combined['length_cat'],
        random_state=42
    )
    
    # Sonra val ve test ayır
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        temp,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp['length_cat'],
        random_state=42
    )
    
    # Kategori kolonunu kaldır
    train = train.drop('length_cat', axis=1)
    val = val.drop('length_cat', axis=1)
    test = test.drop('length_cat', axis=1)
    
    logger.info(f"\n✅ Final split:")
    logger.info(f"  Train: {len(train):,} ({len(train)/len(combined)*100:.1f}%)")
    logger.info(f"  Val:   {len(val):,} ({len(val)/len(combined)*100:.1f}%)")
    logger.info(f"  Test:  {len(test):,} ({len(test)/len(combined)*100:.1f}%)")
    
    # Data leakage kontrolü
    logger.info("\n🔍 Data leakage kontrolü...")
    train_imgs = set(train['image_path'])
    val_imgs = set(val['image_path'])
    test_imgs = set(test['image_path'])
    
    train_val_overlap = len(train_imgs & val_imgs)
    train_test_overlap = len(train_imgs & test_imgs)
    val_test_overlap = len(val_imgs & test_imgs)
    
    logger.info(f"  Train-Val overlap: {train_val_overlap}")
    logger.info(f"  Train-Test overlap: {train_test_overlap}")
    logger.info(f"  Val-Test overlap: {val_test_overlap}")
    
    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        logger.error("❌ UYARI: Data leakage tespit edildi!")
    else:
        logger.info("✅ Data leakage yok!")
    
    # Kaydet
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Kaydediliyor: {output_path}")
    
    # CSV olarak kaydet
    train.to_csv(output_path / 'final_train.csv', index=False)
    val.to_csv(output_path / 'final_val.csv', index=False)
    test.to_csv(output_path / 'final_test.csv', index=False)
    
    # JSON olarak da kaydet
    train.to_json(output_path / 'final_train.json', orient='records', indent=2)
    val.to_json(output_path / 'final_val.json', orient='records', indent=2)
    test.to_json(output_path / 'final_test.json', orient='records', indent=2)
    
    # İstatistikler
    for name, df in [('train', train), ('val', val), ('test', test)]:
        stats = {
            'total_samples': len(df),
            'avg_word_count': float(df['word_count'].mean()),
            'min_word_count': int(df['word_count'].min()),
            'max_word_count': int(df['word_count'].max()),
            'view_distribution': df['view'].value_counts().to_dict(),
            'augmented_ratio': float(df['has_augmentation'].mean())
        }
        
        import json
        with open(output_path / f'{name}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    logger.info("\n✅ İşlem tamamlandı!")
    logger.info(f"Dosyalar: {output_path}")
    
    # Özet yazdır
    logger.info("\n" + "="*70)
    logger.info("FİNAL ÖZET")
    logger.info("="*70)
    
    for name, df in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
        logger.info(f"\n{name}:")
        logger.info(f"  Kayıt: {len(df):,}")
        logger.info(f"  Oran: {len(df)/len(combined)*100:.1f}%")
        logger.info(f"  Ortalama kelime: {df['word_count'].mean():.1f}")
        logger.info(f"  Min-Max kelime: {df['word_count'].min()}-{df['word_count'].max()}")
    
    return train, val, test


if __name__ == "__main__":
    # PATH'LERİ BURAYA GİR
    CLEANED_PATH = './cleaned'      # Temizlenmiş veri
    OUTPUT_PATH = './final_resplit' # Yeni split
    
    # Yeniden böl (85/7.5/7.5)
    train, val, test = resplit_dataset(
        cleaned_path=CLEANED_PATH,
        output_path=OUTPUT_PATH,
        train_ratio=0.85,
        val_ratio=0.075,
        test_ratio=0.075
    )
    
    print("\n✅ Başarıyla tamamlandı!")
    print(f"📁 Yeni dosyalar: {OUTPUT_PATH}")   