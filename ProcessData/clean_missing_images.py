# clean_missing_images_fixed.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

def remove_missing_images(csv_path, image_root, output_path):
    """
    Dosyası bulunmayan kayıtları kaldır
    """
    print(f"\n🧹 Temizleniyor: {csv_path}")
    
    df = pd.read_csv(csv_path)
    image_root = Path(image_root)
    
    initial_count = len(df)
    print(f"   Başlangıç: {initial_count:,} kayıt")
    print(f"   Kolonlar: {df.columns.tolist()}")  # Kolonları göster
    
    # Var olan dosyaları filtrele
    valid_mask = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Kontrol"):
        img_path = row['image_path']
        full_path = image_root / img_path
        valid_mask.append(full_path.exists())
    
    # Filtrelenmiş dataframe
    df_clean = df[valid_mask].copy()
    
    removed_count = initial_count - len(df_clean)
    print(f"   ❌ Kaldırılan: {removed_count:,}")
    print(f"   ✅ Kalan: {len(df_clean):,}")
    print(f"   📉 Kayıp oranı: {removed_count/initial_count*100:.2f}%")
    
    # Kaydet
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / Path(csv_path).name
    df_clean.to_csv(output_file, index=False)
    print(f"   💾 Kaydedildi: {output_file}")
    
    # İstatistikler - sadece var olan kolonları kullan
    stats = {
        'initial_count': initial_count,
        'removed_count': removed_count,
        'final_count': len(df_clean),
        'loss_rate': removed_count/initial_count*100,
    }
    
    # Opsiyonel kolonlar - varsa ekle
    if 'word_count' in df_clean.columns:
        stats['avg_word_count'] = float(df_clean['word_count'].mean())
    
    if 'view' in df_clean.columns:
        stats['view_distribution'] = df_clean['view'].value_counts().to_dict()
    
    if 'report' in df_clean.columns:
        # Rapor uzunluğunu hesapla
        df_clean['report_length'] = df_clean['report'].fillna('').astype(str).apply(len)
        stats['avg_report_length'] = float(df_clean['report_length'].mean())
    
    return df_clean, stats

if __name__ == "__main__":
    IMAGE_ROOT = r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\official_data_iccv_final'
    INPUT_DIR = './Datasets/processed'
    OUTPUT_DIR = './Datasets/cleaned_final'
    
    all_stats = {}
    
    print("\n" + "="*70)
    print("🚀 EKSİK GÖRÜNTÜ TEMİZLEME BAŞLIYOR")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        csv_path = f'{INPUT_DIR}/{split}.csv'
        
        try:
            df_clean, stats = remove_missing_images(csv_path, IMAGE_ROOT, OUTPUT_DIR)
            all_stats[split] = stats
        except FileNotFoundError:
            print(f"   ⚠️ {csv_path} bulunamadı, atlanıyor...")
            continue
    
    # Genel özet
    print("\n" + "="*70)
    print("📊 GENEL ÖZET")
    print("="*70)
    
    total_initial = sum(s['initial_count'] for s in all_stats.values())
    total_final = sum(s['final_count'] for s in all_stats.values())
    total_removed = sum(s['removed_count'] for s in all_stats.values())
    
    for split, stats in all_stats.items():
        print(f"\n{split.upper()}:")
        print(f"   Başlangıç: {stats['initial_count']:,}")
        print(f"   Final: {stats['final_count']:,}")
        print(f"   Kaldırılan: {stats['removed_count']:,}")
        print(f"   Kayıp: {stats['loss_rate']:.2f}%")
    
    print(f"\n{'='*70}")
    print("TOPLAM:")
    print(f"   Başlangıç: {total_initial:,}")
    print(f"   Final: {total_final:,}")
    print(f"   Kaldırılan: {total_removed:,}")
    print(f"   Genel Kayıp: {(total_removed/total_initial)*100:.2f}%")
    print("="*70)
    
    # Stats'ları kaydet
    output_stats_file = f'{OUTPUT_DIR}/cleaning_stats.json'
    with open(output_stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Tüm işlemler tamamlandı!")
    print(f"📁 Temiz dosyalar: {OUTPUT_DIR}")
    print(f"📊 İstatistikler: {output_stats_file}")