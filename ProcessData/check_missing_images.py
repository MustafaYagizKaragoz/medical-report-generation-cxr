# check_missing_images.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def find_missing_images(csv_path, image_root):
    """
    CSV'deki tüm görüntüleri kontrol et ve eksik olanları bul
    """
    print("🔍 Eksik görüntüler taranıyor...")
    
    df = pd.read_csv(csv_path)
    image_root = Path(image_root)
    
    missing = []
    valid = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Kontrol ediliyor"):
        img_path = row['image_path']
        full_path = image_root / img_path
        
        if full_path.exists():
            valid.append(idx)
        else:
            missing.append(idx)
    
    print(f"\n📊 SONUÇ:")
    print(f"   ✅ Var olan: {len(valid):,} ({len(valid)/len(df)*100:.2f}%)")
    print(f"   ❌ Eksik: {len(missing):,} ({len(missing)/len(df)*100:.2f}%)")
    
    return valid, missing

if __name__ == "__main__":
    IMAGE_ROOT = r"C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\OriginalData\official_data_iccv_final"
    
    # Her split için kontrol et
    for split in ['train_processed', 'val_processed', 'test_processed']:
        csv_path = f'../Data/processed/{split}.csv'
        print(f"\n{'='*70}")
        print(f"📁 {split.upper()} SET")
        print('='*70)
        
        valid, missing = find_missing_images(csv_path, IMAGE_ROOT)
        
        # Eksik dosya listesini kaydet
        if missing:
            with open(f'missing_{split}.txt', 'w') as f:
                f.write('\n'.join(map(str, missing)))
            print(f"   💾 Eksik indeksler kaydedildi: missing_{split}.txt")