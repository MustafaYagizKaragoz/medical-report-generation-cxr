# test_images.py

import pandas as pd
from pathlib import Path
from PIL import Image

def test_image_paths(csv_path='../Datasets/cleaned_final/train.csv', 
                     image_root=r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\OriginalData\official_data_iccv_final'):
    """
    İlk 10 görüntüyü test et
    """
    
    print("="*70)
    print("🖼️ IMAGE PATH TESTİ")
    print("="*70)
    
    image_root = Path(image_root)
    df = pd.read_csv(csv_path)
    
    print(f"\nCSV: {csv_path}")
    print(f"Image Root: {image_root}")
    print(f"Toplam kayıt: {len(df):,}\n")
    
    success = 0
    failed = 0
    
    for idx in range(min(10, len(df))):
        img_path = df.iloc[idx]['image_path']
        full_path = image_root / img_path
        
        print(f"{idx+1}. Görüntü:")
        print(f"   Relative: {img_path}")
        
        if full_path.exists():
            try:
                with Image.open(full_path) as img:
                    print(f"   ✅ OK - Size: {img.size}, Mode: {img.mode}")
                    success += 1
            except Exception as e:
                print(f"   ❌ Error açılırken: {e}")
                failed += 1
        else:
            print(f"   ❌ Dosya bulunamadı!")
            print(f"   Denenen path: {full_path}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"SONUÇ: ✅ {success}/10 başarılı, ❌ {failed}/10 hata")
    print("="*70)
    
    if failed > 0:
        print("\n⚠️ UYARI: Bazı görüntüler bulunamadı!")
        print("IMAGE_ROOT path'ini kontrol et!")
    else:
        print("\n🎉 Tüm görüntüler erişilebilir!")


if __name__ == "__main__":
    test_image_paths()