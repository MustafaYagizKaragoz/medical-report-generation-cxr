# prepare_for_training.py

import pandas as pd
from pathlib import Path

def prepare_training_format(input_path='./final_resplit', output_path='./Datasets/processed'):
    """
    Dataset.py için basit format: image_path, caption
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRAINING FORMAT HAZIRLANYOR")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 {split.upper()} işleniyor...")
        
        # Yükle
        df = pd.read_csv(input_path / f'final_{split}.csv')
        print(f"   Yüklendi: {len(df):,} kayıt")
        
        # Sadece gerekli kolonlar: image_path, caption
        final_df = pd.DataFrame({
            'image_path': df['image_path'],
            'caption': df['report']  # report -> caption olarak rename
        })
        
        # Null kontrolü
        before = len(final_df)
        final_df = final_df.dropna()
        after = len(final_df)
        
        if before != after:
            print(f"   ⚠️ {before - after} null kayıt kaldırıldı")
        
        # Kaydet
        output_file = output_path / f'{split}.csv'
        final_df.to_csv(output_file, index=False)
        
        print(f"   ✅ Kaydedildi: {output_file}")
        print(f"   Kayıt: {len(final_df):,}")
        print(f"   Columns: {final_df.columns.tolist()}")
        
        # Örnek göster
        print(f"   Örnek:")
        print(f"      Path: {final_df['image_path'].iloc[0][:60]}...")
        print(f"      Caption: {final_df['caption'].iloc[0][:80]}...")
    
    print("\n" + "="*70)
    print("✅ TÜM FORMATLAR HAZIR!")
    print("="*70)


if __name__ == "__main__":
    prepare_training_format()