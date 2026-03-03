# inspect_captions.py
import pandas as pd
import numpy as np

def analyze_captions():
    """
    Caption'ları detaylı analiz et
    """
    print("="*70)
    print("📝 CAPTION ANALİZİ")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(f'../Datasets/cleaned_final/{split}.csv')
        
        print(f"\n{'='*70}")
        print(f"📁 {split.upper()}")
        print('='*70)
        
        # Caption uzunlukları
        df['caption_length'] = df['caption'].astype(str).apply(len)
        df['word_count'] = df['caption'].astype(str).apply(lambda x: len(x.split()))
        
        print(f"\n📊 İstatistikler:")
        print(f"   Toplam kayıt: {len(df):,}")
        print(f"   Ortalama karakter: {df['caption_length'].mean():.0f}")
        print(f"   Ortalama kelime: {df['word_count'].mean():.0f}")
        print(f"   Min kelime: {df['word_count'].min()}")
        print(f"   Max kelime: {df['word_count'].max()}")
        print(f"   Medyan kelime: {df['word_count'].median():.0f}")
        
        # Boş caption kontrolü
        empty_captions = df['caption'].isna().sum()
        if empty_captions > 0:
            print(f"   ⚠️ Boş caption: {empty_captions}")
        
        # Örnek caption'lar
        print(f"\n📄 Örnek Caption'lar:")
        for i in range(min(3, len(df))):
            caption = df.iloc[i]['caption']
            print(f"\n   [{i+1}] ({len(str(caption).split())} kelime)")
            print(f"   {str(caption)[:300]}...")
        
        # Uzunluk dağılımı
        print(f"\n📈 Kelime Sayısı Dağılımı:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = df['word_count'].quantile(p/100)
            print(f"   {p}th percentile: {val:.0f} kelime")

if __name__ == "__main__":
    analyze_captions()