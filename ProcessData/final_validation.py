# final_validation.py
import pandas as pd
from pathlib import Path

def validate_cleaned_data():
    """
    Temizlenmiş veriyi son kez doğrula
    """
    OUTPUT_DIR = r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed'
    
    print("="*70)
    print("🔍 FİNAL VERİ SETİ DOĞRULAMA")
    print("="*70)
    
    for split in ['train_processed', 'val_processed', 'test_processed']:
        csv_path = f'{OUTPUT_DIR}/{split}.csv'
        df = pd.read_csv(csv_path)
        
        print(f"\n📁 {split.upper()}:")
        print(f"   Kayıt sayısı: {len(df):,}")
        print(f"   Kolonlar: {df.columns.tolist()}")
        
        # Boş değer kontrolü
        print(f"\n   Boş değerler:")
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Örnek veri
        print(f"\n   İlk kayıt örneği:")
        print(f"      image_path: {df.iloc[0]['image_path']}")
        if 'report' in df.columns:
            report_preview = str(df.iloc[0]['report'])[:100]
            print(f"      report: {report_preview}...")
    
    print("\n" + "="*70)
    print("✅ Doğrulama tamamlandı!")
    print("="*70)

if __name__ == "__main__":
    validate_cleaned_data()