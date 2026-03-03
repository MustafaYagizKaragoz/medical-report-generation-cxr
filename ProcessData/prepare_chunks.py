import pandas as pd
import re

def create_chunked_dataset(input_csv, output_csv):
    print(f"⏳ {input_csv} okunuyor...")
    df = pd.read_csv(input_csv)
    new_rows = []

    for index, row in df.iterrows():
        img_path = row['image_path']
        report = str(row['final_report']).lower() # Her şeyi küçük harfe çevir
        
        # 1. Raporu noktalardan böl
        sentences = report.split('.')
        
        for sentence in sentences:
            # 2. Baştaki ve sondaki boşlukları temizle
            clean_sentence = sentence.strip()
            
            # 3. "1.", "2." gibi numaralandırmaları ve gereksiz karakterleri temizle
            # Sadece harfleri ve boşlukları tut (Makaledeki taktik)
            clean_sentence = re.sub(r'[^a-z\s]', '', clean_sentence)
            clean_sentence = clean_sentence.strip()
            
            # 4. Sadece mantıklı uzunluktaki cümleleri al (örn: 2 kelimeden büyükse)
            if len(clean_sentence.split()) >= 2:
                new_rows.append({
                    'image_path': img_path,
                    'final_report': clean_sentence + ' .' # Cümle sonu noktası ekle (EOS için)
                })

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)
    
    print(f"✅ İşlem Tamamlandı!")
    print(f"📉 Orijinal Rapor Sayısı: {len(df)}")
    print(f"📈 Parçalanmış (Chunked) Cümle Sayısı: {len(new_df)}")

# Train, Val ve Test için ayrı ayrı çalıştır
if __name__ == "__main__":
    # Kendi dosya yollarını buraya yaz
    create_chunked_dataset('train_processed.csv', 'train_chunked.csv')
    create_chunked_dataset('val_processed.csv', 'val_chunked.csv')
    create_chunked_dataset('test_processed.csv', 'test_chunked.csv')