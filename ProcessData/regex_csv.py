import pandas as pd
import re
import os

# DOSYA İSİMLERİNİ BURAYA YAZ (Eğer farklıysa değiştir)
TRAIN_FILE = "train.csv"
VAL_FILE   = "val.csv"
TEST_FILE  = "test.csv"

# ---------------------------------------------------------
# 1. TEMİZLEME FONKSİYONLARI (Regex ve Parsing)
# ---------------------------------------------------------

def clean_text(text):
    """
    Metni temizler: Tarihleri, kıyaslama cümlelerini ve gereksiz boşlukları atar.
    Noktalama işaretlerini (.) ve (,) KORUR.
    """
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # İdari ve gereksiz notları sil
    text = re.sub(r'comment:.*', '', text)
    text = re.sub(r'signed by.*', '', text)
    text = re.sub(r'dr\.\s+[a-z]+', '', text) 

    # "Öncekiyle kıyasla" (Comparison) cümlelerini temizle
    comparison_patterns = [
        r'comparison is made with.*?(?=\.)',
        r'as compared to.*?(?=\.)',
        r'compared with.*?(?=\.)',
        r'comparison to.*?(?=\.)',
        r'in comparison with.*?(?=\.)',
        r'prior exam.*?(?=\.)',
    ]
    for pattern in comparison_patterns:
        text = re.sub(pattern, '', text)

    # Yeni satırları boşluğa çevir
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def parse_caption(caption):
    """
    Caption sütununu 'Findings' ve 'Impression' olarak analiz eder.
    YENİ STRATEJİ: Öncelik 'Findings' (Bulgular) kısmındadır.
    """
    caption = str(caption)
    caption_lower = caption.lower()
    
    findings = ""
    impression = ""
    
    # 'impression:' etiketi varsa oradan böl
    if 'impression:' in caption_lower:
        parts = caption_lower.split('impression:')
        findings_part = parts[0]
        impression_part = parts[1]
        
        findings = findings_part.replace('findings:', '').strip()
        impression = impression_part.strip()
    else:
        # Etiket yoksa tamamını Findings olarak kabul et
        findings = caption_lower.replace('findings:', '').strip()
    
    clean_f = clean_text(findings)
    clean_i = clean_text(impression)
    
    # YENİ STRATEJİ: Makaledeki gibi "Findings" (Bulgular) odaklı 
    # İstersen ikisini birleştirerek Transformer'a maksimum bağlam verebilirsin:
    # return f"{clean_f} {clean_i}".strip()
    
    # Eğer sadece Findings istiyorsan (Makale Standardı):
    if clean_f and len(clean_f) > 5:
        return clean_f
    # Eğer doktor Findings yazmamış, sadece Impression yazmışsa veriyi kaybetmemek için onu al:
    elif clean_i and len(clean_i) > 2:
        return clean_i
    else:
        return "" # Boş rapor

# ---------------------------------------------------------
# 2. İŞLEM FONKSİYONU
# ---------------------------------------------------------

def process_dataset(filename, is_train=False):
    print(f"\nİşleniyor: {filename} ...")
    
    if not os.path.exists(filename):
        print(f"HATA: {filename} bulunamadı!")
        return None

    df = pd.read_csv(filename)
    
    # 1. Metin Temizliği
    df['final_report'] = df['caption'].apply(parse_caption)
    
    # Boş raporları sil
    initial_len = len(df)
    df = df[df['final_report'] != ""]
    print(f"   Boş satırlar silindi: {initial_len} -> {len(df)}")
    
    # 2. Deduplication (Sadece TRAIN seti için)
    if is_train:
        print("   TRAIN seti tespit edildi: Tekrarlar azaltılıyor (Deduplication)...")
        before_dedup = len(df)
        # Aynı rapora sahip en fazla 5 görüntü tut
        df = df.groupby('final_report').head(5).reset_index(drop=True)
        print(f"   Tekrar temizliği: {before_dedup} -> {len(df)} (Atılan: {before_dedup - len(df)})")
    else:
        print("   Test/Val seti: Deduplication yapılmadı (Dağılım korundu).")
        
    # Dosyayı kaydet
    output_name = filename.replace('.csv', '_processed.csv')
    df.to_csv(output_name, index=False)
    print(f"   Kaydedildi: {output_name}")

# ---------------------------------------------------------
# 3. ÇALIŞTIRMA
# ---------------------------------------------------------

# Train dosyasını işle (Deduplication VAR)
process_dataset(TRAIN_FILE, is_train=True)

# Val dosyasını işle (Deduplication YOK)
process_dataset(VAL_FILE, is_train=False)

# Test dosyasını işle (Deduplication YOK)
process_dataset(TEST_FILE, is_train=False)

print("\n--- TÜM İŞLEMLER TAMAMLANDI ---")