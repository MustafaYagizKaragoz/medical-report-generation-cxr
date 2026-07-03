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
    """
    caption = str(caption)
    
    findings = ""
    impression = ""
    
    # Case-insensitive split on 'impression:'
    if re.search(r'impression:', caption, re.IGNORECASE):
        parts = re.split(r'impression:', caption, maxsplit=1, flags=re.IGNORECASE)
        findings_part = parts[0]
        impression_part = parts[1]
        
        # Case-insensitive strip of 'findings:'
        findings = re.sub(r'findings:', '', findings_part, flags=re.IGNORECASE).strip()
        impression = impression_part.strip()
    else:
        findings = re.sub(r'findings:', '', caption, flags=re.IGNORECASE).strip()
        
    clean_f = clean_text(findings)
    clean_i = clean_text(impression)
    
    # Combined Findings & Impression
    combined_text = f"{clean_f} {clean_i}".strip()
    
    # Strip any remaining 'findings:' or 'impression:' headers case-insensitively
    combined_text = re.sub(r'\b(findings|impression)\s*:\s*', '', combined_text, flags=re.IGNORECASE)
    
    return combined_text.strip()

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