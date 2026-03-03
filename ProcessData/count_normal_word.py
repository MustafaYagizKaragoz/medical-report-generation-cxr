import pandas as pd
import re
from collections import Counter
import os

# Veri seti yolu (Kendi klasör yapına göre ayarlandı)
csv_path = r"C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed\train_chunked.csv"

print(f"⏳ Veri seti yükleniyor: {csv_path}")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"❌ Dosya bulunamadı! Yolun doğru olduğundan emin ol: {csv_path}")
    exit()

# Sütun adını kontrol et (Genelde 'caption_text' veya 'caption' olur)
text_column = 'final_report' 
if text_column not in df.columns:
    if 'caption' in df.columns:
        text_column = 'caption'
    else:
        print(f"❌ Metin kolonu bulunamadı! Mevcut kolonlar: {df.columns.tolist()}")
        exit()

print(f"✅ Veri seti yüklendi. Toplam parça (chunk): {len(df):,}")

# Sadece string olanları al ve küçük harfe çevir
texts = df[text_column].dropna().astype(str).str.lower()

# İstatistikler için sayaçlar
word_counts = Counter()
total_words = 0
normal_sentence_count = 0

print("🔍 Kelimeler analiz ediliyor, lütfen bekleyin...")

for text in texts:
    # Sadece harfleri al (noktalama işaretlerini boşluk yapıp kelimeleri ayırır)
    words = re.findall(r'\b[a-z]+\b', text)
    word_counts.update(words)
    total_words += len(words)
    
    # İçinde 'normal' geçen chunk sayısı
    if 'normal' in words:
        normal_sentence_count += 1

# Sayılmasını istediğimiz spesifik (güvenli vs riskli) kelimeler
target_words = ['normal', 'no', 'clear', 'unremarkable', 'pneumonia', 'effusion', 'cardiomegaly']

print("\n" + "="*60)
print("📊 EĞİTİM VERİ SETİ KELİME DAĞILIMI (CLASS IMBALANCE ANALİZİ)")
print("="*60)
print(f"Toplam Cümle (Chunk) Sayısı : {len(texts):,}")
print(f"Toplam Kelime Sayısı        : {total_words:,}")
print("-" * 60)

for word in target_words:
    count = word_counts[word]
    percentage = (count / total_words) * 100 if total_words > 0 else 0
    print(f"'{word:<12}' kelimesi : {count:>7,} kez geçiyor. (Tüm kelimelerin %{percentage:.2f}'si)")

print("-" * 60)
chunk_percentage = (normal_sentence_count / len(texts)) * 100
print(f"İçinde 'normal' geçen cümle/chunk sayısı: {normal_sentence_count:,} (Tüm veri setinin %{chunk_percentage:.2f}'si)")
print("="*60)

# En çok geçen 10 kelimeyi de görelim
print("\n🏆 VERİ SETİNDE EN ÇOK GEÇEN 10 KELİME:")
for word, count in word_counts.most_common(10):
    print(f" - {word:<10}: {count:,}")