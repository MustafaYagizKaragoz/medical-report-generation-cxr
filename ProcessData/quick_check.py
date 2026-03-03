# quick_check.py

import pandas as pd

# CSV'leri yükle
train = pd.read_csv('final_resplit/final_train.csv')
val = pd.read_csv('final_resplit/final_val.csv')
test = pd.read_csv('final_resplit/final_test.csv')

print("="*70)
print("DATASET FORMAT KONTROLÜ")
print("="*70)

print(f"\nTrain shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")

print("\nİlk 2 satır:")
print(train[['image_path', 'report', 'word_count']].head(2))

print("\nSample report:")
print(train['report'].iloc[0][:200] + "...")