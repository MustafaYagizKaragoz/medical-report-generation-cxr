# dataset_cnn_lstm.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pickle
from collections import Counter
from tqdm import tqdm
import json
from .vocabulary import Vocabulary



# ============================================================================
# DATASET
# ============================================================================

class MIMICCXRDatasetCNNLSTM(Dataset):
    """
    MIMIC-CXR Dataset for CNN-LSTM model
    """
    
    def __init__(
        self, 
        csv_path, 
        image_root, 
        vocabulary,
        transform=None,
        max_length=30
    ):
        """
        Args:
            csv_path: CSV dosya yolu
            image_root: Görüntü kök dizini
            vocabulary: Vocabulary nesnesi
            transform: Görüntü transformları
            max_length: Maksimum caption uzunluğu
        """
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_length = max_length
        
        # Boş caption'ları temizle
        self.df = self.df.dropna(subset=['final_report'])
        self.df = self.df[self.df['final_report'].astype(str).str.strip() != '']
        
        print(f"✅ {len(self.df):,} kayıt yüklendi: {Path(csv_path).name}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Görüntüyü yükle
        img_path = self.image_root / row['image_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Görüntü yüklenemedi: {img_path}")
            # Siyah görüntü döndür
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Caption'ı encode et
        caption = str(row['final_report']).strip()
        caption_encoded = self.vocabulary.encode(caption, max_length=self.max_length)
        caption_length = len(caption_encoded)
        
        return {
            'image': image,
            'caption': torch.LongTensor(caption_encoded),
            'caption_length': caption_length,
            'caption_text': caption,
            'image_path': row['image_path']
        }


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function - farklı uzunluktaki caption'ları batch'le
    
    Args:
        batch: List of samples from dataset
    Returns:
        Dictionary with batched data
    """
    # Sort batch by caption length (descending) - LSTM için gerekli
    batch.sort(key=lambda x: x['caption_length'], reverse=True)
    
    # Batch elemanlarını ayır
    images = torch.stack([item['image'] for item in batch])
    caption_lengths = torch.LongTensor([item['caption_length'] for item in batch])
    
    # Caption'ları pad'le
    max_length = max([item['caption_length'] for item in batch])
    padded_captions = torch.zeros(len(batch), max_length, dtype=torch.long)
    
    for i, item in enumerate(batch):
        caption = item['caption']
        padded_captions[i, :len(caption)] = caption
    
    caption_texts = [item['caption_text'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'captions': padded_captions,
        'caption_lengths': caption_lengths,
        'caption_texts': caption_texts,
        'image_paths': image_paths
    }


# ============================================================================
# DATALOADER CREATOR
# ============================================================================

def create_vocabulary_and_dataloaders(
    data_dir=r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed',
    image_root=r'C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\OriginalData\official_data_iccv_final',
    vocab_path='./vocabulary.pkl',
    batch_size=32,
    num_workers=0,
    max_length=150,
    freq_threshold=5,
    build_vocab=False
):
    """
    Vocabulary oluştur ve DataLoader'ları hazırla
    
    Args:
        data_dir: Temiz veri dizini
        image_root: Görüntü kök dizini
        vocab_path: Vocabulary kayıt yolu
        batch_size: Batch boyutu
        num_workers: DataLoader worker sayısı
        max_length: Maksimum caption uzunluğu
        freq_threshold: Minimum kelime frekansı
        build_vocab: Vocabulary'yi sıfırdan oluştur (False ise yükle)
    
    Returns:
        train_loader, val_loader, test_loader, vocabulary
    """
    from torchvision import transforms
    
    print("\n" + "="*70)
    print("📚 VOCABULARY VE DATALOADER HAZIRLANIYOR")
    print("="*70)
    
    # ========================================
    # 1. VOCABULARY OLUŞTUR/YÜKLE
    # ========================================
    
    vocabulary = Vocabulary(freq_threshold=freq_threshold)
    
    if build_vocab:
        # Train caption'larını yükle
        train_df = pd.read_csv(f'{data_dir}/train_processed.csv')
        train_captions = train_df['final_report'].dropna().tolist()
        
        # Vocabulary oluştur
        vocabulary.build_vocabulary(train_captions)
        
        # Kaydet
        vocabulary.save(vocab_path)
    else:
        # Mevcut vocabulary'yi yükle
        vocabulary.load(vocab_path)
    
    # ========================================
    # 2. IMAGE TRANSFORMS
    # ========================================
    
    # Training transforms (data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ========================================
    # 3. DATASETS
    # ========================================
    
    print(f"\n📦 Dataset'ler oluşturuluyor...")
    
    train_dataset = MIMICCXRDatasetCNNLSTM(
        csv_path=f'{data_dir}/train_processed.csv',
        image_root=image_root,
        vocabulary=vocabulary,
        transform=train_transform,
        max_length=max_length
    )
    
    val_dataset = MIMICCXRDatasetCNNLSTM(
        csv_path=f'{data_dir}/val_processed.csv',
        image_root=image_root,
        vocabulary=vocabulary,
        transform=val_transform,
        max_length=max_length
    )
    
    test_dataset = MIMICCXRDatasetCNNLSTM(
        csv_path=f'{data_dir}/test_processed.csv',
        image_root=image_root,
        vocabulary=vocabulary,
        transform=val_transform,
        max_length=max_length
    )
    
    # ========================================
    # 4. DATALOADERS
    # ========================================
    
    print(f"\n🔄 DataLoader'lar oluşturuluyor...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # ========================================
    # 5. ÖZET
    # ========================================
    
    print("\n" + "="*70)
    print("✅ HAZIR!")
    print("="*70)
    print(f"Vocabulary boyutu: {len(vocabulary):,}")
    print(f"Max caption uzunluğu: {max_length}")
    print(f"Batch size: {batch_size}")
    print(f"\nDataset boyutları:")
    print(f"  Train: {len(train_dataset):,} ({len(train_loader):,} batches)")
    print(f"  Val:   {len(val_dataset):,} ({len(val_loader):,} batches)")
    print(f"  Test:  {len(test_dataset):,} ({len(test_loader):,} batches)")
    print("="*70)
    
    return train_loader, val_loader, test_loader, vocabulary


# ============================================================================
# TEST SCRIPT
# ============================================================================

def test_dataset():
    """
    Dataset ve DataLoader'ı test et
    """
    print("\n" + "="*70)
    print("🧪 DATASET TESTİ")
    print("="*70)
    
    # DataLoader'ları oluştur
    train_loader, val_loader, test_loader, vocabulary = create_vocabulary_and_dataloaders(
        batch_size=8,
        num_workers=0,
        max_length=30,
        freq_threshold=5,
        build_vocab=True  # İlk seferinde True, sonra False
    )
    
    # İlk batch'i test et
    print("\n📦 İlk batch yükleniyor...")
    batch = next(iter(train_loader))
    
    print(f"\n✅ Batch başarıyla yüklendi!")
    print(f"\nBatch içeriği:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Captions shape: {batch['captions'].shape}")
    print(f"  Caption lengths: {batch['caption_lengths']}")
    
    print(f"\n📝 İlk caption örneği:")
    print(f"  Encoded: {batch['captions'][0][:20].tolist()}...")
    print(f"  Decoded: {vocabulary.decode(batch['captions'][0])}")
    print(f"  Original: {batch['caption_texts'][0][:100]}...")
    
    # Vocabulary istatistikleri
    print(f"\n📊 Vocabulary istatistikleri:")
    print(f"  Toplam kelime: {len(vocabulary):,}")
    print(f"  PAD token idx: {vocabulary.word2idx['<PAD>']}")
    print(f"  SOS token idx: {vocabulary.word2idx['<SOS>']}")
    print(f"  EOS token idx: {vocabulary.word2idx['<EOS>']}")
    print(f"  UNK token idx: {vocabulary.word2idx['<UNK>']}")
    
    print("\n" + "="*70)
    print("✅ TEST BAŞARILI!")
    print("="*70)
    
    return train_loader, val_loader, test_loader, vocabulary


if __name__ == "__main__":
    # Test et
    train_loader, val_loader, test_loader, vocabulary = test_dataset()