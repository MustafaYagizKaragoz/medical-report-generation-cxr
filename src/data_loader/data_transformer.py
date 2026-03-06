import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from functools import partial

TOKENIZER_NAME = "microsoft/BioGPT"

class MedicalTransformerDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, transform=None, max_length=150):
        """
        Transformer mimarisi için bütüncül (chunking olmayan) veri seti.
        """
        print(f"📁 Veri seti yükleniyor: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
        self.text_col = "final_report"

        # Boş olanları temizle
        self.df = self.df.dropna(subset=[self.text_col, 'image_path']).reset_index(drop=True)
        print(f"✅ Toplam {len(self.df):,} adet tam rapor (full report) yüklendi.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Görüntüyü Yükle
        img_path = os.path.join(self.image_dir, row['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Resim açılamadı: {img_path}")
            image = Image.new('RGB', (384, 384), color='black')

        if self.transform:
            image = self.transform(image)

        # 2. Bütüncül Raporu Al (Temizleyerek)
        text = str(row[self.text_col]).lower().strip()

        return {
            'image': image,
            'text': text
        }

def transformer_collate_fn(batch, tokenizer, max_length=150):
    """
    Batch'in içindeki metinleri alır ve Transformer'ın anlayacağı
    'input_ids' ve 'attention_mask' formatına çevirir.
    Dinamik padding yapar.
    """
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]

    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        'images': images,
        'input_ids': encoded_inputs['input_ids'],
        'attention_mask': encoded_inputs['attention_mask'],
        'raw_texts': texts
    }


def get_train_transform(image_size=384):
    """
    Eğitim için tıbbi görüntüye uygun güvenli data augmentation.
    
    ⚠️ RandomHorizontalFlip KULLANILMAZ:
       Göğüs röntgeninde kalp solda olur. Flip yapılırsa kalp sağa geçer
       ve model yanlış lateralite (sağ/sol) ilişkisi öğrenir.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop(image_size),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_eval_transform(image_size=384):
    """Validation/Test için transform (augmentation yok)"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_transformer_dataloaders(
    train_csv, 
    val_csv, 
    image_dir, 
    test_csv=None,
    batch_size=4, 
    max_length=150, 
    image_size=384,
    num_workers=4
):
    """
    Eğitim, Doğrulama ve (opsiyonel) Test DataLoader'larını oluşturur.
    
    Args:
        train_csv: Eğitim CSV yolu
        val_csv: Validation CSV yolu
        image_dir: Görüntü kök dizini
        test_csv: Test CSV yolu (opsiyonel)
        batch_size: Batch boyutu
        max_length: Maksimum token uzunluğu
        image_size: Görüntü boyutu (384 önerilir)
        num_workers: DataLoader worker sayısı (Windows için 0)
    
    Returns:
        train_loader, val_loader, tokenizer (test_csv verilmişse test_loader da döner)
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # PAD token ayarla (BioGPT'de yok)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    # Dataset'ler
    train_dataset = MedicalTransformerDataset(train_csv, image_dir, tokenizer, train_transform, max_length)
    val_dataset = MedicalTransformerDataset(val_csv, image_dir, tokenizer, eval_transform, max_length)

    custom_collate = partial(transformer_collate_fn, tokenizer=tokenizer, max_length=max_length)

    # DataLoader'lar
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader (opsiyonel)
    if test_csv is not None:
        test_dataset = MedicalTransformerDataset(test_csv, image_dir, tokenizer, eval_transform, max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader, tokenizer

    return train_loader, val_loader, tokenizer


# --- TEST BLOĞU ---
if __name__ == "__main__":
    TRAIN_CSV = r"C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed\train_processed.csv"
    VAL_CSV = r"C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed\val_processed.csv"
    TEST_CSV = r"C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\Data\processed\test_processed.csv"
    IMAGE_DIR = r"C:\Users\ygz70\Desktop\Bitirme Projesi Yeni\OriginalData\official_data_iccv_final"

    train_loader, val_loader, test_loader, tokenizer = get_transformer_dataloaders(
        train_csv=TRAIN_CSV, 
        val_csv=VAL_CSV, 
        test_csv=TEST_CSV,
        image_dir=IMAGE_DIR, 
        batch_size=4
    )

    print(f"\n📦 İlk Transformer Batch'i çekiliyor...")
    batch = next(iter(train_loader))
    
    print(f"Resim Tensor Boyutu    : {batch['images'].shape}")
    print(f"Input IDs Boyutu (Text): {batch['input_ids'].shape}")
    print(f"Attention Mask Boyutu  : {batch['attention_mask'].shape}")
    
    print(f"\n📝 Tokenizer Testi:")
    orijinal_metin = batch['raw_texts'][0]
    tokenler = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
    
    print(f"Orijinal : {orijinal_metin[:100]}...")
    print(f"Tokenler : {tokenler[:15]}...")