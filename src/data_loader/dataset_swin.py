"""
dataset_swin.py
===============
Swin-B + DistilGPT-2 eğitim pipeline'ı için veri yükleme modülü.

dataset_vit.py'den farklar:
  - ImageNet normalizasyon istatistikleri:
      ViT : mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
      Swin: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    (Swin resmi HF preprocessing ile eşdeğer)
  - image_size: 224×224 (değişmez)
  - Tokenizer: distilgpt2 (değişmez)
  - Collate fn ve etiket maskeleme mantığı aynı
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms

pd.set_option("future.no_silent_downcasting", True)


# =========================================================================
# SABİTLER
# =========================================================================
PATHOLOGY_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]

# Swin-B için ImageNet normalizasyon istatistikleri
# (microsoft/swin-base-patch4-window7-224 ile eşdeğer)
SWIN_MEAN = [0.485, 0.456, 0.406]
SWIN_STD  = [0.229, 0.224, 0.225]


# =========================================================================
# DATASET
# =========================================================================
class MedicalSwinDataset(Dataset):
    """
    CSV → (image, input_ids, attention_mask, labels, labels_cls, raw_text)

    CSV Sütunları:
        image_path   : görüntü dosyasının göreceli yolu
        final_report : rapor metni
        + 14 patoloji sütunu (0, 1, NaN veya -1)
    """

    def __init__(
        self,
        csv_file:   str,
        image_dir:  str,
        tokenizer,
        transform=None,
        max_length: int = 160,
    ):
        self.df = pd.read_csv(csv_file)
        self.image_dir  = image_dir
        self.tokenizer  = tokenizer
        self.transform  = transform
        self.max_length = max_length
        self.text_col   = "final_report"

        self.df       = self.df.dropna(subset=[self.text_col, "image_path"]).reset_index(drop=True)
        self.has_labels = all(col in self.df.columns for col in PATHOLOGY_COLS)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Görüntü ───────────────────────────────────────────────────────
        img_path = os.path.join(self.image_dir, row["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        # ── Metin Tokenizasyonu ───────────────────────────────────────────
        text = str(row[self.text_col]).lower().strip()
        eos  = self.tokenizer.eos_token or ""
        if eos and not text.endswith(eos):
            text = text + " " + eos

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # ── Generation Etiketi: EOS-bilinçli maskeleme ────────────────────
        labels = input_ids.clone()
        eos_id = self.tokenizer.eos_token_id

        eos_positions = (labels == eos_id).nonzero(as_tuple=False)
        if len(eos_positions) > 0:
            first_eos_pos = int(eos_positions[0].item())
            labels[first_eos_pos + 1:] = -100
        else:
            labels[labels == self.tokenizer.pad_token_id] = -100

        # ── Patoloji Etiketleri ───────────────────────────────────────────
        if self.has_labels:
            labels_raw = row[PATHOLOGY_COLS].fillna(0.0).values.astype(float)
            labels_cls = torch.tensor(labels_raw, dtype=torch.float32)
            labels_cls[labels_cls == -1.0] = 0.0
            labels_cls[labels_cls != 1.0]  = 0.0
        else:
            labels_cls = torch.zeros(len(PATHOLOGY_COLS), dtype=torch.float32)

        return {
            "image":          image,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "labels_cls":     labels_cls,
            "raw_text":       text,
        }


# =========================================================================
# COLLATE FN
# =========================================================================
def swin_collate_fn(batch: list) -> dict:
    return {
        "images":         torch.stack([b["image"]          for b in batch]),
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "labels_cls":     torch.stack([b["labels_cls"]     for b in batch]),
        "raw_texts":      [b["raw_text"] for b in batch],
    }


# =========================================================================
# TRANSFORMLAR
# =========================================================================
def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Swin-B için eğitim augmentasyonu (hafif, medikal görüntüye uygun)."""
    return transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=SWIN_MEAN, std=SWIN_STD),
    ])


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Değerlendirme için standart transform (augmentasyon yok)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=SWIN_MEAN, std=SWIN_STD),
    ])


# =========================================================================
# DATALOADER FACTORY
# =========================================================================
def get_swin_dataloaders(
    train_csv:      str,
    val_csv:        str,
    image_dir:      str,
    test_csv:       str  = None,
    batch_size:     int  = 4,
    max_length:     int  = 160,
    image_size:     int  = 224,
    num_workers:    int  = 0,
    tokenizer_name: str  = "distilgpt2",
):
    """
    Swin-B + DistilGPT-2 pipeline için train/val(/test) dataloader'larını döndürür.

    Returns:
        (train_loader, val_loader, tokenizer)                  — test_csv=None
        (train_loader, val_loader, test_loader, tokenizer)     — test_csv verilmişse
    """
    print(f"\n   📦 Tokenizer yükleniyor: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"   ✅ Vocab size : {tokenizer.vocab_size:,}")
    print(f"   ✅ EOS token  : '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"   ✅ PAD token  : '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"   ℹ️  ImageNet normalization: mean={SWIN_MEAN}, std={SWIN_STD}")

    train_tf = get_train_transform(image_size)
    eval_tf  = get_eval_transform(image_size)

    loader_kwargs = dict(
        collate_fn=swin_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_ds = MedicalSwinDataset(train_csv, image_dir, tokenizer, train_tf, max_length)
    val_ds   = MedicalSwinDataset(val_csv,   image_dir, tokenizer, eval_tf,  max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"   📊 Train: {len(train_ds):,} örnek | {len(train_loader):,} batch")
    print(f"   📊 Val  : {len(val_ds):,} örnek | {len(val_loader):,} batch")

    if test_csv is not None:
        test_ds     = MedicalSwinDataset(test_csv, image_dir, tokenizer, eval_tf, max_length)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        print(f"   📊 Test : {len(test_ds):,} örnek | {len(test_loader):,} batch")
        return train_loader, val_loader, test_loader, tokenizer

    return train_loader, val_loader, tokenizer
