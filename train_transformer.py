"""
🧠 Transformer (Swin + BioGPT) Eğitim Scripti
================================================
Medikal Rapor Üretimi için Transformer tabanlı model eğitimi.

Kullanım:
    python train_transformer.py

Özellikler:
    - HuggingFace VisionEncoderDecoderModel (Swin + BioGPT)
    - Mixed Precision Training (AMP)
    - Gradient Accumulation
    - Fine-tuning (encoder unfreeze) aşaması
    - Checkpoint save/load
    - Early Stopping
    - Loss curve plotting
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import time
import json
from pathlib import Path
from datetime import datetime

# --- MODÜLER IMPORTLAR ---
from config import Config
from src.data_loader.data_transformer import get_transformer_dataloaders
from src.models.transformer_model import MedicalTransformer
from src.utils.visualization import plot_loss_curve
from src.utils.early_stopping import EarlyStopping


# =========================================================================
# 🔥 LOGGER SINIFI
# =========================================================================
class TransformerTrainingLogger:
    """Training sırasında metrikleri JSON'a kaydeder"""
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / f"transformer_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'timestamps': [],
            'learning_rates': [],
            'model_type': 'SwinTransformer_BioGPT'
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        self.history['epochs'].append(epoch + 1)
        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['learning_rates'].append(lr)
        self.history['timestamps'].append(datetime.now().isoformat())
        
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_summary(self) -> dict:
        return {
            'total_epochs': len(self.history['epochs']),
            'best_train_loss': min(self.history['train_losses']) if self.history['train_losses'] else None,
            'best_val_loss': min(self.history['val_losses']) if self.history['val_losses'] else None,
            'log_file': str(self.log_file)
        }


# =========================================================================
# 📊 CHECKPOINT FONKSIYONLARI
# =========================================================================
def save_transformer_checkpoint(model, optimizer, scaler, epoch, loss, save_dir):
    """
    Transformer checkpoint kaydet.
    HuggingFace modeli kendi formatında, optimizer/scaler ayrı PyTorch formatında.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # HuggingFace model + tokenizer kaydet
    model.save_pretrained(save_dir)
    
    # Optimizer, scaler, epoch bilgisi kaydet
    training_state = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(training_state, os.path.join(save_dir, "training_state.pth"))
    print(f"💾 Checkpoint kaydedildi: {save_dir}")


def load_transformer_checkpoint(save_dir, optimizer, scaler, device):
    """Transformer checkpoint yükle"""
    state_path = os.path.join(save_dir, "training_state.pth")
    
    if not os.path.exists(state_path):
        print(f"⚠️ Training state bulunamadı: {state_path}")
        return 0, float('inf')
    
    try:
        state = torch.load(state_path, map_location=device)
        
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            print(f"⚠️ Optimizer state yüklenemedi (normal): {e}")
        
        if scaler is not None and state.get("scaler"):
            try:
                scaler.load_state_dict(state["scaler"])
            except:
                pass
        
        start_epoch = state["epoch"] + 1
        loss = state["loss"]
        
        print(f"✅ Training state yüklendi!")
        print(f"   📍 Epoch {start_epoch}'dan devam edilecek")
        print(f"   📊 Önceki Loss: {loss:.4f}")
        
        return start_epoch, loss
        
    except Exception as e:
        print(f"❌ Checkpoint yükleme hatası: {e}")
        return 0, float('inf')


# =========================================================================
# 🚂 TRAINING LOOPS
# =========================================================================
def train_one_epoch(loader, model, optimizer, device, epoch, scaler=None, 
                    grad_accum_steps=1, grad_clip=1.0):
    """Bir epoch eğitim — HuggingFace model API kullanır"""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    loop = tqdm(loader, total=len(loader), leave=True, 
                desc=f"Epoch {epoch+1} Train", ncols=120)
    
    for batch_idx, batch in enumerate(loop):
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Mixed Precision Training
        if scaler is not None:
            with autocast():
                outputs = model(
                    pixel_values=images,
                    labels=input_ids,
                    attention_mask=attention_mask
                )
                loss = outputs.loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(
                pixel_values=images,
                labels=input_ids,
                attention_mask=attention_mask
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += outputs.loss.item()
        num_batches += 1
        
        loop.set_postfix({
            'loss': f'{outputs.loss.item():.4f}',
            'avg': f'{total_loss/num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / num_batches


def validate(loader, model, device, epoch):
    """Validation döngüsü"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    loop = tqdm(loader, total=len(loader), leave=True, 
                desc=f"Epoch {epoch+1} Val", ncols=120)
    
    with torch.no_grad():
        for batch in loop:
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                pixel_values=images,
                labels=input_ids,
                attention_mask=attention_mask
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            loop.set_postfix({'loss': f'{outputs.loss.item():.4f}'})
    
    return total_loss / num_batches


# =========================================================================
# 🎯 ANA FONKSİYON
# =========================================================================
def main():
    # --- SETUP ---
    Config.setup()
    device = Config.DEVICE
    
    print(f"\n{'='*70}")
    print(f"🚀 TRANSFORMER (Swin + BioGPT) MEDICAL REPORT - EĞITIM BAŞLIYOR")
    print(f"{'='*70}")
    print(f"📱 Cihaz:                  {device}")
    print(f"📂 Veri:                   {Config.DATA_DIR}")
    print(f"💾 Checkpoint:             {Config.TRANSFORMER_CHECKPOINT_DIR}")
    print(f"📊 Image Size:             {Config.TRANSFORMER_IMAGE_SIZE}×{Config.TRANSFORMER_IMAGE_SIZE}")
    print(f"📊 Batch Size:             {Config.TRANSFORMER_BATCH_SIZE}")
    print(f"🔢 Total Epochs:           {Config.EPOCHS}")
    print(f"📈 Learning Rate:          {Config.TRANSFORMER_LEARNING_RATE}")
    print(f"🔓 Fine-tune Start Epoch:  {Config.TRANSFORMER_FINE_TUNE_START_EPOCH}")
    print(f"{'='*70}\n")
    
    # --- AYARLAR ---
    GRAD_ACCUM_STEPS = getattr(Config, 'TRANSFORMER_GRAD_ACCUM_STEPS', 4)
    GRAD_CLIP = getattr(Config, 'TRANSFORMER_GRAD_CLIP', 1.0)
    USE_AMP = getattr(Config, 'TRANSFORMER_USE_AMP', True)
    
    print(f"⚙️ Gradient Accumulation:  {GRAD_ACCUM_STEPS} adım")
    print(f"⚙️ Gradient Clip:          {GRAD_CLIP}")
    print(f"⚙️ Mixed Precision (AMP):  {'Evet' if USE_AMP else 'Hayır'}")
    print()
    
    # --- DATA LOADING ---
    print("📥 Veri yükleniyor...")
    train_loader, val_loader, tokenizer = get_transformer_dataloaders(
        train_csv=Config.TRAIN_PROCESSED_CSV,
        val_csv=Config.VAL_PROCESSED_CSV,
        image_dir=Config.IMAGE_DIR,
        batch_size=Config.TRANSFORMER_BATCH_SIZE,
        max_length=Config.TRANSFORMER_MAX_LENGTH,
        image_size=Config.TRANSFORMER_IMAGE_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    print(f"✅ Veri yüklendi!")
    print(f"   📊 Train Batches: {len(train_loader)}")
    print(f"   📊 Val Batches:   {len(val_loader)}")
    print(f"   📝 Tokenizer:     BioGPT ({len(tokenizer):,} tokens)\n")
    
    # --- MODEL ---
    print("🧠 Model oluşturuluyor...")
    
    # Resume kontrolü
    if Config.TRANSFORMER_RESUME and os.path.exists(Config.TRANSFORMER_CHECKPOINT_FILE):
        print(f"📥 Kaydedilmiş model yükleniyor: {Config.TRANSFORMER_CHECKPOINT_FILE}")
        model = MedicalTransformer.from_pretrained(
            Config.TRANSFORMER_CHECKPOINT_FILE, 
            freeze_encoder=True
        )
    else:
        model = MedicalTransformer(
            encoder_name=Config.TRANSFORMER_ENCODER,
            decoder_name=Config.TRANSFORMER_DECODER,
            freeze_encoder=True,
            max_length=Config.TRANSFORMER_MAX_LENGTH,
            num_beams=Config.TRANSFORMER_NUM_BEAMS
        )
    
    model = model.to(device)
    
    # --- OPTIMIZER ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.TRANSFORMER_LEARNING_RATE,
        weight_decay=getattr(Config, 'TRANSFORMER_WEIGHT_DECAY', 0.01)
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_AMP and device.type == 'cuda' else None
    
    print(f"✅ Optimizer: AdamW (LR={Config.TRANSFORMER_LEARNING_RATE})")
    print(f"✅ Loss: CrossEntropyLoss (HuggingFace dahili)\n")
    
    # --- LOGGER & TRACKING ---
    logger = TransformerTrainingLogger(Config.LOG_DIR)
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=0.001)
    
    # --- CHECKPOINT RESUME ---
    start_epoch = 0
    if Config.TRANSFORMER_RESUME and os.path.exists(Config.TRANSFORMER_CHECKPOINT_FILE):
        start_epoch, _ = load_transformer_checkpoint(
            Config.TRANSFORMER_CHECKPOINT_FILE, optimizer, scaler, device
        )
        
        if start_epoch > Config.TRANSFORMER_FINE_TUNE_START_EPOCH:
            print("🔓 Fine-tuning aşaması başlamış, encoder açılıyor...")
            model.unfreeze_encoder()
            
            # Optimizer'ı güncelle (encoder parametreleri dahil)
            optimizer = optim.AdamW([
                {'params': [p for n, p in model.named_parameters() 
                           if 'encoder' in n and p.requires_grad], 
                 'lr': Config.TRANSFORMER_LEARNING_RATE * 0.1},
                {'params': [p for n, p in model.named_parameters() 
                           if 'encoder' not in n and p.requires_grad], 
                 'lr': Config.TRANSFORMER_LEARNING_RATE},
            ], weight_decay=getattr(Config, 'TRANSFORMER_WEIGHT_DECAY', 0.01))
    
    # =====================================================================
    # 🎯 MAIN TRAINING LOOP
    # =====================================================================
    print(f"\n{'='*70}")
    print("🎯 EĞITIM DÖNGÜSÜ BAŞLIYOR")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, Config.EPOCHS):
        start_time = time.time()
        
        # --- FINE-TUNING CHECK ---
        if epoch == Config.TRANSFORMER_FINE_TUNE_START_EPOCH and epoch > 0:
            print(f"\n{'='*70}")
            print(f"🔓 FINE-TUNING AŞAMASI BAŞLIYOR!")
            print(f"{'='*70}")
            
            model.unfreeze_encoder()
            
            # Differential learning rate
            optimizer = optim.AdamW([
                {'params': [p for n, p in model.named_parameters() 
                           if 'encoder' in n and p.requires_grad], 
                 'lr': Config.TRANSFORMER_LEARNING_RATE * 0.1,
                 'name': 'encoder'},
                {'params': [p for n, p in model.named_parameters() 
                           if 'encoder' not in n and p.requires_grad], 
                 'lr': Config.TRANSFORMER_LEARNING_RATE,
                 'name': 'decoder'},
            ], weight_decay=getattr(Config, 'TRANSFORMER_WEIGHT_DECAY', 0.01))
            
            print(f"✅ Encoder LR: {Config.TRANSFORMER_LEARNING_RATE * 0.1:.2e}")
            print(f"✅ Decoder LR: {Config.TRANSFORMER_LEARNING_RATE:.2e}\n")
        
        # --- TRAIN & VALIDATE ---
        train_loss = train_one_epoch(
            train_loader, model, optimizer, device, epoch,
            scaler=scaler, grad_accum_steps=GRAD_ACCUM_STEPS, grad_clip=GRAD_CLIP
        )
        train_history.append(train_loss)
        
        val_loss = validate(val_loader, model, device, epoch)
        val_history.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        duration = time.time() - start_time
        
        # --- EPOCH SUMMARY ---
        print(f"\n{'─'*70}")
        print(f"✅ Epoch {epoch+1}/{Config.EPOCHS} Tamamlandı ({duration/60:.1f} dk)")
        print(f"{'─'*70}")
        print(f"📊 Train Loss:       {train_loss:.4f}")
        print(f"📊 Val Loss:         {val_loss:.4f}")
        print(f"🔌 Learning Rate:    {current_lr:.2e}")
        print(f"{'─'*70}\n")
        
        # --- LOGGING & SAVING ---
        logger.log_epoch(epoch, train_loss, val_loss, current_lr)
        
        # Loss curve çiz
        if len(train_history) > 0:
            plot_loss_curve(
                train_history, val_history,
                save_dir=Config.TRANSFORMER_CHECKPOINT_DIR,
                filename="transformer_training_loss.png"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"✅✅ YENİ EN İYİ! Val Loss: {best_val_loss:.4f} → {val_loss:.4f}")
            best_val_loss = val_loss
            save_transformer_checkpoint(
                model, optimizer, scaler, epoch, val_loss,
                os.path.join(Config.TRANSFORMER_CHECKPOINT_DIR, "best_model")
            )
        else:
            print(f"❌ Val Loss düşmedi. Best: {best_val_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_transformer_checkpoint(
                model, optimizer, scaler, epoch, val_loss,
                os.path.join(Config.TRANSFORMER_CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}")
            )
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*70}")
            print(f"🛑 EARLY STOPPING! {Config.PATIENCE} epoch boyunca iyileşme yok.")
            print(f"{'='*70}")
            break
        
        print("─" * 70)
    
    # =====================================================================
    # 🎉 TRAINING COMPLETED
    # =====================================================================
    print(f"\n{'='*70}")
    print("🎉 TRANSFORMER EĞITIMI TAMAMLANDI!")
    print(f"{'='*70}")
    
    summary = logger.get_summary()
    print(f"\n📊 EĞITIM ÖZETİ:")
    print(f"   📈 Total Epochs:    {summary['total_epochs']}")
    print(f"   ⬇️ Best Train Loss:  {summary['best_train_loss']:.4f}")
    print(f"   ⬇️ Best Val Loss:    {summary['best_val_loss']:.4f}")
    print(f"   📁 Log:             {summary['log_file']}")
    print(f"\n💾 En İyi Model: {Config.TRANSFORMER_CHECKPOINT_DIR}/best_model/")
    print(f"{'='*70}\n")
    
    # 🚀 Test otomatik çalıştır
    print("🔬 Test aşamasına geçiliyor...")
    try:
        import test_transformer
        test_transformer.main()
    except Exception as e:
        print(f"⚠️ Otomatik test çalıştırılamadı: {e}")
        print(f"💡 Manuel test: python test_transformer.py")
    
    # Auto shutdown
    if Config.SHUTDOWN_AFTER_TRAIN:
        print("\n⚠️ Bilgisayar 60 saniye içinde kapanacak...")
        print("   💡 İptal: 'shutdown /a'")
        time.sleep(5)
        os.system("shutdown /s /t 60")


if __name__ == "__main__":
    main()
