import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import json
from pathlib import Path
from datetime import datetime
import inspect

# --- MODÜLER IMPORTLAR ---
from config import Config
from src.data_loader.dataset import create_vocabulary_and_dataloaders
from src.models.cnn_lstm import ImageCaptioningModel 
from src.utils.visualization import plot_loss_curve
from src.utils.early_stopping import EarlyStopping 

# =========================================================================
# 🔥 LOGGER SINIFI
# =========================================================================
class TrainingLogger:
    """Training sırasında metrikleri JSON ve console'a kaydeder"""
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'timestamps': [],
            'learning_rates': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Epoch metrikleri kayıt et"""
        self.history['epochs'].append(epoch + 1)
        self.history['train_losses'].append(train_loss)
        self.history['val_losses'].append(val_loss)
        self.history['learning_rates'].append(lr)
        self.history['timestamps'].append(datetime.now().isoformat())
        
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_summary(self) -> dict:
        """Training özeti döndür"""
        return {
            'total_epochs': len(self.history['epochs']),
            'best_train_loss': min(self.history['train_losses']) if self.history['train_losses'] else None,
            'best_val_loss': min(self.history['val_losses']) if self.history['val_losses'] else None,
            'log_file': str(self.log_file)
        }


# =========================================================================
# 📊 CHECKPOINT FONKSIYONLARI
# =========================================================================
def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Checkpoint kaydet"""
    print(f"💾 Checkpoint kaydediliyor: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "model_type": "ImageCaptioningModel",
    }
    
    torch.save(checkpoint, filename)
    print(f"✅ Checkpoint kaydedildi!")


def load_checkpoint(filepath, model, optimizer, device):
    """Checkpoint yükle"""
    print(f"📥 Checkpoint yükleniyor: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ HATA: Checkpoint bulunamadı! {filepath}")
        return 0, float('inf')
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"⚠️ Optimizer state yüklenemedi (normal): {str(e)}")
        
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        
        print(f"✅ Checkpoint başarıyla yüklendi!")
        print(f"   📍 Eğitime {start_epoch}. epoch'tan devam edilecek")
        print(f"   📊 Önceki Loss: {loss:.4f}")
        
        return start_epoch, loss
        
    except Exception as e:
        print(f"❌ Checkpoint yükleme hatası: {str(e)}")
        return 0, float('inf')


# =========================================================================
# 🎯 MODEL INITIALIZATION - SMART
# =========================================================================
def get_model_init_params(vocab_size):
    """
    ImageCaptioningModel'in gerekli parametrelerini belirle
    Model sınıfının signature'ını kontrol ederek
    """
    try:
        # Model sınıfının __init__ parametrelerini al
        sig = inspect.signature(ImageCaptioningModel.__init__)
        params = list(sig.parameters.keys())
        
        print(f"🔍 ImageCaptioningModel.__init__ parametreleri: {params}")
        
        # Hangi parametrelerin kullanılacağını belirle
        model_kwargs = {}
        
        # Vocab size her zaman gerekli
        if 'vocab_size' in params:
            model_kwargs['vocab_size'] = vocab_size
        
        # Diğer parametreler
        if 'embed_size' in params:
            model_kwargs['embed_size'] = Config.CNN_EMBED_SIZE
        if 'hidden_size' in params:
            model_kwargs['hidden_size'] = Config.CNN_HIDDEN_SIZE
        if 'attention_dim' in params:
            model_kwargs['attention_dim'] = Config.CNN_ATTENTION_DIM
        if 'num_layers' in params:
            model_kwargs['num_layers'] = 2
        if 'dropout' in params:
            model_kwargs['dropout'] = Config.CNN_DROPOUT
        if 'freeze_backbone' in params:
            model_kwargs['freeze_backbone'] = True
        
        print(f"📊 Model init parametreleri: {model_kwargs}")
        return model_kwargs
        
    except Exception as e:
        print(f"⚠️ Model parametreleri otomatik belirlenemedi: {str(e)}")
        print(f"💡 Fallback: Sadece vocab_size kullanılıyor")
        
        return {'vocab_size': vocab_size}


# =========================================================================
# 🚂 TRAINING LOOPS
# =========================================================================
def train_one_epoch(loader, model, optimizer, criterion, device, epoch, config):
    """Bir epoch eğitim"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    loop = tqdm(loader, total=len(loader), leave=True, desc=f"Epoch {epoch+1} Train", ncols=100)
    
    for batch_idx, batch in enumerate(loop):
        imgs = batch['images'].to(device)
        captions = batch['captions'].to(device)
        
        # 🔥 FIXED: Model forward call
        # Model'in forward metodu ne kadar argüman alıyor kontrol et
        try:
            # Try 2 argüman (images, captions) - Yeni model format
            outputs,alphas = model(imgs, captions)
            
            # outputs format: [B, seq_len, vocab_size] veya 
            # (outputs, captions, lengths, alphas)
            
            # Eğer tuple dönerse parse et
            if isinstance(outputs, tuple):
                outputs, captions_out, lengths, alphas = outputs
            else:
                # Sadece outputs dönerse
                alphas = None
                
        except Exception as e:
            print(f"❌ Forward pass error: {str(e)}")
            print(f"💡 Caption lengths göndermeyi deniyorum...")
            # Fallback: caption_lengths ile
            caption_lengths = batch.get('caption_lengths', None)
            if caption_lengths is not None:
                outputs = model(imgs, captions, caption_lengths)
                if isinstance(outputs, tuple):
                    outputs, captions_out, lengths, alphas = outputs
            else:
                raise e
        
        # Loss hesaplama
        # captions'ın shape'i: [B, seq_len]
        # outputs'ın shape'i: [B, seq_len, vocab_size]
        targets = captions[:, 1:]  # <SOS> tokenı hariç tutuyoruz
        
        # Reshape for loss computation
        outputs_flat = outputs.reshape(-1, outputs.shape[-1])  # [B*seq_len, vocab_size]
        targets_flat = targets.reshape(-1)  # [B*seq_len]
        
        loss = criterion(outputs_flat, targets_flat)
        
        # Attention regularization (eğer varsa)
        if alphas is not None and alphas.numel() > 0:
            try:
                attention_loss = ((1 - alphas.sum(dim=2)) ** 2).mean()
                attention_weight = getattr(config, 'CNN_ATTENTION_LOSS_WEIGHT', 1.0)
                loss = loss + attention_weight * attention_loss
            except:
                pass  # Attention loss hesaplanamadı
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_clip = getattr(config, 'CNN_GRAD_CLIP', 5.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    return total_loss / num_batches


def validate(loader, model, criterion, device, epoch, config):
    """Validation döngüsü"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    loop = tqdm(loader, total=len(loader), leave=True, desc=f"Epoch {epoch+1} Val", ncols=100)
    
    with torch.no_grad():
        for batch in loop:
            imgs = batch['images'].to(device)
            captions = batch['captions'].to(device)
            
            # 🔥 Same forward logic as training
            try:
                outputs, alphas = model(imgs, captions)
                
                if isinstance(outputs, tuple):
                    outputs, captions_out, lengths, alphas = outputs
                else:
                    alphas = None
                    
            except Exception as e:
                caption_lengths = batch.get('caption_lengths', None)
                if caption_lengths is not None:
                    outputs = model(imgs, captions, caption_lengths)
                    if isinstance(outputs, tuple):
                        outputs, captions_out, lengths, alphas = outputs
                else:
                    raise e
            
            # Loss hesaplama
            targets = captions[:, 1:]
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])
            targets_flat = targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            
            if alphas is not None and alphas.numel() > 0:
                try:
                    attention_loss = ((1 - alphas.sum(dim=2)) ** 2).mean()
                    attention_weight = getattr(config, 'CNN_ATTENTION_LOSS_WEIGHT', 1.0)
                    loss = loss + attention_weight * attention_loss
                except:
                    pass
            
            total_loss += loss.item()
            num_batches += 1
            
            loop.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


# =========================================================================
# 🔄 FINE-TUNING
# =========================================================================
def setup_fine_tuning_optimizer(model, config):
    """Fine-tuning optimizer setup"""
    print("\n🔓 Fine-Tuning Optimizer Setup")
    print(f"   Encoder LR:  {config.CNN_FINE_TUNE_LR}")
    print(f"   Decoder LR:  {config.CNN_LEARNING_RATE}")
    
    # Try to get encoder and decoder
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name.lower():
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    if encoder_params and decoder_params:
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.CNN_FINE_TUNE_LR, 'name': 'encoder'},
            {'params': decoder_params, 'lr': config.CNN_LEARNING_RATE, 'name': 'decoder'}
        ])
    else:
        # Fallback: single optimizer
        print("⚠️ Encoder/Decoder separation failed, using single optimizer")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.CNN_LEARNING_RATE
        )
    
    return optimizer


# =========================================================================
# 🎯 ANA FONKSİYON
# =========================================================================
def main():
    # --- SETUP ---
    Config.setup()
    print(f"\n{'='*70}")
    print(f"🚀 CNN-LSTM MEDICAL REPORT GENERATION - EĞITIM BAŞLIYOR")
    print(f"{'='*70}")
    print(f"📱 Cihaz:                   {Config.DEVICE}")
    print(f"📂 Veri Kaynağı:            {Config.DATA_DIR}")
    print(f"💾 Checkpoint Yolu:         {Config.CHECKPOINT_DIR}")
    print(f"📊 Image Size:              {Config.CNN_IMAGE_SIZE}×{Config.CNN_IMAGE_SIZE}")
    print(f"📊 Batch Size:              {Config.CNN_BATCH_SIZE}")
    print(f"🔢 Total Epochs:            {Config.EPOCHS}")
    print(f"📈 Learning Rate:           {Config.CNN_LEARNING_RATE}")
    print(f"🔓 Fine-tune Start Epoch:   {Config.CNN_FINE_TUNE_START_EPOCH}")
    print(f"{'='*70}\n")
    
    # --- DATA LOADING ---
    print("📥 Veri yükleniyor...")
    train_loader, val_loader, test_loader, vocab = create_vocabulary_and_dataloaders(
        data_dir=Config.DATA_DIR,
        image_root=Config.IMAGE_DIR,
        vocab_path=Config.VOCAB_PATH,
        batch_size=Config.CNN_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        build_vocab=False,
        freq_threshold=2,
    )
    print(f"✅ Veri yüklendi!")
    print(f"   📚 Vocabulary Size: {len(vocab)}")
    print(f"   📊 Train Batches:   {len(train_loader)}")
    print(f"   📊 Val Batches:     {len(val_loader)}")
    print(f"   📊 Test Batches:    {len(test_loader)}\n")
    
    # --- MODEL INITIALIZATION - SMART ---
    print("🧠 Model oluşturuluyor...")
    
    if Config.MODEL_TYPE == 'cnn_lstm':
        # Smart parameter detection
        model_kwargs = get_model_init_params(len(vocab))
        
        try:
            model = ImageCaptioningModel(**model_kwargs).to(Config.DEVICE)
            print(f"✅ Model oluşturuldu başarıyla!")
            print(f"   Parameters: {model_kwargs}")
        except Exception as e:
            print(f"❌ Model oluşturma hatası: {str(e)}")
            print(f"💡 Fallback: Minimal parametrelerle try...")
            model = ImageCaptioningModel(vocab_size=len(vocab)).to(Config.DEVICE)
    else:
        raise ValueError(f"❌ Model tipi desteklenmiyor: {Config.MODEL_TYPE}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Total Parameters:      {total_params:,}")
    print(f"   📊 Trainable Parameters:  {trainable_params:,}\n")
    
    # --- OPTIMIZER & CRITERION ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.CNN_LEARNING_RATE,
        weight_decay=Config.CNN_WEIGHT_DECAY
    )
    
    pad_idx = vocab.word2idx.get("<PAD>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    print(f"✅ Optimizer:     AdamW (LR={Config.CNN_LEARNING_RATE}, WD={Config.CNN_WEIGHT_DECAY})")
    print(f"✅ Loss Function: CrossEntropyLoss (ignore PAD token)\n")
    
    # --- LOGGER & TRACKING ---
    logger = TrainingLogger(Config.LOG_DIR)
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=0.001)
    
    # --- CHECKPOINT LOADING ---
    start_epoch = 0
    RESUME = getattr(Config, 'RESUME', False)
    CHECKPOINT_FILE = getattr(Config, 'CHECKPOINT_FILE', None)
    
    if RESUME and CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
        print(f"📥 Checkpoint yükleniyor: {CHECKPOINT_FILE}")
        start_epoch, _ = load_checkpoint(CHECKPOINT_FILE, model, optimizer, Config.DEVICE)
        
        if start_epoch > Config.CNN_FINE_TUNE_START_EPOCH:
            print("🔓 Fine-tuning aşaması başlamış, encoder açılıyor...")
            optimizer = setup_fine_tuning_optimizer(model, Config)
    else:
        if RESUME:
            print("⚠️ RESUME=True ama checkpoint dosyası bulunamadı.")
        print("📌 Yeni eğitim başlatılıyor...")
    
    # =========================================================================
    # 🎯 MAIN TRAINING LOOP
    # =========================================================================
    print(f"\n{'='*70}")
    print("🎯 EĞITIM DÖNGÜSÜ BAŞLIYOR")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, Config.EPOCHS):
        start_time = time.time()
        
        # --- FINE-TUNING CHECK ---
        if epoch == Config.CNN_FINE_TUNE_START_EPOCH and epoch > 0:
            print(f"\n{'='*70}")
            print(f"🔓 FINE-TUNING AŞAMASI BAŞLIYOR!")
            print(f"{'='*70}")
            print(f"⏰ Epoch: {epoch}")
            
            optimizer = setup_fine_tuning_optimizer(model, Config)
            print(f"✅ Optimizer güncellendi\n")
        
        # --- TRAIN & VALIDATE ---
        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, Config.DEVICE, epoch, Config)
        train_history.append(train_loss)
        
        val_loss = validate(val_loader, model, criterion, Config.DEVICE, epoch, Config)
        val_history.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        end_time = time.time()
        duration = end_time - start_time
        
        # --- EPOCH SUMMARY ---
        print(f"\n{'─'*70}")
        print(f"✅ Epoch {epoch+1}/{Config.EPOCHS} Tamamlandı ({duration:.1f}s)")
        print(f"{'─'*70}")
        print(f"📊 Train Loss:        {train_loss:.4f}")
        print(f"📊 Val Loss:          {val_loss:.4f}")
        print(f"🔌 Learning Rate:     {current_lr:.2e}")
        print(f"{'─'*70}\n")
        
        # --- LOGGING & SAVING ---
        logger.log_epoch(epoch, train_loss, val_loss, current_lr)
        
        plot_loss_curve(
            train_history, 
            val_history, 
            save_dir=Config.CHECKPOINT_DIR, 
            filename="training_loss.png"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"✅✅ ÇOK İYİ! Val Loss düştü ({best_val_loss:.4f} → {val_loss:.4f})")
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            )
        else:
            print(f"❌ Val Loss düşmedi. Best: {best_val_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(Config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            )
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{'='*70}")
            print(f"🛑 EARLY STOPPING TETİKLENDİ!")
            print(f"{'='*70}")
            print(f"📍 Epoch: {epoch+1}")
            print(f"📊 {Config.PATIENCE} epoch boyunca iyileşme yok.")
            print(f"✅ En iyi model: {Config.CHECKPOINT_DIR}/best_model.pth")
            break
        
        print("─" * 70)
    
    # =========================================================================
    # 🎉 TRAINING COMPLETED
    # =========================================================================
    print(f"\n{'='*70}")
    print("🎉 EĞITIM TAMAMLANDI!")
    print(f"{'='*70}")
    
    summary = logger.get_summary()
    print(f"\n📊 EĞITIM ÖZETİ:")
    print(f"   📈 Total Epochs:     {summary['total_epochs']}")
    print(f"   ⬇️ Best Train Loss:   {summary['best_train_loss']:.4f}")
    print(f"   ⬇️ Best Val Loss:     {summary['best_val_loss']:.4f}")
    print(f"   📁 Log File:         {summary['log_file']}")
    print(f"\n💾 Model Kaydedildi: {Config.CHECKPOINT_DIR}/best_model.pth")
    print(f"{'='*70}\n")
    
    # Auto shutdown
    if Config.SHUTDOWN_AFTER_TRAIN:
        print("\n⚠️ DİKKAT: Bilgisayar 60 saniye içinde kapatılacak...")
        print("   💡 İptal: 'shutdown /a' (Windows) veya CTRL+C")
        time.sleep(5)
        os.system("shutdown /s /t 60")


if __name__ == "__main__":
    main()