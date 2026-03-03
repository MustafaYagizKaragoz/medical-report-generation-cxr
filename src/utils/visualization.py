import matplotlib.pyplot as plt
import os

def plot_loss_curve(train_losses, val_losses, save_dir, filename="loss_curve.png"):
    """
    Train ve Validation loss değerlerini çizdirip kaydeder.
    
    Args:
        train_losses (list): Her epoch'taki eğitim hatası
        val_losses (list): Her epoch'taki doğrulama hatası
        save_dir (str): Kaydedilecek klasör yolu
        filename (str): Dosya adı
    """
    # Klasör yoksa oluştur (Garanti olsun)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 6))
    
    # Train Loss Çiz
    plt.plot(train_losses, label=f'Train Loss (Min: {min(train_losses):.4f})', 
             color='#3498db', linewidth=2, marker='o')
    
    # Val Loss Çiz
    plt.plot(val_losses, label=f'Val Loss (Min: {min(val_losses):.4f})', 
             color='#e74c3c', linewidth=2, marker='s')
    
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Dosyayı kaydet
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Hafızayı temizle
    
    print(f"📈 Grafik güncellendi: {save_path}")