import numpy as np
import torch

class EarlyStopping:
    """
    Validation loss iyileşmeyi durdurduğunda eğitimi erkenden bitirir.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        Args:
            patience (int): Loss iyileşmediğinde beklenecek epoch sayısı.
            min_delta (float): İyileşme sayılması için gereken minimum fark.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # Loss düşmedi (veya yeterince düşmedi)
            self.counter += 1
            print(f'⚠️ Early Stopping sayacı: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss düştü, sayacı sıfırla
            self.best_loss = val_loss
            self.counter = 0