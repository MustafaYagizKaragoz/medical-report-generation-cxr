import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pickle
from collections import Counter
from tqdm import tqdm
import json



class Vocabulary:
    """
    Kelime haznesi oluştur ve yönet
    """
    
    def __init__(self, freq_threshold=5):
        """
        Args:
            freq_threshold: Bu sayıdan az geçen kelimeleri <UNK> yap
        """
        self.freq_threshold = freq_threshold
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.start_token = "<SOS>"  # Start of sentence
        self.end_token = "<EOS>"    # End of sentence
        self.unk_token = "<UNK>"    # Unknown
        
        # Vocabulary mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Special token'ları ekle"""
        special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def build_vocabulary(self, caption_list):
        """
        Caption listesinden vocabulary oluştur
        
        Args:
            caption_list: List of caption strings
        """
        print("\n🔨 Vocabulary oluşturuluyor...")
        
        # Kelimeleri say
        for caption in tqdm(caption_list, desc="Kelimeler sayılıyor"):
            words = self.tokenize(caption)
            self.word_freq.update(words)
        
        # Threshold'u geçen kelimeleri ekle
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
        
        print(f"✅ Vocabulary hazır!")
        print(f"   Toplam kelime: {len(self.word2idx):,}")
        print(f"   Threshold: {self.freq_threshold}")
        print(f"   En sık 10 kelime: {self.word_freq.most_common(10)}")
    

    def tokenize(self, text):
        """
        Daha gelişmiş Regex tabanlı tokenizer.
        Sayıları (1.5) bozmaz, noktalama işaretlerini ayırır.
        """
        text = text.lower().strip()
        
        # Regex Mantığı:
        # 1. Kelimeler ve içindeki tireler/kesmeler (örn: x-ray, patient's)
        # 2. Sayılar ve ondalık kısımları (örn: 1.5, 10)
        # 3. Geri kalan noktalama işaretleri (tek tek)
        pattern = r"[a-z]+(?:[-'][a-z]+)*|\d+(?:\.\d+)?|[.,!?;:]"
        
        tokens = re.findall(pattern, text)
        return tokens
    
    def encode(self, text, max_length=None):
        """
        Metni index'lere çevir
        
        Args:
            text: String caption
            max_length: Maksimum uzunluk (None ise sınırsız)
        Returns:
            encoded: List of indices
        """
        tokens = self.tokenize(text)
        
        # <SOS> token ekle
        encoded = [self.word2idx[self.start_token]]
        
        # Kelimeleri encode et
        for token in tokens:
            if token in self.word2idx:
                encoded.append(self.word2idx[token])
            else:
                encoded.append(self.word2idx[self.unk_token])
        
        # <EOS> token ekle
        encoded.append(self.word2idx[self.end_token])
        
        # Max length kontrolü
        if max_length is not None:
            if len(encoded) > max_length:
                encoded = encoded[:max_length-1] + [self.word2idx[self.end_token]]
        
        return encoded
    
    def decode(self, indices, skip_special_tokens=True):
        """
        Index'leri metne çevir
        
        Args:
            indices: List of indices or tensor
            skip_special_tokens: Special token'ları atla
        Returns:
            text: String
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()
        
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            
            if skip_special_tokens:
                if word in [self.pad_token, self.start_token, self.end_token]:
                    continue
            
            words.append(word)
        
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath):
        """Vocabulary'yi kaydet"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'freq_threshold': self.freq_threshold
            }, f)
        print(f"💾 Vocabulary kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Vocabulary'yi yükle"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_freq = data['word_freq']
            self.freq_threshold = data['freq_threshold']
        print(f"✅ Vocabulary yüklendi: {filepath}")
