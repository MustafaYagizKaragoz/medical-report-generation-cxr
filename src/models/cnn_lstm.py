import torch
import torch.nn as nn
import torchvision.models as models
import math
from typing import Tuple, List, Optional

# =========================================================================
# 1. ATTENTION (DİKKAT) MEKANİZMASI - İYİLEŞTİRİLMİŞ
# =========================================================================
class ImprovedAdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=512):
        super().__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        
        # 🔥 Attention network (daha derin)
        self.full_att = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, 1)
        )
        
        # 🔥 Context gating (iyileştirilmiş - daha derin)
        self.context_gate = nn.Sequential(
            nn.Linear(encoder_dim + decoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, encoder_out: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: [B, 49, encoder_dim]
            hidden: [B, decoder_dim]
        
        Returns:
            context: [B, encoder_dim]
            alpha: [B, 49] - attention weights
        """
        att1 = self.encoder_att(encoder_out)  # [B, 49, attention_dim]
        att2 = self.decoder_att(hidden).unsqueeze(1)  # [B, 1, attention_dim]
        
        combined = self.relu(att1 + att2)  # [B, 49, attention_dim]
        combined = self.dropout(combined)
        
        energy = self.full_att(combined)  # [B, 49, 1]
        alpha = torch.softmax(energy, dim=1)  # [B, 49, 1]
        
        context = (encoder_out * alpha).sum(dim=1)  # [B, encoder_dim]
        
        gate_input = torch.cat([context, hidden], dim=1)
        gate = self.context_gate(gate_input)  # [B, 1]
        context = gate * context  # Adaptive weighting
        
        return context, alpha.squeeze(-1)  # [B, encoder_dim], [B, 49]


# =========================================================================
# 2. ENCODER (GÖRSEL ÖZELLİK ÇIKARICI) - İYİLEŞTİRİLMİŞ
# =========================================================================
class ImprovedCNNEncoder(nn.Module):
    def __init__(self, embed_size: int, freeze_backbone: bool = True):
        super().__init__()
        
        # DenseNet121 with ImageNet weights
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = densenet.features
        
        # 🔥 Fine-tuning: ilk katmanları dondur (Transfer Learning)
        if freeze_backbone:
            for param in list(self.features.children())[:-2]:
                for p in param.parameters():
                    p.requires_grad = False
        
        # 🔥 Güçlü projection
        self.feature_projection = nn.Sequential(
            nn.Conv2d(1024, embed_size, kernel_size=1),
            nn.BatchNorm2d(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        
        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_size = embed_size
        
    def _get_sinusoidal_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding (Vaswani et al., 2017)"""
        pe = torch.zeros(1, seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
        """
        x = self.features(images)  # [B, 1024, H_out, W_out]
        x = self.feature_projection(x)  # [B, embed_size, H_out, W_out]
        
        B, C, H_out, W_out = x.shape
        seq_len = H_out * W_out  # 512x512 için burası otomatik 256 olacak!
        
        x = x.permute(0, 2, 3, 1).reshape(B, seq_len, C)  # [B, seq_len, embed_size]
        
        # 🔥 DİNAMİK POSITIONAL ENCODING (Resim boyutuna göre anlık üretilir)
        pos_enc = self._get_sinusoidal_encoding(seq_len, self.embed_size).to(images.device)
        x = x + pos_enc
        
        x = self.layer_norm(x)
        
        return x

# =========================================================================
# 3. DECODER (DİL MODELİ) - İYİLEŞTİRİLMİŞ
# =========================================================================
class ImprovedLSTMDecoder(nn.Module):
    def __init__(
        self, 
        embed_size: int, 
        hidden_size: int, 
        vocab_size: int, 
        embedding_matrix: Optional[torch.Tensor] = None,
        num_layers: int = 2, 
        attention_dim: int = 512, 
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # 🔥 Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_matrix is not None:
            self._load_embedding_matrix(embedding_matrix)
        else:
            # Xavier uniform initialization
            nn.init.xavier_uniform_(self.embedding.weight)
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 🔥 Attention Module
        self.attention = ImprovedAdditiveAttention(
            encoder_dim=embed_size,
            decoder_dim=hidden_size,
            attention_dim=attention_dim
        )
        
        # 🔥 LSTM with dropout between layers
        self.lstm = nn.LSTM(
            embed_size + embed_size,  # word embedding + visual context
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 🔥 Hidden state initialization
        self.init_h = nn.Sequential(
            nn.Linear(embed_size, hidden_size), 
            nn.Tanh()
        )
        self.init_c = nn.Sequential(
            nn.Linear(embed_size, hidden_size), 
            nn.Tanh()
        )
        
        # 🔥 Output layer (deep)
        self.out_dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
    
    def _load_embedding_matrix(self, embedding_matrix: torch.Tensor) -> None:
        """Safely load pre-trained embedding matrix"""
        if not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        
        # Ensure correct shape
        assert embedding_matrix.shape[0] == self.vocab_size, \
            f"Embedding matrix vocab size {embedding_matrix.shape[0]} != {self.vocab_size}"
        assert embedding_matrix.shape[1] == self.embed_size, \
            f"Embedding matrix embed size {embedding_matrix.shape[1]} != {self.embed_size}"
        
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True
    
    def init_hidden(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden states from image features
        
        Args:
            features: [B, 49, embed_size]
        
        Returns:
            h: [num_layers, B, hidden_size]
            c: [num_layers, B, hidden_size]
        """
        mean_features = features.mean(dim=1)  # [B, embed_size]
        h = self.init_h(mean_features)  # [B, hidden_size]
        c = self.init_c(mean_features)  # [B, hidden_size]
        
        # Expand to num_layers
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, B, hidden_size]
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return h, c
    
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (with teacher forcing)
        """
        embeddings = self.embedding_dropout(self.embedding(captions))  # [B, seq_len, embed_size]
        B, seq_len = captions.shape
        h, c = self.init_hidden(features)
        
        outputs = []
        alphas = []
        # 🚀 KRİTİK DÜZELTME: seq_len - 1 kadar dönüyoruz!
        # Çünkü <EOS> tokeninden sonrasını tahmin etmemize gerek yok.
        for t in range(seq_len - 1):
            context,alpha  = self.attention(features, h[-1])  
            
            lstm_input = torch.cat(
                [embeddings[:, t], context], 
                dim=1
            ).unsqueeze(1)  
            
            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = out.squeeze(1)  
            
            combined = torch.cat([out, context], dim=1)  
            logits = self.fc(self.out_dropout(combined))  
            
            outputs.append(logits)
            alphas.append(alpha)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [B, seq_len-1, vocab_size]
        alphas = torch.stack(alphas, dim=1)  # [B, seq_len-1, 49]
        
        return outputs,alphas

    def generate(
        self, 
        features: torch.Tensor, 
        vocab,                          # ← DÜZELTİLDİ: Vocabulary nesnesi
        max_len: int = 150,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        Greedy / top-k decoding ile inference.
        
        Args:
            features: [B, seq_len, embed_size]
            vocab: Vocabulary nesnesi (word2idx/idx2word içeren)
            max_len: Max caption uzunluğu
            temperature: Softmax sıcaklığı
            top_k: Top-k sampling (None → greedy argmax)
        Returns:
            captions: List of token id listesi
            alphas_list: Attention weight listesi
        """
        B = features.size(0)
        device = features.device
        
        # 🔥 DÜZELTİLDİ: Kendi Vocabulary sınıfımızı kullanıyoruz
        start_token = vocab.word2idx["<SOS>"]
        end_token   = vocab.word2idx["<EOS>"]
        
        current_words = torch.full((B,), start_token, dtype=torch.long, device=device)
        h, c = self.init_hidden(features)
        
        captions = [[] for _ in range(B)]
        alphas_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            embeddings = self.embedding(current_words)  # [B, embed_size]
            context, alpha = self.attention(features, h[-1])
            alphas_list.append(alpha)
            
            lstm_input = torch.cat([embeddings, context], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = out.squeeze(1)
            
            combined = torch.cat([out, context], dim=1)
            logits = self.fc(combined) / temperature
            
            if top_k is not None:
                threshold = torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                logits[logits < threshold] = float('-inf')
                probs = torch.softmax(logits, dim=1)
                predicted_word = torch.multinomial(probs, 1).squeeze(1)
            else:
                predicted_word = logits.argmax(dim=1)
            
            # Biten sequence'ları güncelleme
            for i in range(B):
                if not finished[i]:
                    captions[i].append(predicted_word[i].item())
            
            finished |= (predicted_word == end_token)
            if finished.all():
                break
            
            current_words = predicted_word
        
        return captions, alphas_list

# =========================================================================
# 4. ANA MODEL - ImageCaptioningModel
# =========================================================================
class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        embedding_matrix: Optional[torch.Tensor] = None,
        num_layers: int = 2,
        attention_dim: int = 512,
        dropout: float = 0.5,
        freeze_backbone: bool = True
    ):
        """
        Image Captioning Model for Medical Report Generation
        
        Args:
            vocab_size: Vocabulary size (CRITICAL - match your tokenizer!)
            embed_size: Embedding dimension
            hidden_size: LSTM hidden dimension
            embedding_matrix: Pre-trained embeddings (optional)
            num_layers: Number of LSTM layers
            attention_dim: Attention mechanism dimension
            dropout: Dropout rate
            freeze_backbone: Freeze DenseNet backbone for transfer learning
        """
        super(ImageCaptioningModel, self).__init__()
        
        # Validation
        assert vocab_size > 0, "vocab_size must be positive"
        assert embed_size > 0, "embed_size must be positive"
        assert hidden_size > 0, "hidden_size must be positive"
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Encoder: CNN (DenseNet-121)
        self.encoder = ImprovedCNNEncoder(embed_size=embed_size, freeze_backbone=freeze_backbone)
        
        # Decoder: LSTM with Additive Attention
        self.decoder = ImprovedLSTMDecoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            embedding_matrix=embedding_matrix,
            num_layers=num_layers,
            attention_dim=attention_dim,
            dropout=dropout
        )
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training
        
        Args:
            images: [B, 3, 224, 224] - Input chest X-ray images
            captions: [B, seq_len] - Target captions (with <CLS> at start)
        
        Returns:
            outputs: [B, seq_len, vocab_size] - Predicted logits
        """
        # Encode images to visual features
        features = self.encoder(images)  # [B, 49, embed_size]
        
        # Decode to generate captions (teacher forcing)
        outputs,alphas = self.decoder(features, captions)  # [B, seq_len, vocab_size]
        
        return outputs, alphas
        
    def generate(
        self, 
        images: torch.Tensor, 
        vocab, 
        max_len: int = 150,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:

        features = self.encoder(images)  # [B, 49, embed_size]
        captions, alphas = self.decoder.generate(
            features, 
            vocab, 
            max_len=max_len,
            temperature=temperature,
            top_k=top_k
        )
        return captions, alphas
    
    def count_parameters(self) -> dict:
        """Count trainable and non-trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            'trainable': trainable,
            'non_trainable': non_trainable,
            'total': trainable + non_trainable
        }

