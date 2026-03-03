import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, AutoTokenizer

class MedicalTransformer(nn.Module):
    def __init__(self, 
                 encoder_name="microsoft/swin-base-patch4-window12-384",
                 decoder_name="microsoft/BioGPT", 
                 freeze_encoder=True,
                 max_length=150,
                 num_beams=5):
        """
        Swin Transformer (Encoder) + BioGPT (Decoder) Mimarisi
        
        Args:
            encoder_name: Swin Transformer model adı
            decoder_name: BioGPT model adı
            freeze_encoder: True ise encoder dondurulur
            max_length: Maksimum rapor uzunluğu
            num_beams: Beam search genişliği
        """
        super(MedicalTransformer, self).__init__()
        
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        
        print(f"\n{'='*70}")
        print(f"🧠 Swin + BioGPT Transformer Başlatılıyor...")
        print(f"{'='*70}")
        print(f"👁️  Encoder : {encoder_name}")
        print(f"🗣️  Decoder : {decoder_name}")

        # 1. BioGPT Tokenizer'ı Yükle
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        
        # GPT modellerinde PAD token'ı yok → EOS token'ını PAD olarak ata
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("⚠️ BioGPT için PAD token = EOS token olarak ayarlandı.")

        # 2. Modelleri Birleştir (HuggingFace Cross-Attention otomatik kurar)
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_name, 
            decoder_name,
        )
        
        # 3. Model Konfigürasyonları
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # 4. Üretim (Generation) Ayarları
        self.model.config.max_length = max_length
        self.model.config.min_length = 20
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 1.2
        self.model.config.num_beams = num_beams
        self.model.config.temperature = 0.8

        # 5. Fine-Tuning Kontrolü
        if freeze_encoder:
            self._freeze_encoder()
        else:
            print("🔓 Bütün model eğitime açık (Full Fine-Tuning).")
             
        # 6. Model stats
        self._print_stats()

    def _freeze_encoder(self):
        """Encoder (Swin) parametrelerini dondur"""
        print("🔒 Encoder (Swin) donduruldu! Sadece Decoder (BioGPT) eğitilecek.")
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _print_stats(self):
        """Model istatistiklerini yazdır"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n📊 Model İstatistikleri:")
        print(f"   Toplam parametreler:  {total_params:,}")
        print(f"   Eğitilebilir:         {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Dondurulmuş:          {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Vocabulary:           {len(self.tokenizer):,} tokens")
        print(f"{'='*70}\n")

    def forward(self, pixel_values, labels=None, attention_mask=None):
        """
        Forward pass.
        
        Args:
            pixel_values: [B, 3, 384, 384] — Görüntüler
            labels: [B, seq_len] — Hedef token ID'leri (eğitimde)
            attention_mask: [B, seq_len] — Label attention mask
        
        Returns:
            HuggingFace model çıktısı (outputs.loss, outputs.logits)
        """
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
            decoder_attention_mask=attention_mask
        )
        return outputs

    def generate_report(self, pixel_values, max_length=150, num_beams=5, 
                       do_sample=False, temperature=0.8):
        """
        Görüntüden tıbbi rapor üret.
        
        Args:
            pixel_values: [B, 3, 384, 384]
            max_length: Maksimum üretim uzunluğu
            num_beams: Beam search genişliği
            do_sample: Sampling kullan (True) veya beam search (False)
            temperature: Sampling sıcaklığı
        
        Returns:
            List[str]: Üretilen raporlar
        """
        generated_ids = self.model.generate(
            pixel_values,
            max_length=max_length,
            min_length=20,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.2,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0
        )
        
        generated_text = self.tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_text

    def unfreeze_encoder(self):
        """Fine-tuning için encoder kilidini aç"""
        print(f"\n🔓 Encoder (Swin) kilidi açıldı! Fine-Tuning başlıyor...")
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        
        # Güncellenen istatistikler
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Eğitilebilir: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    def save_pretrained(self, save_dir):
        """Model ve tokenizer'ı HuggingFace formatında kaydet"""
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"💾 Model kaydedildi: {save_dir}")
    
    @classmethod
    def from_pretrained(cls, load_dir, freeze_encoder=False):
        """Kaydedilmiş modeli yükle"""
        print(f"📥 Model yükleniyor: {load_dir}")
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        
        instance.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        
        instance.model = VisionEncoderDecoderModel.from_pretrained(load_dir)
        
        if freeze_encoder:
            instance._freeze_encoder()
        
        instance._print_stats()
        return instance