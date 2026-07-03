import os
import sys
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.data_loader.vocabulary import Vocabulary
from src.models.cnn_lstm import ImageCaptioningModel
from src.models.swin_distilgpt2 import SwinDistilGPT2ForMTL
from transformers import AutoTokenizer
from test_cnnlstm import beam_search

# Windows Console Unicode Fix
if sys.platform.startswith("win"):
    import io
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

class InferenceDataset(Dataset):
    """
    Dataset to load raw chest X-ray images and reference reports from test CSV.
    """
    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        # Ensure drop empty reports
        self.df = self.df.dropna(subset=["final_report"]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_path"])
        ref_report = str(row["final_report"]).strip()
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback black image
            image = Image.new("RGB", (224, 224), color="black")
            
        return {
            "image_path": row["image_path"],
            "image": image,
            "ref_report": ref_report
        }

def custom_collate(batch):
    return {
        "image_paths": [item["image_path"] for item in batch],
        "images": [item["image"] for item in batch],
        "ref_reports": [item["ref_report"] for item in batch]
    }

def main():
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_csv = os.path.join(Config.BASE_DIR, "test_predictions_combined.csv")
    print(f"Output predictions will be saved to: {output_csv}")
    
    # 1. Load Vocabulary and Tokenizer
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    vocab_size = len(vocab)
    
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load CNN-LSTM Model
    print("\n📥 Loading CNN-LSTM Model...")
    cnn_model = ImageCaptioningModel(
        vocab_size=vocab_size,
        embed_size=Config.CNN_EMBED_SIZE,
        hidden_size=Config.CNN_HIDDEN_SIZE,
        attention_dim=Config.CNN_ATTENTION_DIM,
        num_layers=2,
        dropout=0.0,
        freeze_backbone=False
    ).to(device)
    cnn_ckpt = torch.load(Config.CHECKPOINT_FILE, map_location=device)
    cnn_model.load_state_dict(cnn_ckpt["state_dict"], strict=False)
    cnn_model.eval()
    print("✅ CNN-LSTM loaded successfully!")

    # 3. Load Swin-GPT Model
    print("\n📥 Loading Swin-DistilGPT2 Model...")
    swin_ckpt_path = os.path.join(Config.SWIN_CHECKPOINT_DIR, "best_model_swin_distilgpt2.pth")
    swin_model = SwinDistilGPT2ForMTL.from_pretrained_mtl(
        encoder_name="microsoft/swin-base-patch4-window7-224",
        decoder_name="distilgpt2",
        enable_gradient_checkpointing=False,
    ).to(device)
    swin_ckpt = torch.load(swin_ckpt_path, map_location=device)
    state = swin_ckpt.get("model_state", swin_ckpt)
    swin_model.load_state_dict(state, strict=False)
    swin_model.eval()
    print("✅ Swin-DistilGPT2 loaded successfully!")

    # 4. Set up transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    cnn_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    swin_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 5. Set up DataLoader
    dataset = InferenceDataset(Config.TEST_PROCESSED_CSV, Config.IMAGE_DIR)
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=custom_collate
    )
    print(f"\n📊 Total test set samples: {len(dataset):,}")

    # Prepare file header
    if os.path.exists(output_csv):
        print(f"⚠️ Warning: {output_csv} already exists. Overwriting...")
    
    # Save incrementally
    header_df = pd.DataFrame(columns=["Image ID", "Reference Report", "CNN-LSTM Prediction", "Swin Prediction"])
    header_df.to_csv(output_csv, index=False, encoding="utf-8")

    # 6. Evaluation Loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔮 Generating Predictions", ncols=100):
            image_paths = batch["image_paths"]
            images = batch["images"]
            ref_reports = batch["ref_reports"]
            
            # Prepare tensor batches
            batch_cnn_imgs = torch.stack([cnn_transform(img) for img in images]).to(device)
            batch_swin_imgs = torch.stack([swin_transform(img) for img in images]).to(device)
            
            # --- CNN-LSTM Inference (Beam Search sample-by-sample) ---
            cnn_features = cnn_model.encoder(batch_cnn_imgs)
            cnn_preds = []
            for i in range(len(images)):
                features_i = cnn_features[i].unsqueeze(0)
                pred_ids = beam_search(cnn_model, features_i, vocab, beam_size=5, max_len=150)
                pred_text = vocab.decode(pred_ids)
                cnn_preds.append(pred_text)
                
            # --- Swin-GPT Inference (Batch Generate) ---
            swin_gen_ids = swin_model.generate(
                pixel_values=batch_swin_imgs,
                max_new_tokens=150,
                num_beams=5,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            swin_preds = tokenizer.batch_decode(swin_gen_ids, skip_special_tokens=True)
            swin_preds = [p.strip() for p in swin_preds]
            
            # --- Write to CSV ---
            batch_df = pd.DataFrame({
                "Image ID": image_paths,
                "Reference Report": ref_reports,
                "CNN-LSTM Prediction": cnn_preds,
                "Swin Prediction": swin_preds
            })
            batch_df.to_csv(output_csv, mode="a", header=False, index=False, encoding="utf-8")
            
    print(f"\n🎉 Completed! Predictions saved successfully to: {output_csv}")

if __name__ == "__main__":
    main()
