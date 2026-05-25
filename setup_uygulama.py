"""
setup_uygulama.py
=================
Desktop\\uygulama klasorune demo icin gerekli tum dosyalari kopyalar.
Sadece 10 secili goruntu kopyalanir (tum veri seti degil).
"""

import sys
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from config import Config

DEST        = Path.home() / "Desktop" / "uygulama"
RANDOM_SEED = 2024
NUM_SAMPLES = 10


def copy_file(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  OK  {dst.relative_to(DEST)}")


def copy_dir(src, dst):
    src, dst = Path(src), Path(dst)
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  OK  {dst.relative_to(DEST)}/")
    else:
        print(f"  WARN klasor bulunamadi: {src}")


def main():
    print("\n" + "=" * 60)
    print(f"  Hedef: {DEST}")
    print("=" * 60)

    DEST.mkdir(parents=True, exist_ok=True)

    # ── 1. Kod dosyalari ────────────────────────────────────────
    print("\n[1] Kod dosyalari...")
    for f in ["demo_app.py", "config.py", "CALISTIR.bat"]:
        src = BASE_DIR / f
        if src.exists():
            copy_file(src, DEST / f)
        else:
            print(f"  WARN bulunamadi: {f}")

    # ── 2. src/ klasoru ─────────────────────────────────────────
    print("\n[2] src/ klasoru...")
    needed_src = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/cnn_lstm.py",
        "src/models/swin_distilgpt2.py",
        "src/data_loader/__init__.py",
        "src/data_loader/vocabulary.py",
        "src/utils/__init__.py",
    ]
    for rel in needed_src:
        src = BASE_DIR / rel
        if src.exists():
            copy_file(src, DEST / rel)
        else:
            print(f"  WARN bulunamadi: {rel}")

    # ── 3. Checkpoint'ler ────────────────────────────────────────
    print("\n[3] Checkpoint dosyalari (buyuk — bekleyin)...")
    for ckpt_dir in ["checkpoints_densenet_findings", "checkpoints_swin_distilgpt2"]:
        src_dir = BASE_DIR / ckpt_dir
        dst_dir = DEST / ckpt_dir
        if src_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            pth_files = list(src_dir.glob("*.pth"))
            if not pth_files:
                print(f"  WARN {ckpt_dir}/ icerisinde .pth bulunamadi")
            for f in pth_files:
                copy_file(f, dst_dir / f.name)
        else:
            print(f"  WARN klasor bulunamadi: {ckpt_dir}/")

    # ── 4. Vocabulary + CSV ──────────────────────────────────────
    print("\n[4] Vocabulary ve CSV...")
    vocab_src = BASE_DIR / "Data" / "vocab" / "vocabulary.pkl"
    if vocab_src.exists():
        copy_file(vocab_src, DEST / "Data" / "vocab" / "vocabulary.pkl")
    else:
        print(f"  WARN bulunamadi: {vocab_src}")

    for csv_name in ["labeled_reports_test.csv"]:
        csv_src = BASE_DIR / "Data" / "processed" / csv_name
        if csv_src.exists():
            copy_file(csv_src, DEST / "Data" / "processed" / csv_name)
        else:
            print(f"  WARN bulunamadi: {csv_name}")

    # ── 5. Secili 10 goruntu ─────────────────────────────────────
    print("\n[5] 10 secili goruntu belirleniyor...")
    test_csv = BASE_DIR / "Data" / "processed" / "labeled_reports_test.csv"
    selection_file = BASE_DIR / "demo_presentation" / "selected_indices.json"

    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["final_report"])
    df = df[df["final_report"].astype(str).str.strip() != ""].reset_index(drop=True)

    if selection_file.exists():
        with open(selection_file) as f:
            indices = json.load(f)
        print(f"  Mevcut secim yuklendi: {selection_file.name}")
    else:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        indices = sorted(random.sample(range(len(df)), min(NUM_SAMPLES, len(df))))
        print(f"  Yeni secim olusturuldu (seed={RANDOM_SEED})")

    # selected_indices.json'u hedefe kaydet
    dest_sel = DEST / "demo_presentation" / "selected_indices.json"
    dest_sel.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_sel, "w") as f:
        json.dump(indices, f, indent=2)
    print(f"  Kaydedildi: demo_presentation/selected_indices.json")

    selected_df = df.iloc[indices]
    image_paths = selected_df["image_path"].tolist()

    print(f"\n[6] 10 goruntu kopyalaniyor...")
    image_dir    = Path(Config.IMAGE_DIR)
    dest_img_dir = DEST / "OriginalData" / "official_data_iccv_final"

    copied, missing = 0, 0
    for rel_path in image_paths:
        src_img = image_dir / rel_path
        dst_img = dest_img_dir / rel_path
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            print(f"  OK  OriginalData/.../{Path(rel_path).name}")
            copied += 1
        else:
            print(f"  WARN goruntu bulunamadi: {rel_path}")
            missing += 1

    # ── Ozet ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TAMAMLANDI!")
    print(f"  Hedef klasor : {DEST}")
    print(f"  Goruntu      : {copied}/10 kopyalandi" +
          (f"  ({missing} eksik)" if missing else ""))
    print("=" * 60)
    print(f"\n  Calismak icin:")
    print(f"  -> {DEST}\\CALISTIR.bat dosyasina cift tiklayin")
    print()


if __name__ == "__main__":
    main()
