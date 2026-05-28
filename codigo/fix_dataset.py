from pathlib import Path

img_dir = Path("codigo/dataset_flat/images/train")
lbl_dir = Path("codigo/dataset_flat/labels/train")

removed = 0

for img in img_dir.glob("*.jpg"):
    label = lbl_dir / (img.stem + ".txt")

    if not label.exists():
        img.unlink()
        removed += 1

print(f"Removidas {removed} imagens sem label")