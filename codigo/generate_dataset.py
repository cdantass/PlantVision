import shutil
from pathlib import Path

def flatten_dataset(split="train"):
    src = Path(f"codigo/dataset/images/{split}")
    dst = Path(f"codigo/dataset_flat/images/{split}")

    dst.mkdir(parents=True, exist_ok=True)

    for class_dir in src.iterdir():
        if class_dir.is_dir():
            for img in class_dir.glob("*.jpg"):
                new_name = f"{class_dir.name}_{img.name}"
                shutil.copy(img, dst / new_name)

    print(f"{split} reorganizado!")


# roda os dois
flatten_dataset("train")
flatten_dataset("val")