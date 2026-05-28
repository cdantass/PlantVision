from pathlib import Path
import cv2


def clean_images(split):
    img_dir = Path(f"codigo/dataset_flat/images/{split}")

    removed = 0
    total = 0

    for img_path in img_dir.glob("*"):
        total += 1

        try:
            img = cv2.imread(str(img_path))

            # Se não conseguiu ler → arquivo inválido
            if img is None:
                print(f"[REMOVIDO - inválido] {img_path.name}")
                img_path.unlink()
                removed += 1

        except Exception as e:
            print(f"[ERRO] {img_path.name}: {e}")
            img_path.unlink()
            removed += 1

    print(f"\n[{split.upper()}]")
    print(f"Total: {total}")
    print(f"Removidos: {removed}")


if __name__ == "__main__":
    clean_images("train")
    clean_images("val")

    print("\n✅ Limpeza concluída!")