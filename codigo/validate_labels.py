from pathlib import Path


def validate_split(split: str):
    label_dir = Path(f"codigo/dataset_flat/labels/{split}")

    print(f"\n🔍 Validando {split.upper()}...\n")

    total_files = 0
    errors = 0

    for label_file in label_dir.glob("*.txt"):
        total_files += 1

        lines = label_file.read_text().strip().splitlines()

        if not lines:
            print(f"[ERRO] VAZIO: {label_file.name}")
            errors += 1
            continue

        for i, line in enumerate(lines):
            parts = line.split()

            # Deve ter 5 valores
            if len(parts) != 5:
                print(f"[ERRO] FORMATO ({label_file.name} linha {i+1}): {line}")
                errors += 1
                continue

            cls, x, y, w, h = parts

            try:
                x, y, w, h = map(float, (x, y, w, h))
            except ValueError:
                print(f"[ERRO] NÃO NUMÉRICO ({label_file.name} linha {i+1})")
                errors += 1
                continue

            # YOLO exige valores normalizados entre 0 e 1
            for val, name in zip([x, y, w, h], ["x", "y", "w", "h"]):
                if not (0 <= val <= 1):
                    print(f"[ERRO] FORA DO RANGE ({label_file.name} linha {i+1}): {name}={val}")
                    errors += 1

    print(f"\n📊 {split.upper()}:")
    print(f"Arquivos analisados: {total_files}")
    print(f"Erros encontrados: {errors}")


if __name__ == "__main__":
    validate_split("train")
    validate_split("val")

    print("\n✅ Validação concluída!")