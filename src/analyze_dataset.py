
from pathlib import Path
from collections import Counter
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "construction-ppe"
YAML_PATH = DATASET_DIR / "data.yaml"

# Load dataset configuration
with open(YAML_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

class_names = config["names"]
class_counts = Counter()
image_counts = Counter()

print("\n========== SAFESITE AI DATASET ANALYSIS ==========\n")

for split in ["train", "val"]:
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    print(f"{split.upper()} SET")
    print(f"Images: {len(images)}")
    print(f"Label files: {len(labels)}\n")

    for label_file in labels:
        present_classes = set()

        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:
                values = line.strip().split()

                if not values:
                    continue

                class_id = int(values[0])
                class_counts[(split, class_id)] += 1
                present_classes.add(class_id)

        for class_id in present_classes:
            image_counts[(split, class_id)] += 1

print("\n========== CLASS DISTRIBUTION ==========\n")

print(f"{'Class':<15} {'Train Objects':>14} {'Val Objects':>12} "
      f"{'Train Images':>13} {'Val Images':>11}")

for class_id, class_name in class_names.items():
    print(
        f"{class_name:<15} "
        f"{class_counts[('train', int(class_id))]:>14} "
        f"{class_counts[('val', int(class_id))]:>12} "
        f"{image_counts[('train', int(class_id))]:>13} "
        f"{image_counts[('val', int(class_id))]:>11}"
    )

print("\n========== ANALYSIS COMPLETE ==========\n")
