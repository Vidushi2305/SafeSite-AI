"""
Generate submission assets for CSE471.
Outputs sample detections and data dictionary inside submission_package/Data/.
"""

import json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VAL_DIR = PROJECT_ROOT / "data" / "construction-ppe" / "images" / "val"
OUTPUT_DIR = PROJECT_ROOT / "submission_package" / "Data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPLIANT_CLASSES = {"helmet", "vest", "gloves", "boots", "goggles"}
VIOLATION_CLASSES = {"no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"}
WORKER_CLASSES = {"Person"}
EQUIPMENT_CLASSES = {"forklift"}


def get_category(class_name: str) -> str:
    if class_name in WORKER_CLASSES:
        return "Worker"
    if class_name in COMPLIANT_CLASSES:
        return "Compliant PPE"
    if class_name in VIOLATION_CLASSES:
        return "Violation"
    return "Equipment"


def generate_assets():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Pick 30 validation images
    val_images = sorted(list(VAL_DIR.glob("*.jpg")))[:30]
    print(f"Running inference on {len(val_images)} validation images using CPU...")

    results = model.predict(
        source=[str(p) for p in val_images],
        device="cpu",
        conf=0.45,
        iou=0.45,
        verbose=False
    )

    all_detections = []
    # Collect detections per image
    image_detections = {}

    for img_path, res in zip(val_images, results):
        img_name = img_path.name
        img_dets = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = round(float(box.conf[0]), 3)
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]

            det = {
                "image_name": img_name,
                "class_id": cls_id,
                "class_name": cls_name,
                "category": get_category(cls_name),
                "confidence": conf,
                "x_min": xyxy[0],
                "y_min": xyxy[1],
                "x_max": xyxy[2],
                "y_max": xyxy[3],
            }
            img_dets.append(det)

        if img_dets:
            # Sort detections within image by confidence descending
            img_dets.sort(key=lambda d: d["confidence"], reverse=True)
            image_detections[img_name] = img_dets

    # Select exactly 40 representative detection rows (between 35 and 45)
    selected_rows = []

    # 1. Take top-1 detection from each of the images that produced detections
    for img_name, dets in image_detections.items():
        if len(selected_rows) < 30 and dets:
            selected_rows.append(dets[0])

    # 2. Add high-confidence violation and equipment detections to ensure category diversity
    used_keys = {(r["image_name"], r["x_min"], r["y_min"]) for r in selected_rows}
    remaining_dets = []
    for dets in image_detections.values():
        for d in dets:
            key = (d["image_name"], d["x_min"], d["y_min"])
            if key not in used_keys:
                remaining_dets.append(d)

    # Prioritize violations and equipment first, then compliant PPE
    remaining_dets.sort(
        key=lambda d: (
            0 if d["category"] == "Violation"
            else (1 if d["category"] == "Equipment" else 2),
            -d["confidence"]
        )
    )

    for d in remaining_dets:
        if len(selected_rows) >= 40:
            break
        selected_rows.append(d)

    # Sort final records by image_name and confidence
    selected_rows.sort(key=lambda r: (r["image_name"], -r["confidence"]))

    assert 35 <= len(selected_rows) <= 45, f"Row count {len(selected_rows)} is not in 35-45 range."
    print(f"Exporting {len(selected_rows)} real detection rows across {len(set(r['image_name'] for r in selected_rows))} images...")

    df_detections = pd.DataFrame(selected_rows)

    # 1. Save sample_detections.csv
    csv_path = OUTPUT_DIR / "sample_detections.csv"
    df_detections.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df_detections)} rows)")

    # 2. Save sample_detections.json
    json_path = OUTPUT_DIR / "sample_detections.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(selected_rows, f, indent=2)
    print(f"Saved: {json_path}")

    # 3. Create and save data_dictionary.csv
    data_dict = [
        {
            "column_name": "image_name",
            "data_type": "string",
            "unit_or_format": "filename (.jpg)",
            "description": "Base filename of the analyzed validation image from data/construction-ppe/images/val/",
            "example_value": selected_rows[0]["image_name"]
        },
        {
            "column_name": "class_id",
            "data_type": "integer",
            "unit_or_format": "0-11 index",
            "description": "Numeric class index corresponding to the 12-class SafeSite AI YOLO ontology",
            "example_value": str(selected_rows[0]["class_id"])
        },
        {
            "column_name": "class_name",
            "data_type": "string",
            "unit_or_format": "label string",
            "description": "Specific object label (e.g., Person, helmet, vest, boots, forklift, no_gloves)",
            "example_value": selected_rows[0]["class_name"]
        },
        {
            "column_name": "category",
            "data_type": "string",
            "unit_or_format": "categorical",
            "description": "Operational safety classification group: Worker, Compliant PPE, Violation, or Equipment",
            "example_value": selected_rows[0]["category"]
        },
        {
            "column_name": "confidence",
            "data_type": "float",
            "unit_or_format": "probability [0.0 - 1.0]",
            "description": "Model prediction confidence score generated by the YOLO softmax head",
            "example_value": str(selected_rows[0]["confidence"])
        },
        {
            "column_name": "x_min",
            "data_type": "float",
            "unit_or_format": "pixels",
            "description": "Top-left horizontal pixel coordinate of the detected bounding box",
            "example_value": str(selected_rows[0]["x_min"])
        },
        {
            "column_name": "y_min",
            "data_type": "float",
            "unit_or_format": "pixels",
            "description": "Top-left vertical pixel coordinate of the detected bounding box",
            "example_value": str(selected_rows[0]["y_min"])
        },
        {
            "column_name": "x_max",
            "data_type": "float",
            "unit_or_format": "pixels",
            "description": "Bottom-right horizontal pixel coordinate of the detected bounding box",
            "example_value": str(selected_rows[0]["x_max"])
        },
        {
            "column_name": "y_max",
            "data_type": "float",
            "unit_or_format": "pixels",
            "description": "Bottom-right vertical pixel coordinate of the detected bounding box",
            "example_value": str(selected_rows[0]["y_max"])
        }
    ]

    df_dict = pd.DataFrame(data_dict)
    dict_path = OUTPUT_DIR / "data_dictionary.csv"
    df_dict.to_csv(dict_path, index=False)
    print(f"Saved: {dict_path}")

    print("\nAsset generation completed successfully.")


if __name__ == "__main__":
    generate_assets()
