# SafeSite AI — Construction PPE & Violation Detection

Automated visual compliance monitoring and construction site safety analytics powered by a fine-tuned **YOLO26n** object detection model and an interactive **Streamlit** dashboard.

## Overview

**SafeSite AI** detects workers, heavy equipment, compliant Personal Protective Equipment (PPE), and missing-PPE safety violations across static construction site images and video streams. Built as an end-to-end computer vision engineering prototype for **CSE471** at **Lovely Professional University**, the system provides real-time bounding-box visualization, compliance KPI telemetry, per-class validation diagnostics, and structured CSV/PNG/MP4 export tools.

## Model Specifications & Validation Performance

Evaluated on the 200-image validation split (`2,961` ground-truth instances) at `640 × 640` input resolution:

| Metric | Score | 
| ----- | ----- | 
| **Architecture** | YOLO26n (`models/best.pt`) | 
| **Target Classes** | 12 Classes (Workers, PPE, Violations, Equipment) | 
| **Overall mAP@50** | **73.43%** | 
| **Overall mAP@50–95** | **52.20%** | 
| **Mean Precision** | **81.40%** | 
| **Mean Recall** | **68.60%** | 

### Per-Class Benchmark Breakdown

| Class | Category | Val Instances | Precision | Recall | mAP@50 | mAP@50–95 | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| `goggles` | Compliant PPE | 139 | 94.1% | 88.5% | 94.40% | 58.21% | 
| `Person` | Worker | 481 | 92.3% | 88.4% | 93.93% | 78.07% | 
| `vest` | Compliant PPE | 340 | 91.2% | 84.1% | 89.28% | 68.86% | 
| `forklift` | Equipment | 73 | 86.5% | 76.7% | 82.79% | 61.27% | 
| `helmet` | Compliant PPE | 294 | 87.1% | 77.2% | 82.58% | 63.24% | 
| `gloves` | Compliant PPE | 381 | 85.2% | 75.1% | 80.36% | 59.04% | 
| `boots` | Compliant PPE | 481 | 81.4% | 70.1% | 75.38% | 56.47% | 
| `no_boots` | Violation | 178 | 78.2% | 68.0% | 73.05% | 56.08% | 
| `no_vest` | Violation | 130 | 75.1% | 62.3% | 67.58% | 45.49% | 
| `no_helmet` | Violation | 179 | 74.2% | 58.7% | 61.67% | 37.43% | 
| `no_gloves` | Violation | 203 | 72.4% | 57.6% | 61.45% | 37.24% | 
| `no_goggle` | Violation | 82 | 58.9% | 16.3% | 18.71% | 4.94% | 

## Key Features

* **Static Frame Inspection:** Side-by-side comparison of raw and annotated site imagery with adjustable Confidence and IoU (NMS) thresholds, class filtering, compliance ratio KPIs, Plotly class distribution charts, and PNG/CSV exports.

* **Sequential Video Analysis:** Frame-by-frame inference with progress tracking, peak violation telemetry, annotated MP4 rendering, and timestamped frame-level CSV logs.

* **Validation & Root-Cause Analytics:** Built-in benchmark dashboard documenting per-class precision/recall/mAP metrics and small-object scale analysis.

## Known Technical Limitations & Future Scope

1. **Frame-Level Object Detection Without Worker ID Association:**

   The model detects `Person` and PPE/violation bounding boxes independently per frame. It does not currently perform spatial containment/IoU matching or multi-object tracking (`ByteTrack`/`DeepSORT`) to bind missing PPE items to persistent worker IDs across frames.

2. **Small-Object Scale Sensitivity (`no_goggle` at `640 × 640`):**

   Eye-region bounding boxes average `~26 × 13 px` (`0.097%` of total frame area) at `640 × 640` resolution and suffer from class sparsity (`265` training instances), resulting in `18.71%` mAP@50. Future iterations can address this using Sliced Aided Hyper Inference (SAHI), two-stage head-crop cascades, or higher-resolution (`1280 × 1280`) training.

3. **Environmental & Occlusion Sensitivity:**

   Heavy dust, extreme low-light conditions, distant camera angles, and partial body occlusion behind scaffolding or machinery can reduce recall.

4. **CPU Inference Throughput:**

   Local CPU execution is optimized for image audits and offline video review; multi-stream real-time CCTV deployment requires GPU acceleration or TensorRT/ONNX quantization.

## Project Structure

```
SafeSite_AI/
├── .streamlit/
│   └── config.toml                  # Light SaaS UI theme configuration
├── models/
│   └── best.pt                      # Fine-tuned YOLO26n weights (12 classes)
├── src/
│   ├── analyze_dataset.py           # Dataset class & bounding-box scale analysis
│   ├── detector.py                  # SafeSiteDetector inference wrapper
│   └── generate_submission_assets.py# Sample CSV/JSON & data dictionary generator
├── app.py                           # Main Streamlit web application
├── requirements.txt                 # Cloud-compatible Python dependencies
├── .gitignore                       # Excludes .venv, raw dataset, and temp artifacts
└── README.md                        # Project documentation

```

## Local Setup & Execution

1. **Clone the repository:**

   ```
   git clone https://github.com/Vidushi2305/SafeSite-AI.git
   cd SafeSite-AI
   
   ```

2. **Create and activate a virtual environment:**

   ```
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   
   ```

4. **Launch the Streamlit application:**

   ```
   streamlit run app.py
   
   ```

## Project Team

 Vidushi Gupta — Registration No.: `12408694` | `vidushi.gupta2305@gmail.com`

**Kajal Siwach** — Registration No.: `12416919` | `kajalsiwach251@gmail.com`