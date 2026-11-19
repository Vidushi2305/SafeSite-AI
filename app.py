import io
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

from src.detector import SafeSiteDetector

# ---------------- PATHS & CONSTANTS ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "Person",
    6: "no_helmet",
    7: "no_goggle",
    8: "no_gloves",
    9: "no_boots",
    10: "forklift",
    11: "no_vest",
}

COMPLIANT_CLASSES = {"helmet", "vest", "gloves", "boots", "goggles"}
VIOLATION_CLASSES = {"no_helmet", "no_vest", "no_gloves", "no_boots", "no_goggle"}
WORKER_CLASSES = {"Person"}
EQUIPMENT_CLASSES = {"forklift"}

COLOR_COMPLIANT = "#16A34A"
COLOR_VIOLATION = "#DC2626"
COLOR_WORKER = "#2563EB"
COLOR_EQUIPMENT = "#D97706"
COLOR_NEUTRAL = "#64748B"

# Validation benchmark metrics (evaluated on 200 validation images)
BENCHMARK_DATA = [
    {"class": "goggles", "category": "Compliant PPE", "instances": 139, "precision": 0.941, "recall": 0.885, "map50": 0.9440, "map50_95": 0.5821},
    {"class": "Person", "category": "Worker", "instances": 481, "precision": 0.923, "recall": 0.884, "map50": 0.9393, "map50_95": 0.7807},
    {"class": "vest", "category": "Compliant PPE", "instances": 340, "precision": 0.912, "recall": 0.841, "map50": 0.8928, "map50_95": 0.6886},
    {"class": "forklift", "category": "Equipment", "instances": 73, "precision": 0.865, "recall": 0.767, "map50": 0.8279, "map50_95": 0.6127},
    {"class": "helmet", "category": "Compliant PPE", "instances": 294, "precision": 0.871, "recall": 0.772, "map50": 0.8258, "map50_95": 0.6324},
    {"class": "gloves", "category": "Compliant PPE", "instances": 381, "precision": 0.852, "recall": 0.751, "map50": 0.8036, "map50_95": 0.5904},
    {"class": "boots", "category": "Compliant PPE", "instances": 481, "precision": 0.814, "recall": 0.701, "map50": 0.7538, "map50_95": 0.5647},
    {"class": "no_boots", "category": "Violation", "instances": 178, "precision": 0.782, "recall": 0.680, "map50": 0.7305, "map50_95": 0.5608},
    {"class": "no_vest", "category": "Violation", "instances": 130, "precision": 0.751, "recall": 0.623, "map50": 0.6758, "map50_95": 0.4549},
    {"class": "no_helmet", "category": "Violation", "instances": 179, "precision": 0.742, "recall": 0.587, "map50": 0.6167, "map50_95": 0.3743},
    {"class": "no_gloves", "category": "Violation", "instances": 203, "precision": 0.724, "recall": 0.576, "map50": 0.6145, "map50_95": 0.3724},
    {"class": "no_goggle", "category": "Violation", "instances": 82, "precision": 0.589, "recall": 0.163, "map50": 0.1871, "map50_95": 0.0494},
]

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(
    page_title="SafeSite AI — Safety Monitoring & Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- RESTRAINED TECHNICAL CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden;
        display: none;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    .sidebar .sidebar-content {
        background-color: #F1F5F9;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-compliant {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-violation {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-worker {
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    .badge-equipment {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }
    .spec-table td {
        padding: 0.35rem 0.2rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .spec-table td.label {
        color: #64748B;
        font-weight: 500;
    }
    .spec-table td.val {
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        color: #0F172A;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem !important;
    }
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL INITIALIZATION ----------------
@st.cache_resource
def load_detector_model(model_path_str: str):
    path = Path(model_path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    return SafeSiteDetector(model_path=str(path))


# ---------------- HELPER FUNCTIONS ----------------
def get_class_category(class_name: str) -> str:
    if class_name in WORKER_CLASSES:
        return "Worker"
    if class_name in COMPLIANT_CLASSES:
        return "Compliant PPE"
    if class_name in VIOLATION_CLASSES:
        return "Violation"
    return "Equipment"


def get_category_color(category: str) -> str:
    if category == "Worker":
        return COLOR_WORKER
    if category == "Compliant PPE":
        return COLOR_COMPLIANT
    if category == "Violation":
        return COLOR_VIOLATION
    return COLOR_EQUIPMENT


def render_kpi(label: str, value: str, subtext: str = "", border_color: str = "#E2E8F0"):
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 3px solid {border_color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-sub">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("<h1 style='color: #0F172A; font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem;'>SafeSite AI — Construction PPE & Violation Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 0.88rem; margin-bottom: 1.25rem;'>Automated visual compliance monitoring and site safety analytics powered by fine-tuned YOLO26n</p>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<h3 style='font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Model Specification</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem; margin-bottom: 1.25rem;">
        <table class="spec-table">
            <tr><td class="label">Architecture</td><td class="val">YOLO26n</td></tr>
            <tr><td class="label">Weights</td><td class="val">models/best.pt</td></tr>
            <tr><td class="label">Classes</td><td class="val">12 Classes</td></tr>
            <tr><td class="label">Input Scale</td><td class="val">640 x 640</td></tr>
            <tr><td class="label">Validation mAP50</td><td class="val">73.43%</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;'>Inference Controls</h3>", unsafe_allow_html=True)
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Minimum prediction confidence score to register a detection."
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression intersection-over-union overlap threshold."
    )

    all_class_names = list(CLASS_NAMES.values())
    with st.expander("Filter Classes", expanded=False):
        selected_classes = st.multiselect(
            "Class Filter",
            options=all_class_names,
            default=all_class_names,
            help="Filter visible detections by specific class labels."
        )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #64748B; margin-bottom: 0.4rem;'>Operational Scope</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem; font-size: 0.82rem; color: #475569; line-height: 1.45;">
        Detects objects per frame; does not associate PPE items to individual worker IDs.
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAIN NAVIGATION TABS ----------------
tab_image, tab_video, tab_metrics = st.tabs([
    "Image Inspection",
    "Video Analysis",
    "Validation & Dataset Metrics"
])

# ==============================================================================
# TAB 1: IMAGE INSPECTION
# ==============================================================================
with tab_image:
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 600; color: #0F172A; margin-bottom: 0.75rem;'>Static Frame Inspection</h3>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Select construction site image file",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image:
        input_image = Image.open(uploaded_image).convert("RGB")
        col_btn1, _ = st.columns([1, 4])
        with col_btn1:
            run_image = st.button("Run Inspection", type="primary", use_container_width=True)

        if run_image:
            try:
                detector = load_detector_model("models/best.pt")
                detector.confidence = confidence

                # Run inference with tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    input_image.save(tmp_file.name, format="JPEG")
                    tmp_input_path = tmp_file.name

                # Predict using underlying model to respect IoU and class filtering
                class_id_filter = [k for k, v in CLASS_NAMES.items() if v in selected_classes]
                results = detector.model.predict(
                    source=tmp_input_path,
                    conf=confidence,
                    iou=iou_threshold,
                    classes=class_id_filter if len(class_id_filter) < 12 else None,
                    verbose=False
                )

                res = results[0]
                annotated_bgr = res.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                # Extract structured detections
                detections = []
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = detector.class_names[cls_id]
                    if cls_name not in selected_classes:
                        continue
                    xyxy = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]
                    conf = round(float(box.conf[0]), 3)
                    cat = get_class_category(cls_name)
                    detections.append({
                        "ID": len(detections) + 1,
                        "Class": cls_name,
                        "Category": cat,
                        "Confidence": conf,
                        "BBox": str(xyxy)
                    })

                # Side-by-side view
                col_orig, col_annot = st.columns(2)
                with col_orig:
                    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 0.3rem;'>Input Image</p>", unsafe_allow_html=True)
                    st.image(input_image, use_container_width=True)

                with col_annot:
                    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 0.3rem;'>Annotated Detections</p>", unsafe_allow_html=True)
                    st.image(annotated_rgb, use_container_width=True)

                st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)

                if detections:
                    df = pd.DataFrame(detections)
                    class_counts = df["Class"].value_counts().to_dict()

                    workers_count = sum(class_counts.get(cls, 0) for cls in WORKER_CLASSES)
                    compliant_count = sum(class_counts.get(cls, 0) for cls in COMPLIANT_CLASSES)
                    violations_count = sum(class_counts.get(cls, 0) for cls in VIOLATION_CLASSES)
                    total_ppe = compliant_count + violations_count
                    compliance_pct = f"{(compliant_count / total_ppe * 100):.1f}%" if total_ppe > 0 else "N/A"

                    # 4 KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        render_kpi("Workers", str(workers_count), "Person instances", COLOR_WORKER)
                    with k2:
                        render_kpi("Compliant PPE", str(compliant_count), "Helmet, vest, gloves, boots, goggles", COLOR_COMPLIANT)
                    with k3:
                        render_kpi("Violations", str(violations_count), "Missing PPE instances detected", COLOR_VIOLATION)
                    with k4:
                        render_kpi("Compliance Ratio", compliance_pct, "Compliant / (Compliant + Violations)", COLOR_COMPLIANT if violations_count == 0 else COLOR_VIOLATION)

                    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

                    # Charts & Table Section
                    col_chart, col_table = st.columns([1, 1.2])

                    with col_chart:
                        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;'>Detected Class Distribution</p>", unsafe_allow_html=True)
                        chart_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c])
                        chart_values = [class_counts[c] for c in chart_classes]
                        bar_colors = [get_category_color(get_class_category(c)) for c in chart_classes]

                        fig = go.Figure(go.Bar(
                            x=chart_values,
                            y=chart_classes,
                            orientation="h",
                            marker=dict(color=bar_colors, line=dict(width=0)),
                            text=chart_values,
                            textposition="auto",
                            textfont=dict(family="JetBrains Mono", size=11, color="#FFFFFF")
                        ))
                        fig.update_layout(
                            height=280,
                            margin=dict(l=10, r=10, t=10, b=10),
                            plot_bgcolor="#F8FAFC",
                            paper_bgcolor="#FFFFFF",
                            xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False, dtick=1, tickfont=dict(color="#475569", size=11)),
                            yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A", size=12)),
                            font=dict(family="Inter", size=12, color="#0F172A")
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col_table:
                        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;'>Detection Records</p>", unsafe_allow_html=True)
                        st.dataframe(
                            df[["ID", "Class", "Category", "Confidence", "BBox"]],
                            use_container_width=True,
                            height=280,
                            hide_index=True
                        )

                    # Export Section
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    col_dl1, col_dl2, _ = st.columns([1.2, 1.2, 2.6])

                    with col_dl1:
                        # PNG Buffer
                        is_success, buffer = cv2.imencode(".png", annotated_bgr)
                        if is_success:
                            st.download_button(
                                "Export Annotated PNG",
                                data=buffer.tobytes(),
                                file_name="safesite_annotated.png",
                                mime="image/png",
                                use_container_width=True
                            )

                    with col_dl2:
                        csv_data = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Export Detections CSV",
                            data=csv_data,
                            file_name="safesite_detections.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("No objects met the current confidence and IoU criteria.")

            except Exception as e:
                st.error(f"Inference execution halted: {e}")

# ==============================================================================
# TAB 2: VIDEO ANALYSIS
# ==============================================================================
with tab_video:
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 600; color: #0F172A; margin-bottom: 0.75rem;'>Video Stream Inspection</h3>", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Select video file for sequential analysis",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    if uploaded_video:
        col_vbtn, _ = st.columns([1, 4])
        with col_vbtn:
            run_video = st.button("Analyze Video", type="primary", use_container_width=True)

        if run_video:
            try:
                detector = load_detector_model("models/best.pt")
                detector.confidence = confidence

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_input:
                    tmp_input.write(uploaded_video.getvalue())
                    input_video_path = tmp_input.name

                output_video_path = OUTPUTS_DIR / "annotated_video.mp4"

                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    st.error("Failed to decode the uploaded video stream.")
                    st.stop()

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

                progress_bar = st.progress(0, text="Processing frames...")
                time_placeholder = st.empty()

                frame_idx = 0
                peak_violations = 0
                total_detections_count = 0
                video_telemetry = []

                class_id_filter = [k for k, v in CLASS_NAMES.items() if v in selected_classes]
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    res = detector.model.predict(
                        source=frame,
                        conf=confidence,
                        iou=iou_threshold,
                        classes=class_id_filter if len(class_id_filter) < 12 else None,
                        verbose=False
                    )[0]

                    annotated_frame = res.plot()
                    writer.write(annotated_frame)

                    # Frame telemetry
                    frame_violations = 0
                    frame_workers = 0
                    frame_compliant = 0

                    for box in res.boxes:
                        cls_name = detector.class_names[int(box.cls[0])]
                        if cls_name in selected_classes:
                            total_detections_count += 1
                            if cls_name in VIOLATION_CLASSES:
                                frame_violations += 1
                            elif cls_name in COMPLIANT_CLASSES:
                                frame_compliant += 1
                            elif cls_name in WORKER_CLASSES:
                                frame_workers += 1

                    if frame_violations > peak_violations:
                        peak_violations = frame_violations

                    video_telemetry.append({
                        "frame": frame_idx,
                        "timestamp_s": round(frame_idx / fps, 2),
                        "workers": frame_workers,
                        "compliant_ppe": frame_compliant,
                        "violations": frame_violations,
                        "total_frame_detections": len(res.boxes)
                    })

                    if total_frames > 0:
                        prog_val = min(frame_idx / total_frames, 1.0)
                        progress_bar.progress(prog_val, text=f"Processing frame {frame_idx}/{total_frames} ({(prog_val * 100):.1f}%)")

                cap.release()
                writer.release()
                elapsed = time.time() - start_time
                progress_bar.progress(1.0, text=f"Analysis completed in {elapsed:.1f}s ({frame_idx} frames at {frame_idx/max(elapsed, 0.01):.1f} fps)")

                st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.25rem 0;'>", unsafe_allow_html=True)

                # Video KPI Cards
                vk1, vk2, vk3 = st.columns(3)
                with vk1:
                    render_kpi("Frames Analyzed", str(frame_idx), f"Stream rate: {fps:.1f} fps", COLOR_NEUTRAL)
                with vk2:
                    render_kpi("Peak Violations", str(peak_violations), "Maximum violations in a single frame", COLOR_VIOLATION if peak_violations > 0 else COLOR_COMPLIANT)
                with vk3:
                    render_kpi("Total Detections", str(total_detections_count), f"Avg {(total_detections_count / max(frame_idx, 1)):.1f} detections/frame", COLOR_WORKER)

                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

                # Video Player
                if output_video_path.exists():
                    with open(output_video_path, "rb") as v_file:
                        video_bytes = v_file.read()
                    st.video(video_bytes)

                    # Export controls
                    vdl1, vdl2, _ = st.columns([1.2, 1.2, 2.6])
                    with vdl1:
                        st.download_button(
                            "Export Annotated MP4",
                            data=video_bytes,
                            file_name="safesite_annotated.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    with vdl2:
                        df_video = pd.DataFrame(video_telemetry)
                        v_csv = df_video.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Export Video Log CSV",
                            data=v_csv,
                            file_name="safesite_video_log.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Video analysis execution halted: {e}")

# ==============================================================================
# TAB 3: VALIDATION & DATASET METRICS
# ==============================================================================
with tab_metrics:
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 600; color: #0F172A; margin-bottom: 0.2rem;'>Validation Benchmark & Class Performance</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 0.85rem; margin-bottom: 1.25rem;'>Official benchmark evaluated on the 200-image validation split (2,961 ground truth instances) using fine-tuned YOLO26n.</p>", unsafe_allow_html=True)

    # Overview Metrics Row
    bm1, bm2, bm3, bm4 = st.columns(4)
    with bm1:
        render_kpi("Overall mAP@50", "73.43%", "Macro average across 12 classes", COLOR_COMPLIANT)
    with bm2:
        render_kpi("Overall mAP@50-95", "52.20%", "Strict IoU threshold average", COLOR_NEUTRAL)
    with bm3:
        render_kpi("Validation Precision", "81.40%", "Mean true positive precision", COLOR_WORKER)
    with bm4:
        render_kpi("Validation Recall", "68.60%", "Mean positive detection recall", COLOR_EQUIPMENT)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Plotly Benchmark Horizontal Bar Chart
    df_bm = pd.DataFrame(BENCHMARK_DATA)
    df_bm_sorted = df_bm.sort_values(by="map50", ascending=True)

    def get_map_color(val: float) -> str:
        if val >= 0.80:
            return COLOR_COMPLIANT
        if val >= 0.60:
            return COLOR_EQUIPMENT
        return COLOR_VIOLATION

    bar_colors_bm = [get_map_color(v) for v in df_bm_sorted["map50"]]

    fig_bm = go.Figure(go.Bar(
        x=[round(v * 100, 1) for v in df_bm_sorted["map50"]],
        y=df_bm_sorted["class"],
        orientation="h",
        marker=dict(color=bar_colors_bm),
        text=[f"{v*100:.1f}%" for v in df_bm_sorted["map50"]],
        textposition="auto",
        textfont=dict(family="JetBrains Mono", size=11, color="#FFFFFF")
    ))
    fig_bm.update_layout(
        title=dict(text="Class Performance Ranking (mAP@50)", font=dict(family="Inter", size=13, color="#0F172A")),
        height=380,
        margin=dict(l=10, r=20, t=35, b=20),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", zeroline=False, range=[0, 105], title="mAP@50 (%)", tickfont=dict(color="#475569", size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A", size=12)),
        font=dict(family="Inter", size=12, color="#0F172A")
    )
    st.plotly_chart(fig_bm, use_container_width=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Data Table
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 0.4rem;'>Detailed Validation Results per Class</p>", unsafe_allow_html=True)
    df_display = df_bm.copy()
    df_display["mAP@50"] = df_display["map50"].apply(lambda x: f"{x*100:.2f}%")
    df_display["mAP@50-95"] = df_display["map50_95"].apply(lambda x: f"{x*100:.2f}%")
    df_display["Precision"] = df_display["precision"].apply(lambda x: f"{x*100:.1f}%")
    df_display["Recall"] = df_display["recall"].apply(lambda x: f"{x*100:.1f}%")
    df_display = df_display.rename(columns={
        "class": "Class",
        "category": "Category",
        "instances": "Val Instances"
    })

    st.dataframe(
        df_display[["Class", "Category", "Val Instances", "Precision", "Recall", "mAP@50", "mAP@50-95"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Technical Deep Dive on no_goggle
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #DC2626; border-radius: 6px; padding: 1.1rem 1.3rem;">
        <h4 style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin: 0 0 0.5rem 0;">Technical Root-Cause Analysis: Small-Scale Eye-Region Detection (no_goggle)</h4>
        <p style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem;">
            The <code>no_goggle</code> class exhibits lower validation performance (<strong>18.71% mAP@50</strong>) compared to body-level PPE classes like <code>vest</code> (89.28%) and <code>helmet</code> (82.58%). Investigation of the 1,000-image dataset identifies three structural causes:
        </p>
        <ul style="font-size: 0.83rem; color: #475569; line-height: 1.55; margin: 0 0 0.75rem 1.2rem; padding: 0;">
            <li><strong>Severe Scale Constraints:</strong> The average bounding box area for <code>no_goggle</code> is <strong>0.00097</strong> (0.097% of total frame area), representing an average bounding box of approximately <strong>26 x 13 pixels</strong> at 640x640 input resolution. This falls well below the standard 32x32 pixel threshold for small object detection in CNN/transformer feature pyramids.</li>
            <li><strong>Annotation Imbalance & Sparsity:</strong> The training split contains only 265 <code>no_goggle</code> instances compared to 1,962 <code>Person</code> instances and 1,282 <code>helmet</code> instances. Furthermore, over 55% of workers lack eye-region annotations entirely due to viewing angle, distance, or face occlusion.</li>
            <li><strong>High Relative Aspect Ratio Variance:</strong> Facial orientations (profile vs frontal vs tilted downward) cause extreme bounding box geometry fluctuations that make standard anchor-free box regression challenging on distant subjects.</li>
        </ul>
        <p style="font-size: 0.82rem; color: #64748B; margin: 0; line-height: 1.45;">
            <strong>Recommended Mitigations:</strong> Sliced Aided Hyper Inference (SAHI) with overlapping 320x320 tiles, a two-stage cascaded pipeline (crop detected worker head bounding boxes prior to eye-region classification), or training at 1280x1280 image resolution with scale-targeted data augmentation.
        </p>
    </div>
    """, unsafe_allow_html=True)
