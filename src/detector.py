
from pathlib import Path

import cv2
from ultralytics import YOLO


class SafeSiteDetector:
    def __init__(self, model_path="models/best.pt", confidence=0.40):
        self.model_path = Path(model_path)
        self.confidence = confidence

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}. "
                "Please add your trained YOLO weights."
            )

        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names

    def detect_image(self, image_path, output_path=None):
        image_path = str(image_path)

        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]
        annotated_image = result.plot()

        if output_path:
            cv2.imwrite(str(output_path), annotated_image)

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({
                "class": self.class_names[class_id],
                "confidence": round(float(box.conf[0]), 3),
                "box": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })

        return annotated_image, detections

    def detect_video(self, video_path, output_path):
        video_path = str(video_path)
        output_path = str(output_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open the input video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 25

        output = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not output.isOpened():
            cap.release()
            raise ValueError("Could not create the output video.")

        total_frames = 0

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                output.write(annotated_frame)
                total_frames += 1

        finally:
            cap.release()
            output.release()

        return total_frames
