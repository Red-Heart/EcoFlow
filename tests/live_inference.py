import cv2
from ultralytics import YOLO

# --- Config ---
MODEL_PATH = "Roboflow/runs/detect/train-7/weights/best.pt"   # your 4-class trained model
SOURCE = "road_test.mp4"                             # video file path, or 0 for webcam
CONF_THRESHOLD = 0.55
IMG_SIZE = 640
DEVICE = 0

# BGR colors per class for consistent, distinguishable boxes
CLASS_COLORS = {
    "car": (255, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (0, 200, 0),
    "truck": (0, 0, 255),
}


def draw_detections(frame, result, class_names):
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        color = CLASS_COLORS.get(label, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def main():
    model = YOLO(MODEL_PATH)
    class_names = model.names

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {SOURCE}")

    window_name = "EcoFlow - Vehicle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )

        annotated = draw_detections(frame.copy(), results[0], class_names)

        counts = {}
        for box in results[0].boxes:
            name = class_names[int(box.cls[0])]
            counts[name] = counts.get(name, 0) + 1
        summary = " | ".join(f"{k}: {v}" for k, v in counts.items()) or "No detections"
        cv2.putText(annotated, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()