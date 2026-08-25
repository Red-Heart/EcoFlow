import cv2
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO

# --- Config ---
MODEL_PATH = "Roboflow/runs/detect/train-7/weights/best.pt"
SOURCE = "road_test.mp4"
CONF_THRESHOLD = 0.35          # lowered from 0.6 -- see note below
IMG_SIZE = 640
DEVICE = 0

LINE_X_MARGIN_RATIO = 1
LINE_Y_RATIO = 0.65

SEGMENT_LENGTH_KM = 0.05        # approximate real-world width of the counting line's road stretch
LOG_INTERVAL_SECONDS = 10     # write an emissions snapshot every 5 minutes of video time
LOG_PATH = "emissions_log.csv"

EMISSION_FACTORS_KG_PER_KM = {
    "car": 0.140,
    "motorcycle": 0.060,
    "bus": 0.723,
    "truck": 0.870,
}

CLASS_COLORS = {
    "car": (255, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (0, 200, 0),
    "truck": (0, 0, 255),
}


def point_side_of_line(px, py, x1, y1, x2, y2):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def calculate_emissions(vehicle_counts: dict, segment_length_km: float):
    emissions_by_class = {}
    total = 0.0
    for cls, count in vehicle_counts.items():
        factor = EMISSION_FACTORS_KG_PER_KM.get(cls, 0.0)
        emission = count * factor * segment_length_km
        emissions_by_class[cls] = round(emission, 4)
        total += emission
    return emissions_by_class, round(total, 4)


def append_to_log(timestamp, segment_km, counts, emissions_by_class, total_kg, log_path=LOG_PATH):
    file_exists = os.path.exists(log_path)
    fieldnames = ["timestamp", "segment_length_km", "car_count", "motorcycle_count",
                  "bus_count", "truck_count", "car_kg_co2", "motorcycle_kg_co2",
                  "bus_kg_co2", "truck_kg_co2", "total_kg_co2"]
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "segment_length_km": segment_km,
            "car_count": counts.get("car", 0),
            "motorcycle_count": counts.get("motorcycle", 0),
            "bus_count": counts.get("bus", 0),
            "truck_count": counts.get("truck", 0),
            "car_kg_co2": emissions_by_class.get("car", 0.0),
            "motorcycle_kg_co2": emissions_by_class.get("motorcycle", 0.0),
            "bus_kg_co2": emissions_by_class.get("bus", 0.0),
            "truck_kg_co2": emissions_by_class.get("truck", 0.0),
            "total_kg_co2": total_kg,
        })


def main():
    model = YOLO(MODEL_PATH)
    class_names = model.names

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Could not determine video resolution from source.")

    window_name = "EcoFlow - Vehicle Line Counter + Emissions"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)

    line_start = (int(frame_width * LINE_X_MARGIN_RATIO), int(frame_height * LINE_Y_RATIO))
    line_end = (int(frame_width * (1 - LINE_X_MARGIN_RATIO)), int(frame_height * LINE_Y_RATIO))

    track_last_side = {}
    counted_ids = set()
    class_counts = {name: 0 for name in class_names.values()}

    frame_index = 0
    last_log_frame = 0
    log_interval_frames = int(LOG_INTERVAL_SECONDS * fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        results = model.track(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        r = results[0]
        annotated = frame.copy()
        cv2.line(annotated, line_start, line_end, (0, 255, 255), 3)

        if r.boxes.id is not None:
            for box, track_id in zip(r.boxes, r.boxes.id.int().tolist()):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = class_names[cls_id]
                color = CLASS_COLORS.get(label, (255, 255, 255))

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"#{track_id} {label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 4, color, -1)

                side = point_side_of_line(cx, cy, *line_start, *line_end)
                side_sign = 1 if side > 0 else -1

                if track_id in track_last_side:
                    prev_sign = track_last_side[track_id]
                    if prev_sign != side_sign and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        class_counts[label] += 1

                track_last_side[track_id] = side_sign

        # --- periodic emissions snapshot + log ---
        if frame_index - last_log_frame >= log_interval_frames:
            emissions_by_class, total_kg = calculate_emissions(class_counts, SEGMENT_LENGTH_KM)
            timestamp = datetime.now().isoformat(timespec="seconds")
            append_to_log(timestamp, SEGMENT_LENGTH_KM, class_counts, emissions_by_class, total_kg)
            last_log_frame = frame_index

        # --- overlay: counts + live emissions estimate ---
        emissions_by_class, total_kg = calculate_emissions(class_counts, SEGMENT_LENGTH_KM)

        cv2.rectangle(annotated, (0, 0), (300, 210), (0, 0, 0), -1)
        y_offset = 25
        total_count = sum(class_counts.values())
        cv2.putText(annotated, f"Total crossed: {total_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        for name, count in class_counts.items():
            y_offset += 24
            cv2.putText(annotated, f"{name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS.get(name, (255, 255, 255)), 2, cv2.LINE_AA)

        y_offset += 30
        cv2.putText(annotated, f"Est. CO2: {total_kg} kg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    emissions_by_class, total_kg = calculate_emissions(class_counts, SEGMENT_LENGTH_KM)
    final_timestamp = datetime.now().isoformat(timespec="seconds")
    append_to_log(final_timestamp, SEGMENT_LENGTH_KM, class_counts, emissions_by_class, total_kg)

    print("\n--- Final Counts ---")
    for name, count in class_counts.items():
        print(f"{name}: {count}")
    print(f"Total vehicles crossed: {sum(class_counts.values())}")
    print("\n--- Estimated Emissions ---")
    for cls, val in emissions_by_class.items():
        print(f"{cls}: {val} kg CO2")
    print(f"Total: {total_kg} kg CO2")
    print(f"\nLog written to {LOG_PATH}")


if __name__ == "__main__":
    main()