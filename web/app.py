"""
EcoFlow Flask App (MVC-style) - live cv2 preview + direct ffmpeg piping + risk assessment
+ chart data for the results dashboard + user-configurable log interval/confidence threshold
+ per-video cleanup route + full folder wipe on startup

Flow:
1. On launch, uploads/, static/outputs/, and logs/ are wiped clean (see clear_folder_contents).
2. User uploads a video via the homepage, specifying a log interval and confidence threshold.
3. Filename is sanitized; a matching emissions_log_<name>.csv is created.
4. Video is processed with a LIVE cv2.imshow() window; frames are piped into ffmpeg for
   browser-ready H.264 output. Periodic snapshot rows are appended to the CSV log.
5. Risk assessment is computed from final counts.
6. Results page shows counts, emissions, risk matrix, charts (with PNG export), and a
   "Process another video" link that confirms + deletes this video's files before navigating back.
"""

import os
import csv
import json
import shutil
import subprocess
import cv2
import torch
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from risk_assessment import build_risk_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
LOG_FOLDER = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(PROJECT_DIR, "Roboflow", "runs", "detect", "train-7", "weights", "best.pt")

FFMPEG_PATH = shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SEGMENT_LENGTH_KM = 0.05
LINE_X_MARGIN_RATIO = 1
LINE_Y_RATIO = 0.65

DEFAULT_LOG_INTERVAL_SECONDS = 300
DEFAULT_CONF_THRESHOLD = 0.55

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


def clear_folder_contents(folder_path):
    """Deletes every file/subfolder inside folder_path, but keeps folder_path itself.
    Safe to call on a folder that doesn't exist yet -- it will just be created."""
    if os.path.isdir(folder_path):
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            try:
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.remove(entry_path)
                elif os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
            except OSError as e:
                print(f"Warning: could not remove {entry_path}: {e}")
    os.makedirs(folder_path, exist_ok=True)


print("Clearing previous session data (uploads, outputs, logs)...")
for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, LOG_FOLDER):
    clear_folder_contents(folder)
print("Startup cleanup complete.")

print(f"Inference device: {'GPU (cuda:0)' if DEVICE == 0 else 'CPU'}")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_model_cache = {"model": None}


def get_model():
    if _model_cache["model"] is None:
        print("Loading YOLO model...")
        _model_cache["model"] = YOLO(MODEL_PATH)
        print("Model loaded.")
    return _model_cache["model"]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def point_side_of_line(px, py, x1, y1, x2, y2):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def calculate_emissions(vehicle_counts, segment_length_km):
    emissions_by_class = {}
    total = 0.0
    for cls, count in vehicle_counts.items():
        factor = EMISSION_FACTORS_KG_PER_KM.get(cls, 0.0)
        emission = count * factor * segment_length_km
        emissions_by_class[cls] = round(emission, 4)
        total += emission
    return emissions_by_class, round(total, 4)


def write_log(log_path, timestamp, segment_km, counts, emissions_by_class, total_kg):
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


def read_log_for_chart(log_path):
    timestamps, totals, car_vals, moto_vals, bus_vals, truck_vals = [], [], [], [], [], []
    if not os.path.exists(log_path):
        return {"timestamps": [], "totals": [], "car": [], "motorcycle": [], "bus": [], "truck": []}

    with open(log_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row["timestamp"])
            totals.append(float(row["total_kg_co2"]))
            car_vals.append(float(row["car_kg_co2"]))
            moto_vals.append(float(row["motorcycle_kg_co2"]))
            bus_vals.append(float(row["bus_kg_co2"]))
            truck_vals.append(float(row["truck_kg_co2"]))

    return {
        "timestamps": timestamps,
        "totals": totals,
        "car": car_vals,
        "motorcycle": moto_vals,
        "bus": bus_vals,
        "truck": truck_vals,
    }


def start_ffmpeg_writer(output_path, width, height, fps):
    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video(input_path, base_name, log_interval_seconds, conf_threshold):
    model = get_model()
    class_names = model.names

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video: {frame_width}x{frame_height}, {total_frames} frames, {fps:.1f} fps")
    print(f"Emissions log interval: {log_interval_seconds} seconds | Confidence threshold: {conf_threshold}")

    output_filename = f"{base_name}_annotated.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    ffmpeg_proc = start_ffmpeg_writer(output_path, frame_width, frame_height, fps)

    line_start = (int(frame_width * LINE_X_MARGIN_RATIO), int(frame_height * LINE_Y_RATIO))
    line_end = (int(frame_width * (1 - LINE_X_MARGIN_RATIO)), int(frame_height * LINE_Y_RATIO))

    track_last_side = {}
    counted_ids = set()
    class_counts = {name: 0 for name in class_names.values()}

    window_name = f"EcoFlow - Processing: {base_name} (press 'q' to stop early)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame_width, 1280), min(frame_height, 720))
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    log_filename = f"emissions_log_{base_name}.csv"
    log_path = os.path.join(LOG_FOLDER, log_filename)
    log_interval_frames = max(1, int(log_interval_seconds * fps))

    frame_index = 0
    last_log_frame = 0
    stopped_early = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        results = model.track(
            source=frame,
            conf=conf_threshold,
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
                    if track_last_side[track_id] != side_sign and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        class_counts[label] += 1
                track_last_side[track_id] = side_sign

        emissions_by_class, total_kg = calculate_emissions(class_counts, SEGMENT_LENGTH_KM)

        if frame_index - last_log_frame >= log_interval_frames:
            timestamp = datetime.now().isoformat(timespec="seconds")
            write_log(log_path, timestamp, SEGMENT_LENGTH_KM, class_counts, emissions_by_class, total_kg)
            last_log_frame = frame_index
            print(f"Logged snapshot at frame {frame_index}: {class_counts} | {total_kg} kg CO2")

        cv2.rectangle(annotated, (0, 0), (320, 185), (0, 0, 0), -1)
        y_offset = 25
        total_count = sum(class_counts.values())
        cv2.putText(annotated, f"Total crossed: {total_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for name, count in class_counts.items():
            y_offset += 24
            cv2.putText(annotated, f"{name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS.get(name, (255, 255, 255)), 2, cv2.LINE_AA)

        y_offset += 30
        cv2.putText(annotated, f"Est. CO2: {total_kg:.3f} kg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        progress_text = f"Frame {frame_index}/{total_frames}"
        cv2.putText(annotated, progress_text, (frame_width - 260, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        try:
            ffmpeg_proc.stdin.write(annotated.tobytes())
        except BrokenPipeError:
            print("ffmpeg pipe closed unexpectedly; stopping frame writes.")
            break

        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped early by user (pressed 'q').")
            stopped_early = True
            break

        if frame_index % 100 == 0:
            pct = (frame_index / total_frames * 100) if total_frames else 0
            print(f"Processed {frame_index}/{total_frames} frames ({pct:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"Video processing complete{' (stopped early)' if stopped_early else ''}. ffmpeg output written to {output_path}")

    emissions_by_class, total_kg = calculate_emissions(class_counts, SEGMENT_LENGTH_KM)
    final_timestamp = datetime.now().isoformat(timespec="seconds")
    write_log(log_path, final_timestamp, SEGMENT_LENGTH_KM, class_counts, emissions_by_class, total_kg)

    risk = build_risk_matrix(class_counts, total_kg)

    print(f"Done. Counts: {class_counts} | Total CO2: {total_kg} kg | Overall Risk: {risk.overall_risk}")
    return output_filename, class_counts, emissions_by_class, total_kg, log_filename, risk


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No file part in request.", 400

    file = request.files["video"]
    if file.filename == "":
        return "No file selected. Please choose a video before submitting.", 400

    if not allowed_file(file.filename):
        return "Invalid file type. Please upload a .mp4, .avi, .mov, or .mkv video.", 400

    raw_interval = request.form.get("log_interval", "").strip()
    try:
        log_interval_seconds = int(raw_interval)
        if log_interval_seconds <= 0:
            raise ValueError
    except ValueError:
        print(f"Invalid or missing log_interval ('{raw_interval}'), using default of {DEFAULT_LOG_INTERVAL_SECONDS}s")
        log_interval_seconds = DEFAULT_LOG_INTERVAL_SECONDS

    raw_conf = request.form.get("conf_threshold", "").strip()
    try:
        conf_threshold = float(raw_conf)
        if not (0.0 < conf_threshold < 1.0):
            raise ValueError
    except ValueError:
        print(f"Invalid or missing conf_threshold ('{raw_conf}'), using default of {DEFAULT_CONF_THRESHOLD}")
        conf_threshold = DEFAULT_CONF_THRESHOLD

    original_name = secure_filename(file.filename)
    base_name = os.path.splitext(original_name)[0]
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], original_name)
    file.save(saved_path)
    print(f"Saved upload to {saved_path}")
    print("A live preview window will open on this machine -- press 'q' in it to stop early.")

    try:
        output_filename, class_counts, emissions_by_class, total_kg, log_filename, risk = process_video(
            saved_path, base_name, log_interval_seconds, conf_threshold
        )
    except Exception as e:
        print(f"ERROR during processing: {e}")
        return f"Processing failed: {e}", 500

    log_path = os.path.join(LOG_FOLDER, log_filename)
    chart_data = read_log_for_chart(log_path)

    return render_template(
        "results.html",
        video_name=original_name,
        base_name=base_name,
        output_video=output_filename,
        class_counts=class_counts,
        emissions_by_class=emissions_by_class,
        total_kg=total_kg,
        log_filename=log_filename,
        log_interval_seconds=log_interval_seconds,
        conf_threshold=conf_threshold,
        risk=risk,
        chart_data_json=json.dumps(chart_data),
    )


@app.route("/cleanup/<base_name>", methods=["POST"])
def cleanup(base_name):
    """Deletes the uploaded source video, annotated output video, and emissions CSV log
    associated with a given base_name (the sanitized filename without extension)."""
    safe_base = secure_filename(base_name)
    deleted = []
    errors = []

    for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.startswith(safe_base):
                    fpath = os.path.join(folder, fname)
                    try:
                        os.remove(fpath)
                        deleted.append(fpath)
                    except OSError as e:
                        errors.append(f"{fpath}: {e}")

    log_path = os.path.join(LOG_FOLDER, f"emissions_log_{safe_base}.csv")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            deleted.append(log_path)
        except OSError as e:
            errors.append(f"{log_path}: {e}")

    print(f"Cleanup for '{safe_base}': deleted {len(deleted)} file(s), {len(errors)} error(s).")
    if errors:
        print("Cleanup errors:", errors)

    return jsonify({"deleted": deleted, "errors": errors})


@app.route("/static/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/logs/<filename>")
def download_log(filename):
    return send_from_directory(LOG_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
