import cv2
import time
import threading
import numpy as np
import re
import base64
import requests
import pytesseract
import hailo_platform as hpf
from flask import Flask, Response
import random
import argparse

VIDEO_SOURCE = "/dev/video0"
STREAM_PORT = 8090
MODEL_SIZE = 1280
HEF_PATH = "/home/odysseapi/yolo/yolov11n_1280_FIX.hef"

# =========================
# BACKEND CONFIG
# =========================
BACKEND_PUSH_URL = "http://192.168.2.122:4000/api/push-data"

#BACKEND_PUSH_URL = "http://172.20.10.2:4000/api/push-data"
PUSH_ENABLED = True
PUSH_INTERVAL_SEC = 1.0
PUSH_JPEG_QUALITY = 75
PUSH_ONLY_IF_SWIMMER = True

LABELS = [
    "swimmer",
    "boat",
    "jetski",
    "life_saving_appliances",
    "buoy",
]


# =========================
# DUMMY FLIGHT CONFIG
# =========================
FACING_TO_HEADING_DEG = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}

TRAVEL_TO_AXIS = {
    "N": ("y", 1.0),
    "S": ("y", -1.0),
    "E": ("x", 1.0),
    "W": ("x", -1.0),
}

HEADING_JITTER_DEG = 2.5
POSITION_JITTER_M = 2.0

flight_config_lock = threading.Lock()
flight_config = {
    "base_heading_deg": 0.0,
    "facing_label": "N",
    "travel_direction": "N",
}


# =========================
# ROI CONFIGURATION
# =========================

DETECT_LEFT_MARGIN = 90
DETECT_TOP_MARGIN = 120
DETECT_RIGHT_MARGIN = 300
DETECT_BOTTOM_MARGIN = 140

ALT_LEFT = 215
ALT_TOP = 945
ALT_RIGHT = 285
ALT_BOTTOM = 985

HORIZ_LEFT = 215
HORIZ_TOP = 1005
HORIZ_RIGHT = 285
HORIZ_BOTTOM = 1045

ANGLE_LEFT = 1395
ANGLE_TOP = 510
ANGLE_RIGHT = 1465
ANGLE_BOTTOM = 560

SHOW_DETECTION_ROI = True
SHOW_ALTITUDE_ROI = True
SHOW_HORIZONTAL_ROI = True
SHOW_ANGLE_ROI = True

ANGLE_VALID_MIN = -95
ANGLE_VALID_MAX = 35

ALT_OCR_INTERVAL = 0.35
HORIZ_OCR_INTERVAL = 0.35
ANGLE_OCR_INTERVAL = 0.20

ALT_SCALE = 2.0
HORIZ_SCALE = 2.0
ANGLE_SCALE = 2.0

latest_frame = None
frame_lock = threading.Lock()

ocr_source_frame = None
ocr_frame_lock = threading.Lock()

telemetry_lock = threading.Lock()
telemetry = {
    "altitude_m": None,
    "horizontal_m": None,
    "last_seen_angle_deg": None,
    "last_angle_timestamp": None,
    "alt_raw": "",
    "horizontal_raw": "",
    "angle_raw": "",
}

last_push_time = 0.0
push_lock = threading.Lock()

performance_lock = threading.Lock()
performance_stats = {
    "inference_ms": None,
    "processing_ms": None,
    "ack_ms": None,
}

app = Flask(__name__)

ALT_OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-mM'
HORIZ_OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-mM'
ANGLE_OCR_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-'


def parse_args():
    parser = argparse.ArgumentParser(description="Drone detection stream with backend push")

    parser.add_argument(
        "--facing",
        type=str,
        default="N",
        choices=["N", "S", "E", "W", "NE", "SE", "SW", "NW"],
        help="Direction the drone is facing"
    )

    parser.add_argument(
        "--heading",
        type=float,
        default=None,
        help="Optional manual heading in degrees; overrides --facing"
    )

    parser.add_argument(
        "--travel",
        type=str,
        default="N",
        choices=["N", "S", "E", "W"],
        help="Direction the drone is travelling"
    )

    return parser.parse_args()


def clamp_roi(x0, y0, x1, y1, frame_w, frame_h):
    x0 = max(0, min(frame_w - 2, x0))
    y0 = max(0, min(frame_h - 2, y0))
    x1 = max(x0 + 1, min(frame_w, x1))
    y1 = max(y0 + 1, min(frame_h, y1))
    return x0, y0, x1, y1


def letterbox(image, new_shape=(1280, 1280), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    left = pad_w // 2
    top = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return padded, scale, left, top


def draw_detections(frame, detections):
    for x1, y1, x2, y2, score, cls_id in detections:
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(max(0, x2))
        y2 = int(max(0, y2))

        label = LABELS[cls_id] if 0 <= cls_id < len(LABELS) else f"class_{cls_id}"
        text = f"{label} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def scale_boxes_back(detections, scale, pad_x, pad_y, orig_w, orig_h):
    scaled = []
    for x1, y1, x2, y2, score, cls_id in detections:
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        scaled.append((x1, y1, x2, y2, score, cls_id))
    return scaled


def offset_boxes(detections, offset_x, offset_y):
    shifted = []
    for x1, y1, x2, y2, score, cls_id in detections:
        shifted.append((
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
            score,
            cls_id
        ))
    return shifted


def get_detection_roi(frame_w, frame_h):
    x0 = DETECT_LEFT_MARGIN
    y0 = DETECT_TOP_MARGIN
    x1 = frame_w - DETECT_RIGHT_MARGIN
    y1 = frame_h - DETECT_BOTTOM_MARGIN
    return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)


def get_altitude_roi(frame_w, frame_h):
    return clamp_roi(ALT_LEFT, ALT_TOP, ALT_RIGHT, ALT_BOTTOM, frame_w, frame_h)


def get_horizontal_roi(frame_w, frame_h):
    return clamp_roi(HORIZ_LEFT, HORIZ_TOP, HORIZ_RIGHT, HORIZ_BOTTOM, frame_w, frame_h)


def get_angle_roi(frame_w, frame_h):
    return clamp_roi(ANGLE_LEFT, ANGLE_TOP, ANGLE_RIGHT, ANGLE_BOTTOM, frame_w, frame_h)


def draw_roi_box(frame, roi, color, label):
    x0, y0, x1, y1 = roi
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        frame,
        label,
        (x0, max(20, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_roi_info(frame, detect_roi, alt_roi, horiz_roi, angle_roi):
    lines = [
        f"Detect ROI: {detect_roi}",
        f"Alt ROI:    {alt_roi}",
        f"Horiz ROI:  {horiz_roi}",
        f"Angle ROI:  {angle_roi}",
    ]

    x = 20
    y = 120
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 28


def normalize_ocr_text(text):
    return (
        text.replace("O", "0")
            .replace("o", "0")
            .replace("|", "1")
            .replace("I", "1")
            .replace("l", "1")
            .replace("-", "-")
            .replace("-", "-")
            .replace(" ", "")
            .strip()
    )


def preprocess_for_ocr(roi_bgr, scale=2.0, invert=False):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    gray = cv2.medianBlur(gray, 3)

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    return binary


def parse_altitude_value(text):
    text = normalize_ocr_text(text)
    m = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if -1000.0 <= val <= 10000.0 else None


def parse_angle_value(text):
    text = normalize_ocr_text(text)
    m = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    return val if ANGLE_VALID_MIN <= val <= ANGLE_VALID_MAX else None


def read_altitude_from_frame(frame_bgr, alt_roi):
    x0, y0, x1, y1 = alt_roi
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, ""
    proc = preprocess_for_ocr(roi, scale=ALT_SCALE, invert=False)
    raw = pytesseract.image_to_string(proc, config=ALT_OCR_CONFIG).strip()
    value = parse_altitude_value(raw)
    return value, raw


def read_horizontal_from_frame(frame_bgr, horiz_roi):
    x0, y0, x1, y1 = horiz_roi
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, ""
    proc = preprocess_for_ocr(roi, scale=HORIZ_SCALE, invert=False)
    raw = pytesseract.image_to_string(proc, config=HORIZ_OCR_CONFIG).strip()
    value = parse_altitude_value(raw)
    return value, raw


def read_angle_from_frame(frame_bgr, angle_roi):
    x0, y0, x1, y1 = angle_roi
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, ""
    proc = preprocess_for_ocr(roi, scale=ANGLE_SCALE, invert=False)
    raw = pytesseract.image_to_string(proc, config=ANGLE_OCR_CONFIG).strip()
    value = parse_angle_value(raw)
    return value, raw


def draw_telemetry_overlay(frame):
    with telemetry_lock:
        alt = telemetry["altitude_m"]
        horiz = telemetry["horizontal_m"]
        angle = telemetry["last_seen_angle_deg"]
        angle_ts = telemetry["last_angle_timestamp"]
        alt_raw = telemetry["alt_raw"]
        horiz_raw = telemetry["horizontal_raw"]
        angle_raw = telemetry["angle_raw"]

    with flight_config_lock:
        facing_label = flight_config["facing_label"]
        base_heading_deg = flight_config["base_heading_deg"]
        travel_direction = flight_config["travel_direction"]

    alt_text = "Altitude: None" if alt is None else f"Altitude: {alt:.1f} m"
    horiz_text = "Horizontal: None" if horiz is None else f"Horizontal: {horiz:.1f} m"

    if angle is None:
        angle_text = "Last Angle: None"
    else:
        age = time.time() - angle_ts if angle_ts is not None else 0.0
        angle_text = f"Last Angle: {angle:.1f} deg ({age:.1f}s ago)"

    lines = [
        alt_text,
        horiz_text,
        angle_text,
        f"Facing: {facing_label} ({base_heading_deg:.1f} deg)",
        f"Travel: {travel_direction}",
        f"Alt raw: {alt_raw}",
        f"Horiz raw: {horiz_raw}",
        f"Angle raw: {angle_raw}",
    ]

    x = 20
    y = 210
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        y += 30


def ocr_loop():
    global ocr_source_frame

    last_alt_time = 0.0
    last_horiz_time = 0.0
    last_angle_time = 0.0

    while True:
        with ocr_frame_lock:
            frame = None if ocr_source_frame is None else ocr_source_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        now = time.time()

        if now - last_alt_time >= ALT_OCR_INTERVAL:
            alt_roi = get_altitude_roi(frame.shape[1], frame.shape[0])
            alt_value, alt_raw = read_altitude_from_frame(frame, alt_roi)
            with telemetry_lock:
                telemetry["alt_raw"] = alt_raw
                if alt_value is not None:
                    telemetry["altitude_m"] = alt_value
            last_alt_time = now

        if now - last_horiz_time >= HORIZ_OCR_INTERVAL:
            horiz_roi = get_horizontal_roi(frame.shape[1], frame.shape[0])
            horiz_value, horiz_raw = read_horizontal_from_frame(frame, horiz_roi)
            with telemetry_lock:
                telemetry["horizontal_raw"] = horiz_raw
                if horiz_value is not None:
                    telemetry["horizontal_m"] = horiz_value
            last_horiz_time = now

        if now - last_angle_time >= ANGLE_OCR_INTERVAL:
            angle_roi = get_angle_roi(frame.shape[1], frame.shape[0])
            angle_value, angle_raw = read_angle_from_frame(frame, angle_roi)
            with telemetry_lock:
                telemetry["angle_raw"] = angle_raw
                if angle_value is not None:
                    telemetry["last_seen_angle_deg"] = angle_value
                    telemetry["last_angle_timestamp"] = now
            last_angle_time = now

        time.sleep(0.01)


def find_best_swimmer(detections_full_space):
    best = None
    for det in detections_full_space:
        x1, y1, x2, y2, score, cls_id = det
        if LABELS[cls_id] != "swimmer":
            continue
        if best is None or score > best[4]:
            best = det
    return best


def frame_to_base64_jpeg(frame, quality=75):
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def initialize_flight_config(args):
    travel = args.travel.upper()

    if args.heading is not None:
        base_heading = args.heading % 360.0
        facing = f"MANUAL {base_heading:.1f}"
    else:
        facing = args.facing.upper()
        base_heading = FACING_TO_HEADING_DEG[facing]

    with flight_config_lock:
        flight_config["base_heading_deg"] = base_heading
        flight_config["facing_label"] = facing
        flight_config["travel_direction"] = travel

    print(
        f"[config] facing={facing} -> heading={base_heading:.1f} deg | "
        f"travel={travel}"
    )


def wrap_heading_deg(value):
    return value % 360.0


def get_dummy_drone_state(horizontal_distance_m):
    with flight_config_lock:
        base_heading_deg = flight_config["base_heading_deg"]
        facing_label = flight_config["facing_label"]
        travel_direction = flight_config["travel_direction"]

    distance = float(horizontal_distance_m) if horizontal_distance_m is not None else 0.0
    axis, sign = TRAVEL_TO_AXIS[travel_direction]

    drone_x = 0.0
    drone_y = 0.0
    if axis == "x":
        drone_x = sign * distance
    else:
        drone_y = sign * distance

    drone_x += random.uniform(-POSITION_JITTER_M, POSITION_JITTER_M)
    drone_y += random.uniform(-POSITION_JITTER_M, POSITION_JITTER_M)
    heading_deg = wrap_heading_deg(
        base_heading_deg + random.uniform(-HEADING_JITTER_DEG, HEADING_JITTER_DEG)
    )

    return {
        "x": drone_x,
        "y": drone_y,
        "heading": heading_deg,
        "base_heading": base_heading_deg,
        "facing_label": facing_label,
        "travel_direction": travel_direction,
    }


def build_payload(output_frame, detections_full_space):
    with telemetry_lock:
        altitude = telemetry["altitude_m"]
        horizontal = telemetry["horizontal_m"]
        angle = telemetry["last_seen_angle_deg"]

    drone_state = get_dummy_drone_state(horizontal)
    drone_x = round(float(drone_state["x"]), 1)
    drone_y = round(float(drone_state["y"]), 1)
    drone_z = round(float(altitude), 1) if altitude is not None else 0.0
    drone_heading = round(float(drone_state["heading"]), 1)
    drone_angle = round(float(angle), 1) if angle is not None else 0.0

    best_swimmer = find_best_swimmer(detections_full_space)

    target = None
    if best_swimmer is not None:
        x1, y1, x2, y2, score, cls_id = best_swimmer
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        target = {
            "class": "swimmer",
            "confidence": round(float(score), 2),
            "x": round(cx, 1),
            "y": round(cy, 1),
            "z": drone_z,
            "box": {
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
            },
        }

    img_b64 = frame_to_base64_jpeg(output_frame, quality=PUSH_JPEG_QUALITY)
    if not img_b64:
        return None

    payload = {
        "drone": {
            "name": "DJI Mini 5 Pro",
            "x": drone_x,
            "y": drone_y,
            "z": drone_z,
            "pitch": 0,
            "yaw": 0,
            "roll": 0,
            "heading": drone_heading,
            "angle": drone_angle,
        },
        "target": target,
        "base64": img_b64,
    }
    return payload


def maybe_push_to_backend(output_frame, detections_full_space):
    global last_push_time

    if not PUSH_ENABLED:
        return None

    now = time.time()
    with push_lock:
        if now - last_push_time < PUSH_INTERVAL_SEC:
            return None
        last_push_time = now

    best_swimmer = find_best_swimmer(detections_full_space)
    if PUSH_ONLY_IF_SWIMMER and best_swimmer is None:
        return None

    payload = build_payload(output_frame, detections_full_space)
    if payload is None:
        return None

    ack_start = time.time()
    try:
        response = requests.post(BACKEND_PUSH_URL, json=payload, timeout=2.0)
        ack_ms = (time.time() - ack_start) * 1000.0
        if response.status_code != 200:
            print(f"[push] backend returned {response.status_code}: {response.text[:200]}")
        else:
            print("[push] sent payload successfully")
        return ack_ms
    except Exception as e:
        ack_ms = (time.time() - ack_start) * 1000.0
        print(f"[push] failed: {e}")
        return ack_ms


def update_performance_stats(inference_ms=None, processing_ms=None, ack_ms=None):
    with performance_lock:
        if inference_ms is not None:
            performance_stats["inference_ms"] = inference_ms
        if processing_ms is not None:
            performance_stats["processing_ms"] = processing_ms
        if ack_ms is not None:
            performance_stats["ack_ms"] = ack_ms


def get_performance_stats():
    with performance_lock:
        return (
            performance_stats["inference_ms"],
            performance_stats["processing_ms"],
            performance_stats["ack_ms"],
        )


class HailoLiveDetector:
    def __init__(self, hef_path):
        self.hef = hpf.HEF(hef_path)
        self.target = hpf.VDevice()

        configure_params = hpf.ConfigureParams.create_from_hef(
            self.hef, interface=hpf.HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]

        self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=True,
            format_type=hpf.FormatType.UINT8,
        )
        self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32,
        )

        self.activate_ctx = self.network_group.activate(self.network_group_params)
        self.activate_ctx.__enter__()

        self.infer_pipeline = hpf.InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params,
        )
        self.infer_pipeline.__enter__()

        print("Hailo detector initialized")
        print("Input:", self.input_info.name, self.input_info.shape)
        print("Output:", self.output_info.name, self.output_info.shape)

    def infer(self, preprocessed_1280_rgb):
        model_input = np.expand_dims(preprocessed_1280_rgb.astype(np.uint8), axis=0)
        results = self.infer_pipeline.infer({self.input_info.name: model_input})
        raw = results[self.output_info.name]

        detections = []
        batch_output = raw[0] if isinstance(raw, list) and len(raw) > 0 else raw

        for cls_id, class_dets in enumerate(batch_output):
            if class_dets is None:
                continue

            for det in class_dets:
                if det is None or len(det) != 5:
                    continue

                y1, x1, y2, x2, score = det
                if score <= 0:
                    continue

                detections.append((
                    x1 * MODEL_SIZE,
                    y1 * MODEL_SIZE,
                    x2 * MODEL_SIZE,
                    y2 * MODEL_SIZE,
                    score,
                    cls_id
                ))

        return detections

    def close(self):
        try:
            self.infer_pipeline.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.activate_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.target.release()
        except Exception:
            pass


def capture_loop():
    global latest_frame, ocr_source_frame

    detector = HailoLiveDetector(HEF_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {VIDEO_SOURCE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_counter = 0
    fps_time = time.time()
    stream_fps = 0.0

    try:
        while True:
            loop_start = time.time()

            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            orig_h, orig_w = frame_bgr.shape[:2]
            output_frame = frame_bgr.copy()

            with ocr_frame_lock:
                ocr_source_frame = frame_bgr

            detect_roi = get_detection_roi(orig_w, orig_h)
            alt_roi = get_altitude_roi(orig_w, orig_h)
            horiz_roi = get_horizontal_roi(orig_w, orig_h)
            angle_roi = get_angle_roi(orig_w, orig_h)

            dx0, dy0, dx1, dy1 = detect_roi
            roi_bgr = frame_bgr[dy0:dy1, dx0:dx1]

            inference_ms = None
            if roi_bgr.size != 0:
                roi_h, roi_w = roi_bgr.shape[:2]
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                model_input, scale, pad_x, pad_y = letterbox(roi_rgb, (MODEL_SIZE, MODEL_SIZE))

                inference_start = time.time()
                detections_model_space = detector.infer(model_input)
                inference_ms = (time.time() - inference_start) * 1000.0

                detections_roi_space = scale_boxes_back(
                    detections_model_space, scale, pad_x, pad_y, roi_w, roi_h
                )
                detections_full_space = offset_boxes(detections_roi_space, dx0, dy0)
                draw_detections(output_frame, detections_full_space)
            else:
                detections_full_space = []

            processing_ms = (time.time() - loop_start) * 1000.0
            update_performance_stats(inference_ms=inference_ms, processing_ms=processing_ms)

            ack_ms = maybe_push_to_backend(output_frame, detections_full_space)
            if ack_ms is not None:
                update_performance_stats(ack_ms=ack_ms)

            if SHOW_DETECTION_ROI:
                draw_roi_box(output_frame, detect_roi, (255, 200, 0), "DETECT ROI")
            if SHOW_ALTITUDE_ROI:
                draw_roi_box(output_frame, alt_roi, (0, 255, 0), "ALT ROI")
            if SHOW_HORIZONTAL_ROI:
                draw_roi_box(output_frame, horiz_roi, (0, 200, 255), "HORIZ ROI")
            if SHOW_ANGLE_ROI:
                draw_roi_box(output_frame, angle_roi, (255, 0, 255), "ANGLE ROI")

            draw_roi_info(output_frame, detect_roi, alt_roi, horiz_roi, angle_roi)
            draw_telemetry_overlay(output_frame)

            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                stream_fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

                with telemetry_lock:
                    alt = telemetry["altitude_m"]
                    horiz = telemetry["horizontal_m"]
                    angle = telemetry["last_seen_angle_deg"]
                    alt_raw = telemetry["alt_raw"]
                    horiz_raw = telemetry["horizontal_raw"]
                    angle_raw = telemetry["angle_raw"]

                with flight_config_lock:
                    facing_label = flight_config["facing_label"]
                    base_heading_deg = flight_config["base_heading_deg"]
                    travel_direction = flight_config["travel_direction"]

                best_swimmer = find_best_swimmer(detections_full_space)
                swimmer_text = "none"
                if best_swimmer is not None:
                    x1, y1, x2, y2, score, cls_id = best_swimmer
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    swimmer_text = f"cx={cx:.1f}, cy={cy:.1f}, conf={score:.2f}"

                inference_ms, processing_ms, ack_ms = get_performance_stats()
                inference_text = "n/a" if inference_ms is None else f"{inference_ms:.1f} ms"
                processing_text = "n/a" if processing_ms is None else f"{processing_ms:.1f} ms"
                ack_text = "n/a" if ack_ms is None else f"{ack_ms:.1f} ms"

                print(
                    f"Live FPS: {stream_fps:.2f} | "
                    f"Detections: {len(detections_full_space)} | "
                    f"Inference: {inference_text} | "
                    f"Processing: {processing_text} | "
                    f"Ack: {ack_text} | "
                    f"Facing: {facing_label} ({base_heading_deg:.1f} deg) | "
                    f"Travel: {travel_direction} | "
                    f"Altitude: {alt} | Alt raw: '{alt_raw}' | "
                    f"Horizontal: {horiz} | Horiz raw: '{horiz_raw}' | "
                    f"Camera angle: {angle} | Angle raw: '{angle_raw}' | "
                    f"Best swimmer: {swimmer_text}"
                )
            cv2.putText(
                output_frame,
                f"Live FPS: {stream_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output_frame,
                f"Model input: detection ROI -> {MODEL_SIZE}x{MODEL_SIZE}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            with frame_lock:
                latest_frame = output_frame

    finally:
        cap.release()
        detector.close()


def mjpeg_generator():
    global latest_frame

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            encoded.tobytes() +
            b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
      <head><title>Drone Detection Stream</title></head>
      <body style="background:#111;color:#eee;font-family:sans-serif;">
        <h2>Drone Detection Stream</h2>
        <img src="/video_feed" style="max-width:100%;height:auto;" />
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    args = parse_args()
    initialize_flight_config(args)

    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_ocr = threading.Thread(target=ocr_loop, daemon=True)

    t_capture.start()
    t_ocr.start()

    app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True)