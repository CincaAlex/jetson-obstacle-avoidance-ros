import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==============================
# CONFIG
# ==============================
VIDEO_PATH = "test_video/yolo3.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "dog", "chair", "bench"
}

# praguri
CORRIDOR_STOP_DIST = 0.35
SIDE_DIST = 0.45

# ==============================
# LOAD MIDAS
# ==============================
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
midas.to(DEVICE).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.dpt_transform

# ==============================
# LOAD YOLO
# ==============================
yolo = YOLO("yolov8n.pt")

# ==============================
# VIDEO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # ==============================
    # DEPTH (MiDaS)
    # ==============================
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = pred.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # ==============================
    # CORRIDOR CENTRAL (MiDaS ONLY)
    # ==============================
    cy1, cy2 = int(H * 0.65), int(H * 0.9)
    cx1, cx2 = int(W * 0.45), int(W * 0.55)

    corridor = depth[cy1:cy2, cx1:cx2]
    corridor = corridor[corridor > 0]

    corridor_dist = np.median(corridor) if corridor.size else 1.0

    # vizual corridor
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

    # ==============================
    # YOLO DETECTION
    # ==============================
    results = yolo(frame, verbose=False)[0]

    zone_depths = {
        "left": [],
        "center": [],
        "right": []
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo.names[cls_id]
        conf = float(box.conf[0])

        if label not in OBSTACLE_CLASSES or conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        roi = depth[y1:y2, x1:x2]
        roi = roi[roi > 0]
        if roi.size == 0:
            continue

        dist = np.median(roi)
        cx = (x1 + x2) / 2

        if cx < W * 0.33:
            zone = "left"
        elif cx < W * 0.66:
            zone = "center"
        else:
            zone = "right"

        zone_depths[zone].append(dist)

        # desen bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {dist:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # ==============================
    # DECISION LOGIC
    # ==============================
    decision = "FORWARD"

    if corridor_dist < CORRIDOR_STOP_DIST:
        decision = "STOP"

    else:
        min_dist = {
            z: min(vals) if vals else 1.0
            for z, vals in zone_depths.items()
        }

        if min_dist["center"] < SIDE_DIST:
            if min_dist["left"] > min_dist["right"]:
                decision = "TURN_LEFT"
            else:
                decision = "TURN_RIGHT"
        elif min_dist["left"] < SIDE_DIST:
            decision = "TURN_RIGHT"
        elif min_dist["right"] < SIDE_DIST:
            decision = "TURN_LEFT"

    # ==============================
    # ZONE VISUALIZATION
    # ==============================
    overlay = frame.copy()
    zone_w = W // 3

    zones = {
        "left":   (0, zone_w),
        "center": (zone_w, 2 * zone_w),
        "right":  (2 * zone_w, W)
    }

    for zone, (x1, x2) in zones.items():
        score = min(zone_depths[zone]) if zone_depths[zone] else 1.0

        if score < SIDE_DIST:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            overlay,
            (x1, int(H * 0.55)),
            (x2, H),
            color,
            -1
        )

    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    # ==============================
    # DISPLAY
    # ==============================
    cv2.putText(
        frame,
        f"Decision: {decision}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLO + MiDaS Obstacle Avoidance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()