#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Initialize the ROS node so it can talk to the rest of the system.
rospy.init_node("vision_decision_node")

decision_pub = rospy.Publisher("/obstacle_decision", String, queue_size=10)
rate = rospy.Rate(10) # Run at 10 Hz

VIDEO_PATH = "/home/alex/MiDaS/test_video/yolo3.mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "dog", "chair", "bench"
}

# Safety thresholds
CORRIDOR_STOP_DIST = 0.35
SIDE_DIST = 0.45

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
midas.to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform

yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
rospy.loginfo("Vision node started")

while not rospy.is_shutdown() and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # DEPTH ESTIMATION
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

    # Normalize depth to a 0-1 range for easier math.
    depth = pred.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # This represents the immediate path in front of the robot wheels.
    cy1, cy2 = int(H * 0.65), int(H * 0.9)
    cx1, cx2 = int(W * 0.45), int(W * 0.55)
    corridor = depth[cy1:cy2, cx1:cx2]
    
    corridor = corridor[corridor > 0]
    corridor_dist = np.median(corridor) if corridor.size else 1.0

    results = yolo(frame, verbose=False)[0]
    
    # We will store the distances of objects found in Left, Center, or Right zones.
    zone_depths = {"left": [], "center": [], "right": []}

    for box in results.boxes:
        label = yolo.names[int(box.cls[0])]
        conf = float(box.conf[0])
        
        # Skip objects we don't care about or aren't sure about.
        if label not in OBSTACLE_CLASSES or conf < 0.4:
            continue

        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # "Overlay" the box onto our depth map to find out how far this specific object is.
        roi = depth[y1:y2, x1:x2]
        roi = roi[roi > 0]
        if roi.size == 0:
            continue

        dist = np.median(roi)
        
        # Determine which "lane" the object is in based on its center X coordinate.
        cx = (x1 + x2) / 2
        zone = "left" if cx < W*0.33 else "center" if cx < W*0.66 else "right"
        zone_depths[zone].append(dist)

    decision = "FORWARD"
    
    if corridor_dist < CORRIDOR_STOP_DIST:
        decision = "STOP"
    else:
        # If a zone is empty, we consider it safe (distance = 1.0).
        min_dist = {z: min(v) if v else 1.0 for z, v in zone_depths.items()}

        if min_dist["center"] < SIDE_DIST:
            decision = "TURN_LEFT" if min_dist["left"] > min_dist["right"] else "TURN_RIGHT"
        elif min_dist["left"] < SIDE_DIST:
            decision = "TURN_RIGHT"
        elif min_dist["right"] < SIDE_DIST:
            decision = "TURN_LEFT"

    decision_pub.publish(decision)
    rospy.loginfo(decision)
    rate.sleep()

cap.release()
