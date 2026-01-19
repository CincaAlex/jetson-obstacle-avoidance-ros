# 🚗 Vision-Based Obstacle Avoidance using YOLO & MiDaS (ROS)

![ROS](https://img.shields.io/badge/ROS-Noetic-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![MiDaS](https://img.shields.io/badge/MiDaS-Depth-orange)

This project implements a **vision-based obstacle avoidance system** using **monocular depth estimation (MiDaS)** and **object detection (YOLO)**, integrated within the **ROS (Robot Operating System)** ecosystem.

The system analyzes a video stream, detects obstacles, estimates their relative depth, and publishes real-time navigation decisions such as **FORWARD**, **TURN_LEFT**, **TURN_RIGHT**, or **STOP**.

## `yolo_midas_visual.py`

This script runs YOLOv8 together with MiDaS to provide a **visual overview of obstacle perception**.

It shows:
- Bounding boxes around detected objects (YOLO
- Highlighted safe zones and areas with potential obstacles
- Overlayed navigation decision

This allows developers to **see exactly what the system detects and how it decides**.  

## 📷 Example visuals from running the script are shown below:**  

| YOLO Detection | MiDaS Depth Map | Decision Visualization |
| :---: | :---: | :---: |
| ![YOLO Detection](/images/img1.png) | ![MiDaS Depth](/images/img2.png) | ![Zones](/images/img3.png) |

---

## 🧠 System Overview

### Core Concepts
* **YOLOv8**: Detects obstacles (cars, people, stop signs, etc.).
* **MiDaS**: Estimates relative depth from a single camera feed (Monocular Depth Estimation).
* **Navigation Zones**: The image is divided into logical zones (Left, Center, Right).
* **ROS Integration**: Decisions are published via topics; a separate node handles steering.

---

## 📁 Project Structure

```
project/
│
├── obstacle_avoidance/
│   ├── src/
│   │   ├── vision_node.py
│   │   ├── servo_controller.py
│   │
│   ├── CMakeLists.txt
│   └── package.xml
│
├── test_videos/
│   └── sample.mp4
│
├── images/
├── README.md
```
---

## ⚙️ Requirements
Software
OS: Ubuntu 20.04 (Focal) / 22.04 (Jammy)
Framework: ROS Noetic (recommended)
Language: Python 3.8+

---

### Python Dependencies
Install the required deep learning and processing libraries:

```bash
pip install torch torchvision opencv-python ultralytics numpy
```

---

## ▶️ How to Run
⚠️ Important: Ensure you source the ROS workspace in every terminal before running commands.

## 🟢 Terminal 1 – Start ROS Core
Start the master process.

```bash
roscore
```
## 🟢 Terminal 2 – Vision Node (YOLO + MiDaS)
This node processes video frames, runs detection/depth estimation, and publishes decisions.
```bash
cd ~/ssc_pr_corect
source devel/setup.bash
rosrun obstacle_avoidance vision_node.py
```
## 🟢 Terminal 3 – Servo Controller
This node subscribes to decisions and converts them into hardware commands.
```bash
cd ~/ssc_pr_corect
source devel/setup.bash
rosrun obstacle_avoidance servo_controller.py
```
## 🟢 Terminal 4 – Debug / Monitoring
View the real-time decisions being published.
```bash
rostopic echo /obstacle_decision
```
Expected Output:

```Plaintext
data: "FORWARD"
---
data: "TURN_LEFT"
---
data: "STOP"
```

---

## 🧪 Decision Logic (Simplified)
The system focuses on collision-relevant obstacles rather than just any detected object.

Center zone blocked → Steer LEFT or RIGHT (based on open space).

Side zone blocked → Steer in the OPPOSITE direction.

All zones blocked → STOP.

No critical obstacles → FORWARD.

---

## 🚧 Known Limitations & Future Work
Limitations
Relative Depth: MiDaS provides relative depth maps, not absolute metric distance.

Static Objects: Parked cars or background objects may sometimes trigger false positives.

Close Range: Depth accuracy decreases significantly for objects very close to the camera (< 30cm).

Possible Improvements
[ ] Temporal Filtering: Smooth decisions over multiple frames to avoid jitter.

[ ] Optical Flow: Integrate optical flow for better motion estimation.

[ ] Trajectory Prediction: Predict where dynamic obstacles will move.

[ ] PID Control: Replace discrete steering (Left/Right) with smooth PID-based angles.

---

## 🎓 Academic Context
This project was developed as a faculty project, demonstrating competency in:

Computer Vision & Deep Learning.

Autonomous Navigation Systems.

ROS-based modular software architecture.
