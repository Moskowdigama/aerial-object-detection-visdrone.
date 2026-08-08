
# 🚁 Aerial Object Detection & Drone Analytics Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aerial-object-detection-visdrone-snzjyktmv4dowevckrspuw.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)

An end-to-end Computer Vision and Edge Intelligence application fine-tuning **YOLO11** on the **VisDrone-DET** dataset. The platform is designed to process high-altitude aerial drone telemetry, detecting small, overlapping, and densely clustered targets in real time.

🔗 **Live Interactive App:** [VisDrone Aerial Analytics Engine](https://aerial-object-detection-visdrone-snzjyktmv4dowevckrspuw.streamlit.app/)

---

## 💡 Project Overview

Detecting objects from Unmanned Aerial Vehicles (UAVs) introduces distinct computer vision challenges: extreme scale variations, high-density clustering, dynamic camera angles, and small object sizes (e.g., pedestrians or bicycles occupying a few pixels).

This project fine-tunes a lightweight **YOLO11-nano** detector across 10 distinct ground object categories collected from drone-captured video streams. The accompanying Streamlit web dashboard allows users to adjust detection confidence, filter IoU thresholds, inspect class-wise fleet counts, and stress-test aerial telemetry.

---

## 🌟 Key Features & Web App Highlights

* **Multi-Class Small Object Detection:** Fine-tuned to detect 10 distinct aerial classes: `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, and `motor`.
* **Dynamic Confidence & IoU Tuning:** Interactive sidebar sliders to adjust non-maximum suppression (NMS) confidence and overlap thresholds on the fly.
* **Automated Fleet & Crowd Analytics:** Instant breakdown of detected targets with per-class instance counters and high-density object metrics.
* **Lightweight Edge-Ready Deployment:** Operates on an optimized ~2.58M parameter model yielding low-latency inference (~1.5 ms on GPU / ~7.8 ms on CPU).

---

## 🌐 Real-World Application Use Cases

### 1. Smart City Traffic & Parking Management
* **Use Case:** Monitoring urban intersections, highway bottlenecks, and open parking lots using aerial drone feeds.
* **Impact:** Provides automated vehicle counting (`car`, `bus`, `truck`, `van`), detects traffic congestion patterns, and identifies illegal parking or stalled vehicles in real time.

### 2. Search & Rescue (SAR) & Disaster Response
* **Use Case:** Locating missing persons or survivors in flood zones, forests, or collapsed infrastructure.
* **Impact:** Distinguishes human signatures (`pedestrian`, `people`) from high-altitude imagery where ground visibility is obscured or inaccessible to ground teams.

### 3. Perimeter Security & Critical Infrastructure Surveillance
* **Use Case:** Autonomous drone patrols around borders, industrial complexes, and oil pipelines.
* **Impact:** Triggers automated alerts upon detecting unauthorized human or vehicle entries into restricted zones, reducing reliance on manual camera monitoring.

### 4. Smart Micro-Mobility & Urban Planning
* **Use Case:** Tracking pedestrian movements and light vehicle distribution (`bicycle`, `motor`, `tricycle`).
* **Impact:** Helps urban planners map bike lane usage, evaluate pedestrian foot-traffic density, and optimize city transit infrastructure.

---

## 📊 Architectural Benchmarks & Class Metrics

Fine-tuned on the VisDrone2019-DET dataset over 10 epochs at $640 \times 640$ resolution:

| Model Architecture | Parameters | Input Res | $mAP_{50}$ | $mAP_{50-95}$ | Inference Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO11-nano (Fine-Tuned)** | **2.58 M** | **640x640** | **23.3%** | **13.0%** | **~1.5 ms (GPU)** |

### Per-Class Detection Insights
* **High Accuracy Categories:** Larger vehicles with distinct visual boundaries achieved high precision (e.g., **Car ($mAP_{50} = 66.7\%$)** and **Bus ($mAP_{50} = 32.6\%$)**).
* **Small Target Challenges:** Extremely small or high-altitude targets like **Bicycles ($mAP_{50} = 1.7\%$)** and **Awning-Tricycles ($mAP_{50} = 5.2\%$)** highlight the inherent difficulty of pixel-deprived aerial object detection.

---

## 🛠️ Local Setup & Installation

To run the application locally:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/moskowdigama/aerial-object-detection-visdrone.git](https://github.com/moskowdigama/aerial-object-detection-visdrone.git)
   cd aerial-object-detection-visdrone

 * Create a Virtual Environment:
   python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

 * Install Dependencies:
   pip install -r requirements.txt

 * Launch Streamlit App:
   streamlit run app.py

📁 Repository Structure
├── app.py                      # Streamlit interactive web application
├── best_visdrone_yolo11n.pt    # Fine-tuned YOLO11 model weights
├── sample_drone_val.jpg        # Sample aerial validation image
├── packages.txt                # Linux system dependencies (libgl1)
├── requirements.txt            # Python dependencies (ultralytics-opencv-headless, etc.)
└── README.md                   # Project documentation


