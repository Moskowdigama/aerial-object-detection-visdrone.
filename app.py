import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Aerial Vision Engine - VisDrone", layout="wide")

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

@st.cache_resource
def load_model():
    return YOLO('best_visdrone_yolo11n.pt')

model = load_model()

st.title("🚁 Aerial Object Detection & Drone Analytics Engine")
st.caption("Fine-Tuned YOLO11 Architecture | VisDrone Dataset | Small Object Analytics")

st.sidebar.header("🕹️ Detection Controls")
conf_thresh = st.sidebar.slider("Confidence Threshold:", 0.1, 0.9, 0.25, 0.05)
iou_thresh = st.sidebar.slider("IoU Overlap Threshold:", 0.1, 0.9, 0.45, 0.05)

uploaded_file = st.sidebar.file_uploader("Upload Drone Image:", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1.5, 1.0])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    image = Image.open("sample_drone_val.jpg")
    st.info("ℹ️ Displaying default validation sample image. Upload your own image via the sidebar!")

with col1:
    st.subheader("📷 Raw Input Frame")
    st.image(image, use_container_width=True)

# Run Inference
results = model.predict(image, conf=conf_thresh, iou=iou_thresh)
r = results[0]

# Render Output Bounding Boxes
rendered_bgr = r.plot()
rendered_rgb = rendered_bgr[..., ::-1] # Convert BGR to RGB

with col1:
    st.subheader("🎯 Bounding Box Detections")
    st.image(rendered_rgb, use_container_width=True)

with col2:
    st.subheader("📊 Fleet Analytics & Counts")
    
    boxes = r.boxes
    if len(boxes) > 0:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        
        # Count per class
        counts = {}
        for cid in cls_ids:
            cname = CLASS_NAMES[cid]
            counts[cname] = counts.get(cname, 0) + 1
        
        st.metric("Total Objects Detected", len(boxes))
        
        # Display breakdown table
        st.write("### Object Class Breakdown:")
        for cname, count in counts.items():
            st.write(f"- **{cname.title()}:** {count}")
    else:
        st.warning("No objects detected at current confidence threshold.")
  
