import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import time
from ultralytics import YOLO
import gdown

# Page configuration
st.set_page_config(page_title="YOLO Trash Detector", layout="centered")

# Initialize session state for tracking values across reruns
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'detections' not in st.session_state:
    st.session_state.detections = []

# Create directory for models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Title
st.title("YOLO Trash Detector (CPU)")

# Function to extract file ID from Google Drive link
def extract_file_id_from_drive_url(url):
    if 'drive.google.com' in url:
        if '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
    return None

# --- Sidebar: Model selection and settings ---
st.sidebar.header("Model Setup")

# Model upload or download
model_file = st.sidebar.file_uploader("Upload YOLO Model (.pt)", type=["pt"])
drive_url = st.sidebar.text_input("Or paste Google Drive link to download model")

model_path = None

if model_file is not None:
    # Save uploaded model
    model_path = os.path.join(MODEL_DIR, "uploaded_model.pt")
    with open(model_path, "wb") as f:
        f.write(model_file.read())
    st.sidebar.success("Model uploaded successfully!")
elif drive_url:
    file_id = extract_file_id_from_drive_url(drive_url)
    if file_id:
        model_path = os.path.join(MODEL_DIR, "downloaded_model.pt")
        if not os.path.exists(model_path):
            st.sidebar.info("Downloading model from Google Drive...")
            try:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
                st.sidebar.success("Model downloaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")
                model_path = None
    else:
        st.sidebar.error("Invalid Google Drive link.")

# Settings
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
frame_skip = st.sidebar.selectbox("Process Every Nth Frame (for video)", [1, 2, 3, 4, 5], index=2)
show_fps = st.sidebar.checkbox("Show FPS Counter (for video)", value=True)

# --- Load model ---
model = None
if model_path and os.path.exists(model_path):
    try:
        model = YOLO(model_path)
        # No CUDA, runs on CPU by default
        st.sidebar.success("Model loaded on CPU!")
        st.session_state.model = model
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.session_state.model = None
else:
    st.session_state.model = None

# --- Main: Mode selection ---
mode = st.selectbox("Choose Detection Mode", ["Upload Image", "Upload Video"])  # Removed Real-Time Camera

if model is None:
    st.warning("Please upload or download a model first.")
    st.stop()

# Inference function
def run_inference(img):
    results = model(img, conf=confidence, imgsz=256, verbose=False)  # Removed device="cuda"
    annotated = results[0].plot()
    return annotated, results

# --- Mode: Upload Image ---
if mode == "Upload Image":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img)
        img_np = np.array(img)
        
        # Handle different image formats
        if len(img_np.shape) == 2:  # Grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            
        # Run detection
        with st.spinner("Detecting objects..."):
            annotated, results = run_inference(img_np)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
        
        # Display detections
        st.subheader("Detected Objects")
        boxes = getattr(results[0], 'boxes', None)
        if boxes is not None and hasattr(boxes, 'conf') and hasattr(boxes, 'cls'):
            for conf, cls in zip(boxes.conf, boxes.cls):
                class_name = model.names[int(cls)] if hasattr(model, "names") else str(cls)
                st.write(f"{class_name}: {conf:.2f}")
        else:
            st.write("No objects detected.")

# --- Mode: Upload Video ---
elif mode == "Upload Video":
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_vid:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.close()
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error opening video file")
        else:
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video - use avc1 codec for better browser compatibility
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))
            
            # Show debug option
            debug_mode = st.checkbox("Enable debug mode (show detection frames)")
            
            # Process frames
            frame_count = 0
            progress = st.progress(0)
            detections = []
            all_frames_with_detections = []
            
            with st.spinner("Processing video..."):
                start_time = time.time()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    progress.progress(min(frame_count / total_frames, 1.0))
                    
                    # Process every nth frame
                    if frame_count % frame_skip == 0:
                        # Run detection
                        annotated, results = run_inference(frame)
                        
                        # Log the detection details
                        st.session_state.frame_being_processed = frame:Stroke

                        # Check if there are any detections in this frame
                        boxes = getattr(results[0], 'boxes', None)
                        frame_has_detections = False
                        
                        if boxes is not None and hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                            frame_has_detections = True
                            # Add frame number to the annotated image
                            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Save frames with detections for debugging
                            all_frames_with_detections.append((frame_count, annotated.copy()))
                            
                            # Collect detections
                            for conf, cls in zip(boxes.conf, boxes.cls):
                                class_name = model.names[int(cls)] if hasattr(model, "names") else str(cls)
                                detections.append(f"Frame {frame_count}: {class_name} ({conf:.2f})")
                        
                        # Write the annotated frame
                        out.write(annotated)
                    else:
                        out.write(frame)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                fps_processing = frame_count / processing_time
                
                cap.release()
                out.release()
            
            # Display processing stats
            st.success(f"Video processed! {frame_count} frames in {processing_time:.1f}s ({fps_processing:.1f} FPS)")
            
            # Display detections before video to confirm they exist
            if detections:
                st.subheader(f"Detections Found: {len(detections)}")
                unique_detections = list(dict.fromkeys(detections))[:20]  # Show up to 20 unique detections
                for d in unique_detections:
                    st.write(d)
                
                # Show sample frames with detections if debug mode is on
                if debug_mode and all_frames_with_detections:
                    st.subheader("Sample Frames with Detections")
                    # Show up to 5 frames with detections
                    for i, (frame_num, frame) in enumerate(all_frames_with_detections[:5]):
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                caption=f"Frame {frame_num} with detections", 
                                use_column_width=True)
            else:
                st.warning("No objects detected in the video. Try lowering the confidence threshold.")
            
            # Display processed video
            st.subheader("Processed Video")
            st.video(out_path)
            
            # Cleanup
            os.unlink(tfile.name)
            os.unlink(out_path)

# --- Model Status ---
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.model is not None:
        st.success("✅ Model loaded and ready for detection!")
    else:
        st.warning("⚠️ Model not loaded")
        if model_path is None or not os.path.exists(model_path):
            st.info("💡 Upload a model or provide a Google Drive link in the sidebar.")

with col2:
    if show_fps and mode == "Upload Video":
        st.markdown(f'<div class="fps-counter">FPS: {st.session_state.fps:.1f}</div>', unsafe_allow_html=True)

# Note about Real-Time Camera
st.info("Real-Time Camera mode is not supported on Streamlit Community Cloud due to WebRTC limitations. Use image or video upload modes instead.")

# Model info section
if st.session_state.model is not None:
    st.divider()
    st.subheader("📊 Model Information")
    
    with st.expander("View Details"):
        try:
            st.write(f"**Model Type:** {type(st.session_state.model).__name__}")
            model_names = getattr(st.session_state.model, "names", None)
            if isinstance(model_names, dict):
                st.write(f"**Classes:** {list(model_names.values())}")
                st.write(f"**Number of Classes:** {len(model_names)}")
            else:
                st.write("**Classes:** Not available")
                st.write("**Number of Classes:** Not available")
            st.write("**Model Device:** CPU")
        except Exception as e:
            st.error(f"Error getting model info: {str(e)}")
