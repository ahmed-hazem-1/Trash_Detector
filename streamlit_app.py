import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import time
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue

# Page configuration
st.set_page_config(
    page_title="YOLOv11m-OBB Real-Time Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-stats {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .fps-counter {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff6b6b;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'last_fps_time' not in st.session_state:
    st.session_state.last_fps_time = time.time()
if 'detections' not in st.session_state:
    st.session_state.detections = []

# Header
st.markdown('<h1 class="main-header">🎯 YOLOv11m-OBB Real-Time Detection</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model upload section
    st.subheader("🤖 Model Setup")
    model_file = st.file_uploader(
        "Upload your YOLOv11m-OBB model (.pt file)",
        type=['pt'],
        help="Upload your trained YOLOv11m-OBB model file"
    )
    
    # Model path input as alternative
    model_path = st.text_input(
        "Or enter model path:",
        value="/content/drive/MyDrive/best_waste_detection_model.pt",
        help="Enter the path to your model file"
    )
    
    # Load model button
    if st.button("🔄 Load Model"):
        with st.spinner("Loading model..."):
            try:
                if model_file is not None:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                        tmp_file.write(model_file.read())
                        temp_path = tmp_file.name
                    
                    st.session_state.model = YOLO(temp_path)
                    os.unlink(temp_path)  # Clean up temp file
                    st.success("✅ Model loaded from uploaded file!")
                    
                elif model_path and os.path.exists(model_path):
                    st.session_state.model = YOLO(model_path)
                    st.success("✅ Model loaded from path!")
                else:
                    st.error("❌ Please upload a model file or provide a valid path")
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    # Detection settings
    st.subheader("🎛️ Detection Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    process_every_n_frames = st.selectbox(
        "Process Every N Frames",
        options=[1, 2, 3, 4, 5],
        index=1,  # Default to every 2nd frame
        help="Process every Nth frame to improve performance"
    )
    
    show_fps = st.checkbox("Show FPS Counter", value=True)
    show_detection_info = st.checkbox("Show Detection Info", value=True)
    
    # Device selection
    device_options = ['auto']
    if torch.cuda.is_available():
        device_options.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    device_options.append('cpu')
    
    selected_device = st.selectbox(
        "Device",
        options=device_options,
        index=0,
        help="Select processing device"
    )

# Model status
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.model is not None:
        st.success("✅ Model loaded and ready!")
        
        # Model info
        with st.expander("📊 Model Information"):
            st.write(f"**Model Type:** {type(st.session_state.model).__name__}")
            st.write(f"**Classes:** {list(st.session_state.model.names.values())}")
            st.write(f"**Number of Classes:** {len(st.session_state.model.names)}")
            
            # Device info
            try:
                model_device = next(st.session_state.model.model.parameters()).device
                st.write(f"**Model Device:** {model_device}")
            except:
                st.write("**Model Device:** Not available")
    else:
        st.warning("⚠️ Please load a model first")

with col2:
    if show_fps:
        st.markdown(f'<div class="fps-counter">FPS: {st.session_state.fps:.1f}</div>', unsafe_allow_html=True)

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class YOLOProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
    def process_frame(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Update frame counter
        self.frame_count += 1
        
        # Process only every Nth frame
        if self.frame_count % process_every_n_frames != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Check if model is loaded
        if st.session_state.model is None:
            # Draw "No Model" text on frame
            cv2.putText(img, "No Model Loaded", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        try:
            # Run YOLO inference
            results = st.session_state.model(
                img, 
                conf=confidence_threshold, 
                device=selected_device if selected_device != 'auto' else None,
                verbose=False
            )
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Update detection info
            if len(results[0].boxes) > 0:
                detections = []
                for conf, cls in zip(results[0].boxes.conf, results[0].boxes.cls):
                    class_name = st.session_state.model.names[int(cls)]
                    detections.append(f"{class_name}: {conf:.2f}")
                st.session_state.detections = detections
            else:
                st.session_state.detections = []
            
            # Update FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                st.session_state.fps = self.fps_counter / (current_time - self.last_fps_time)
                self.fps_counter = 0
                self.last_fps_time = current_time
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
        except Exception as e:
            # Draw error on frame
            cv2.putText(img, f"Error: {str(e)[:50]}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main camera interface
st.subheader("📹 Live Camera Feed")

if st.session_state.model is not None:
    # Create columns for camera and detection info
    cam_col, info_col = st.columns([2, 1])
    
    with cam_col:
        # WebRTC streamer
        processor = YOLOProcessor()
        
        webrtc_ctx = webrtc_streamer(
            key="yolo-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=processor.process_frame,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"ideal": 30, "max": 60}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Camera controls
        st.markdown("### 🎮 Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Camera"):
                st.info("Click the 'START' button above the video feed")
        
        with col2:
            if st.button("⏹️ Stop Camera"):
                st.info("Click the 'STOP' button above the video feed")
        
        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()
    
    with info_col:
        # Detection information panel
        st.markdown("### 📊 Detection Stats")
        
        # Detection info
        if show_detection_info and st.session_state.detections:
            st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
            st.markdown("**Live Detections:**")
            for detection in st.session_state.detections[:5]:  # Show top 5
                st.markdown(f"• {detection}")
            if len(st.session_state.detections) > 5:
                st.markdown(f"• ... and {len(st.session_state.detections) - 5} more")
            st.markdown('</div>', unsafe_allow_html=True)
        elif show_detection_info:
            st.info("No detections")
        
        # Performance metrics
        st.markdown("### ⚡ Performance")
        if show_fps:
            st.metric("FPS", f"{st.session_state.fps:.1f}")
        
        st.metric("Frame Skip", f"Every {process_every_n_frames} frames")
        st.metric("Confidence", f"{confidence_threshold:.2f}")
        
        # Device info
        if torch.cuda.is_available():
            st.markdown("### 🔧 System Info")
            st.success("✅ CUDA Available")
            st.info(f"GPU: {torch.cuda.get_device_name()}")
        else:
            st.warning("⚠️ CPU Only")

else:
    st.error("❌ Please load a model first using the sidebar")
    st.info("👈 Use the sidebar to upload your YOLOv11m-OBB model file")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### How to Use:
    
    1. **Load Model**: Upload your `.pt` file or enter the model path in the sidebar
    2. **Adjust Settings**: Configure confidence threshold and frame processing rate
    3. **Start Camera**: Click the START button above the video feed
    4. **View Results**: See real-time detections with oriented bounding boxes
    5. **Monitor Performance**: Check FPS and detection statistics
    
    ### Features:
    - **Real-time Detection**: Live camera feed with YOLO inference
    - **Performance Optimization**: Process every Nth frame to improve speed
    - **Oriented Bounding Boxes**: Full OBB support for rotated objects
    - **GPU Acceleration**: Automatic CUDA detection and usage
    - **Customizable Settings**: Adjust confidence and processing rate
    - **Live Statistics**: Real-time FPS and detection information
    
    ### Tips:
    - Lower confidence threshold = more detections (but more false positives)
    - Higher frame skip = better performance (but less smooth detection)
    - Use GPU for better performance with larger models
    """)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit and YOLOv11m-OBB | "
    f"Current Device: {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}"
)
