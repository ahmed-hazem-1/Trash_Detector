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
import requests
import gdown
from pathlib import Path

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
    .download-progress {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_DRIVE_URL = "https://drive.google.com/file/d/1eP2sxQtSBb3nAC7kfwT8rQiYu0NLjFDm/view?usp=drive_link"
MODEL_FILENAME = "best_waste_detection_model.pt"
MODELS_DIR = Path("./models")

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
if 'model_downloaded' not in st.session_state:
    st.session_state.model_downloaded = False
if 'download_progress' not in st.session_state:
    st.session_state.download_progress = 0

def extract_file_id_from_drive_url(url):
    """Extract file ID from Google Drive URL"""
    if 'drive.google.com' in url:
        if '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
    return None

def download_model_from_drive():
    """Download model from Google Drive using gdown"""
    try:
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / MODEL_FILENAME
        
        # Check if model already exists
        if model_path.exists():
            st.info(f"✅ Model already exists at {model_path}")
            return str(model_path)
        
        # Extract file ID from Google Drive URL
        file_id = extract_file_id_from_drive_url(MODEL_DRIVE_URL)
        if not file_id:
            st.error("❌ Could not extract file ID from Google Drive URL")
            return None
        
        # Download using gdown
        st.info("📥 Downloading model from Google Drive...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use gdown to download
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            # Download with progress
            output_path = gdown.download(download_url, str(model_path), quiet=False)
            
            if output_path and os.path.exists(output_path):
                progress_bar.progress(100)
                status_text.success("✅ Model downloaded successfully!")
                st.session_state.model_downloaded = True
                return str(model_path)
            else:
                st.error("❌ Failed to download model")
                return None
                
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            # Try alternative method
            return download_model_requests_fallback(file_id, model_path)
            
    except Exception as e:
        st.error(f"❌ Error in download process: {str(e)}")
        return None

def download_model_requests_fallback(file_id, model_path):
    """Fallback method using requests (for smaller files)"""
    try:
        st.info("🔄 Trying alternative download method...")
        
        # Direct download URL for Google Drive
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        response = requests.get(download_url, stream=True)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Downloaded: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB")
            
            progress_bar.progress(100)
            status_text.success("✅ Model downloaded successfully!")
            st.session_state.model_downloaded = True
            return str(model_path)
        else:
            st.error(f"❌ HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"❌ Fallback download failed: {str(e)}")
        return None

def load_model_from_path(model_path):
    """Load YOLO model from path"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
            
        with st.spinner("🔄 Loading YOLO model..."):
            model = YOLO(model_path)
            st.success("✅ Model loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Header
st.markdown('<h1 class="main-header">🎯 YOLOv11m-OBB Real-Time Detection</h1>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model download section
    st.subheader("🤖 Model Setup")
    
    # Show model status
    model_path = MODELS_DIR / MODEL_FILENAME
    if model_path.exists():
        st.success("✅ Model file found locally")
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        st.info(f"📁 Size: {model_size_mb:.1f} MB")
    else:
        st.warning("⚠️ Model not found locally")
    
    # Download and load model button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Model"):
            downloaded_path = download_model_from_drive()
            if downloaded_path:
                st.session_state.model_downloaded = True
    
    with col2:
        if st.button("🔄 Load Model"):
            if model_path.exists():
                st.session_state.model = load_model_from_path(str(model_path))
            else:
                st.error("❌ Please download the model first")
    
    # Alternative upload option
    st.subheader("📁 Alternative Upload")
    model_file = st.file_uploader(
        "Upload your own model (.pt file)",
        type=['pt'],
        help="Upload a custom YOLOv11m-OBB model file"
    )
    
    if model_file is not None and st.button("Upload & Load"):
        with st.spinner("Loading uploaded model..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    tmp_file.write(model_file.read())
                    temp_path = tmp_file.name
                
                st.session_state.model = YOLO(temp_path)
                os.unlink(temp_path)
                st.success("✅ Uploaded model loaded!")
                
            except Exception as e:
                st.error(f"❌ Error loading uploaded model: {str(e)}")
    
    st.divider()
    
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
        index=1,
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
    
    # Model info section
    if st.session_state.model is not None:
        st.divider()
        st.subheader("📊 Model Information")
        
        with st.expander("View Details"):
            try:
                st.write(f"**Model Type:** {type(st.session_state.model).__name__}")
                st.write(f"**Classes:** {list(st.session_state.model.names.values())}")
                st.write(f"**Number of Classes:** {len(st.session_state.model.names)}")
                
                # Device info
                try:
                    model_device = next(st.session_state.model.model.parameters()).device
                    st.write(f"**Model Device:** {model_device}")
                except:
                    st.write("**Model Device:** Not available")
                    
            except Exception as e:
                st.error(f"Error getting model info: {str(e)}")

# Model status in main area
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.model is not None:
        st.success("✅ Model loaded and ready for detection!")
    else:
        st.warning("⚠️ Model not loaded")
        if not model_path.exists():
            st.info("💡 Click 'Download Model' in the sidebar to get started")
        else:
            st.info("💡 Click 'Load Model' in the sidebar to load the downloaded model")

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
            cv2.putText(img, "No Model Loaded", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "Download & Load Model First", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
            for detection in st.session_state.detections[:5]:
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
        
        # System info
        st.markdown("### 🔧 System Info")
        if torch.cuda.is_available():
            st.success("✅ CUDA Available")
            st.info(f"GPU: {torch.cuda.get_device_name()}")
        else:
            st.warning("⚠️ CPU Only")
        
        # Model file info
        if model_path.exists():
            model_size = model_path.stat().st_size / (1024 * 1024)
            st.info(f"📁 Model Size: {model_size:.1f} MB")

else:
    st.error("❌ Model not loaded")
    st.info("👈 Use the sidebar to download and load the model")

# Instructions
with st.expander("📖 Instructions & Setup"):
    st.markdown("""
    ### Quick Start:
    
    1. **Download Model**: Click "📥 Download Model" in the sidebar
    2. **Load Model**: Click "🔄 Load Model" after download completes
    3. **Start Camera**: Click the START button above the video feed
    4. **View Results**: See real-time detections with oriented bounding boxes
    
    ### Features:
    - **Automatic Download**: Downloads model from Google Drive automatically
    - **Real-time Detection**: Live camera feed with YOLO inference
    - **Performance Optimization**: Process every Nth frame to improve speed
    - **Oriented Bounding Boxes**: Full OBB support for rotated objects
    - **GPU Acceleration**: Automatic CUDA detection and usage
    - **Customizable Settings**: Adjust confidence and processing rate
    - **Live Statistics**: Real-time FPS and detection information
    
    ### Model Information:
    - **Source**: Google Drive (automatic download)
    - **Type**: YOLOv11m-OBB (Oriented Bounding Boxes)
    - **Application**: Waste detection
    - **Size**: ~40-60 MB (approximate)
    
    ### Tips for Better Performance:
    - **GPU Usage**: Enable CUDA if available for faster inference
    - **Frame Skipping**: Increase frame skip (2-3) for better performance
    - **Confidence**: Lower threshold = more detections, higher threshold = fewer false positives
    - **Resolution**: Lower camera resolution improves processing speed
    
    ### Troubleshooting:
    - **Download Issues**: Check internet connection and try again
    - **Loading Errors**: Ensure the model file downloaded completely
    - **Camera Issues**: Allow camera permissions in your browser
    - **Performance**: Try increasing frame skip or using CPU if GPU has issues
    """)

# Requirements notice
with st.expander("📦 Dependencies"):
    st.markdown("""
    ### Required Python Packages:
    ```bash
    pip install streamlit
    pip install ultralytics
    pip install opencv-python
    pip install torch torchvision
    pip install streamlit-webrtc
    pip install gdown
    pip install requests
    pip install pillow
    pip install numpy
    ```
    
    ### Additional Notes:
    - **gdown**: Used for downloading from Google Drive
    - **torch**: GPU support requires CUDA-compatible version
    - **streamlit-webrtc**: Handles real-time camera streaming
    - **ultralytics**: YOLOv11 implementation
    """)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit and YOLOv11m-OBB | "
    f"Current Device: {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'} | "
    f"Model Status: {'✅ Loaded' if st.session_state.model else '⚠️ Not Loaded'}"
)
