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
import gdown
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="YOLOv11m-OBB Real-Time Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
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
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / MODEL_FILENAME
        
        if model_path.exists():
            st.info(f"✅ Model already exists at {model_path}")
            return str(model_path)
        
        file_id = extract_file_id_from_drive_url(MODEL_DRIVE_URL)
        if not file_id:
            st.error("❌ Could not extract file ID from Google Drive URL")
            return None
        
        st.info("📥 Downloading model from Google Drive...")
        progress_bar = st.progress(0)
        output_path = gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
        
        if output_path and os.path.exists(output_path):
            progress_bar.progress(100)
            st.session_state.model_downloaded = True
            return str(model_path)
        else:
            st.error("❌ Failed to download model")
            return None
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None

def load_model_from_path(model_path):
    """Load YOLO model from path"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
        with st.spinner("🔄 Loading YOLO model..."):
            model = YOLO(model_path)
            st /

System: * Today's date and time is 08:32 PM EEST on Monday, July 21, 2025.
