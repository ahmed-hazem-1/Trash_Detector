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
            try:
                model.to('cuda')
            except:
                st.warning("⚠️ CUDA not available, falling back to CPU")
                model.to('cpu')
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
        index=2,  # Default to 3 to reduce resource usage
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
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]}
    ]
})

class YOLOProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
    def recv(self, frame):
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            print(f"Frame received: {img.shape}")  # Debug
            
            # Update frame counter
            self.frame_count += 1
            
            # Process only every Nth frame
            if self.frame_count % process_every_n_frames != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Check if model is loaded
            if st.session_state.model is None:
                st.warning("⚠️ Model not loaded, using placeholder frame")
                cv2.putText(img, "No Model Loaded", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, "Download & Load Model First", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Run YOLO inference
            try:
                results = st.session_state.model(
                    img, 
                    conf=confidence_threshold, 
                    imgsz=256,
                    device=selected_device if selected_device != 'auto' else None,
                    verbose=False
                )
            except Exception as e:
                st.error(f"⚠️ YOLO inference error: {str(e)}")
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Draw detections (oriented bounding boxes)
            annotated_frame = results[0].plot()
            print(f"Frame processed: {annotated_frame.shape}")  # Debug
            
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
            
            print(f"Detections: {len(st.session_state.detections)}")  # Debug
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception as e:
            print(f"YOLOProcessor error: {str(e)}")  # Debug
            cv2.putText(img, f"Error: {str(e)[:50]}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main interface
st.subheader("📹 Live Camera Feed")

if st.session_state.model is not None:
    try:
        # Create columns for camera and detection info
        cam_col, info_col = st.columns([2, 1])
        
        with cam_col:
            # WebRTC streamer with fallback
            webrtc_ctx = webrtc_streamer(
                key="yolo-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=YOLOProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 320, "ideal": 320, "max": 640},
                        "height": {"min": 240, "ideal": 240, "max": 480},
                        "frameRate": {"ideal": 15, "max": 30}
                    },
                    "audio": False
                },
                async_processing=True,
                on_error=lambda exc: st.error(f"WebRTC error: {str(exc)}"),  # Handle WebRTC errors
            )
            if webrtc_ctx.state.playing:
                st.write("WebRTC is active")
            else:
                st.warning("⚠️ WebRTC not active. Check camera permissions or network connection.")
            
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
            
            if model_path.exists():
                model_size = model_path.stat().st_size / (1024 * 1024)
                st.info(f"📁 Model Size: {model_size:.1f} MB")
    except Exception as e:
        st.error(f"❌ Camera feed initialization failed: {str(e)}")
else:
    st.error("❌ Model not loaded")
    st.info("👈 Use the sidebar to download and load the model")

# New section for image/video upload
st.subheader("📁 Upload Image or Video for Detection")

if st.session_state.model is not None:
    uploaded_file = st.file_uploader(
        "Upload an image or video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi'],
        help="Upload an image (.jpg, .png) or video (.mp4, .avi) for YOLO detection"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with st.spinner("Processing uploaded file..."):
            try:
                # Handle image
                if file_extension in ['jpg', 'jpeg', 'png']:
                    # Read and process image
                    img = Image.open(uploaded_file)
                    img_array = np.array(img)
                    if len(img_array.shape) == 2:  # Convert grayscale to RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    elif img_array.shape[2] == 4:  # Convert RGBA to RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    
                    # Run YOLO inference
                    results = st.session_state.model(
                        img_array,
                        conf=confidence_threshold,
                        imgsz=256,
                        device=selected_device if selected_device != 'auto' else None,
                        verbose=False
                    )
                    
                    # Draw detections
                    annotated_frame = results[0].plot()
                    annotated_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    
                    # Display results
                    st.image(annotated_img, caption="Detected Image", use_column_width=True)
                    
                    # Update detection info
                    detections = []
                    if len(results[0].boxes) > 0:
                        for conf, cls in zip(results[0].boxes.conf, results[0].boxes.cls):
                            class_name = st.session_state.model.names[int(cls)]
                            detections.append(f"{class_name}: {conf:.2f}")
                    
                    if detections:
                        st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                        st.markdown("**Image Detections:**")
                        for detection in detections[:5]:
                            st.markdown(f"• {detection}")
                        if len(detections) > 5:
                            st.markdown(f"• ... and {len(detections) - 5} more")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No detections in the image")
                
                # Handle video
                elif file_extension in ['mp4', 'avi']:
                    # Save uploaded video to temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    
                    # Open video
                    cap = cv2.VideoCapture(tfile.name)
                    if not cap.isOpened():
                        st.error("❌ Could not open video file")
                        os.unlink(tfile.name)
                    else:
                        # Get video properties
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback to 30 if FPS is invalid
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Check file size
                        file_size_mb = os.path.getsize(tfile.name) / (1024 * 1024)
                        st.info(f"📹 Uploaded video size: {file_size_mb:.2f} MB")
                        if file_size_mb > 50:
                            st.warning("⚠️ Video size exceeds 50MB, which may cause issues on some platforms.")
                        
                        # Create output video
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                        if not out.isOpened():
                            st.error("❌ Could not create output video")
                            cap.release()
                            os.unlink(tfile.name)
                        else:
                            # Process video frames
                            frame_count = 0
                            progress_bar = st.progress(0)
                            detections = []
                            
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                frame_count += 1
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                                
                                # Process only every Nth frame
                                if frame_count % process_every_n_frames == 0:
                                    # Run YOLO inference
                                    results = st.session_state.model(
                                        frame,
                                        conf=confidence_threshold,
                                        imgsz=256,
                                        device=selected_device if selected_device != 'auto' else None,
                                        verbose=False
                                    )
                                    
                                    # Draw detections
                                    annotated_frame = results[0].plot()
                                    
                                    # Collect detections
                                    if len(results[0].boxes) > 0:
                                        for conf, cls in zip(results[0].boxes.conf, results[0].boxes.cls):
                                            class_name = st.session_state.model.names[int(cls)]
                                            detections.append(f"{class_name}: {conf:.2f}")
                                    
                                    out.write(annotated_frame)
                                else:
                                    out.write(frame)
                            
                            cap.release()
                            out.release()
                            
                            # Verify output video
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                st.info(f"📹 Output video created: {output_path}, size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
                                st.video(output_path)
                            else:
                                st.error("❌ Output video is empty or not created properly")
                            
                            # Clean up
                            os.unlink(tfile.name)
                            if os.path.exists(output_path):
                                os.unlink(output_path)
                            
                            # Display detections
                            if detections:
                                st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                                st.markdown("**Video Detections (Sampled Frames):**")
                                unique_detections = list(dict.fromkeys(detections))  # Remove duplicates
                                for detection in unique_detections[:5]:
                                    st.markdown(f"• {detection}")
                                if len(unique_detections) > 5:
                                    st.markdown(f"• ... and {len(unique_detections) - 5} more")
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.info("No detections in the video")
                
                else:
                    st.error("❌ Unsupported file format")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                if 'tfile' in locals() and os.path.exists(tfile.name):
                    os.unlink(tfile.name)
                if 'output_path' in locals() and os.path.exists(output_path):
                    os.unlink(output_path)
else:
    st.error("❌ Model not loaded")
    st.info("👈 Use the sidebar to download and load the model")

# Footer
st.markdown("---")
st.markdown(
    f"Made with ❤️ using Streamlit and YOLOv11m-OBB | "
    f"Current Device: {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'} | "
    f"Model Status: {'✅ Loaded' if st.session_state.model else '⚠️ Not Loaded'}"
)
