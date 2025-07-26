import streamlit as st
import torch
import numpy as np
import tempfile
import os
import time
from PIL import Image
import gdown
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Try to import cv2 with fallback handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.error("Please install opencv-python-headless for cloud deployment")
    CV2_AVAILABLE = False

# Try to import YOLO with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    st.error(f"YOLO import failed: {e}")
    st.error("Please install ultralytics package")
    YOLO_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🗑️ YOLO Trash Detector", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .fps-counter {
        background-color: #1f4e79;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'model' not in st.session_state:
    st.session_state.model = None

# Check dependencies
if not CV2_AVAILABLE or not YOLO_AVAILABLE:
    st.stop()

# Title with styling
st.markdown('<div class="main-header"><h1>🗑️ YOLO Trash Detector</h1><p>Professional AI-Powered Waste Detection System</p></div>', unsafe_allow_html=True)

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.warning("⚠️ CUDA not available. Running on CPU (slower performance)")
else:
    st.success("✅ CUDA available. GPU acceleration enabled")

# Create directory for models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_file_id_from_drive_url(url):
    """Extract file ID from Google Drive URL"""
    if 'drive.google.com' in url:
        if '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
    return None

def safe_model_load(model_path):
    """Safely load YOLO model with error handling"""
    try:
        model = YOLO(model_path)
        if device == "cuda":
            model.to("cuda")
        return model, None
    except Exception as e:
        return None, str(e)

def run_inference(model, img, confidence=0.25):
    """Run YOLO inference on image"""
    try:
        results = model(img, conf=confidence, imgsz=640, device=device, verbose=False)
        annotated = results[0].plot()
        return annotated, results, None
    except Exception as e:
        return None, None, str(e)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model setup section
    st.subheader("📁 Model Setup")
    
    model_file = st.file_uploader("Upload YOLO Model (.pt)", type=["pt"])
    drive_url = st.text_input("Or paste Google Drive link")
    
    model_path = None
    
    if model_file is not None:
        model_path = os.path.join(MODEL_DIR, "uploaded_model.pt")
        with open(model_path, "wb") as f:
            f.write(model_file.read())
        st.success("✅ Model uploaded successfully!")
        
    elif drive_url:
        file_id = extract_file_id_from_drive_url(drive_url)
        if file_id:
            model_path = os.path.join(MODEL_DIR, "downloaded_model.pt")
            if not os.path.exists(model_path):
                with st.spinner("Downloading model..."):
                    try:
                        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
                        st.success("✅ Model downloaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Download failed: {e}")
                        model_path = None
            else:
                st.info("ℹ️ Model already exists")
        else:
            st.error("❌ Invalid Google Drive link")
    
    # Load model button
    if model_path and os.path.exists(model_path):
        if st.button("🚀 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model, error = safe_model_load(model_path)
                if model:
                    st.session_state.model = model
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to load model: {error}")
    
    st.divider()
    
    # Detection settings
    st.subheader("⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    frame_skip = st.selectbox("Process Every Nth Frame", [1, 2, 3, 4, 5], index=2)
    show_fps = st.checkbox("Show FPS Counter", value=True)
    
    st.divider()
    
    # Model info
    if st.session_state.model is not None:
        st.subheader("📊 Model Information")
        try:
            model_names = getattr(st.session_state.model, "names", {})
            if isinstance(model_names, dict) and model_names:
                st.write(f"**Classes:** {len(model_names)}")
                with st.expander("View all classes"):
                    for idx, name in model_names.items():
                        st.write(f"{idx}: {name}")
            st.write(f"**Device:** {device}")
        except Exception as e:
            st.error(f"Error getting model info: {e}")

# Main content area
if st.session_state.model is None:
    st.warning("⚠️ Please upload and load a YOLO model first")
    st.info("💡 Use the sidebar to upload a model file or provide a Google Drive link")
    st.stop()

# Mode selection
st.subheader("🎯 Detection Mode")
mode = st.selectbox("Choose detection method:", 
                   ["📸 Upload Image", "🎥 Upload Video", "📹 Real-Time Camera"],
                   format_func=lambda x: x)

# Image detection mode
if mode == "📸 Upload Image":
    st.subheader("Image Detection")
    
    uploaded_img = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_img:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            img = Image.open(uploaded_img)
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            img_np = np.array(img)
            
            # Handle different image formats
            if len(img_np.shape) == 2:  # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            
            with st.spinner("🔍 Detecting objects..."):
                annotated, results, error = run_inference(st.session_state.model, img_np, confidence)
                
                if error:
                    st.error(f"Detection failed: {error}")
                elif annotated is not None:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Display detection details
                    st.subheader("🎯 Detected Objects")
                    boxes = getattr(results[0], 'boxes', None)
                    if boxes is not None and hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                        detection_data = []
                        for conf, cls in zip(boxes.conf, boxes.cls):
                            class_name = st.session_state.model.names.get(int(cls), f"Class_{int(cls)}")
                            detection_data.append({
                                "Object": class_name,
                                "Confidence": f"{conf:.2f}",
                                "Percentage": f"{conf*100:.1f}%"
                            })
                        
                        st.dataframe(detection_data, use_container_width=True)
                        st.success(f"✅ Found {len(detection_data)} objects")
                    else:
                        st.info("No objects detected. Try lowering the confidence threshold.")

# Video detection mode
elif mode == "🎥 Upload Video":
    st.subheader("Video Detection")
    
    uploaded_vid = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_vid:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_vid.read())
        tfile.close()
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("❌ Error opening video file")
        else:
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"📹 Video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
            
            if st.button("🚀 Process Video", type="primary"):
                # Create output video
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                detections = []
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Process every nth frame
                    if frame_count % frame_skip == 0:
                        annotated, results, error = run_inference(st.session_state.model, frame, confidence)
                        if annotated is not None:
                            # Add frame number
                            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Collect detections
                            boxes = getattr(results[0], 'boxes', None)
                            if boxes is not None and hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                                for conf, cls in zip(boxes.conf, boxes.cls):
                                    class_name = st.session_state.model.names.get(int(cls), f"Class_{int(cls)}")
                                    detections.append({
                                        "frame": frame_count,
                                        "object": class_name,
                                        "confidence": float(conf)
                                    })
                            
                            out.write(annotated)
                        else:
                            out.write(frame)
                    else:
                        out.write(frame)
                
                cap.release()
                out.release()
                
                processing_time = time.time() - start_time
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                    st.metric("Processing FPS", f"{frame_count/processing_time:.1f}")
                
                with col2:
                    st.metric("Total Detections", len(detections))
                    unique_objects = len(set(d["object"] for d in detections))
                    st.metric("Unique Objects", unique_objects)
                
                # Show detection summary
                if detections:
                    st.subheader("🎯 Detection Summary")
                    detection_summary = {}
                    for d in detections:
                        obj = d["object"]
                        if obj not in detection_summary:
                            detection_summary[obj] = []
                        detection_summary[obj].append(d["confidence"])
                    
                    summary_data = []
                    for obj, confs in detection_summary.items():
                        summary_data.append({
                            "Object": obj,
                            "Count": len(confs),
                            "Avg Confidence": f"{np.mean(confs):.2f}",
                            "Max Confidence": f"{max(confs):.2f}"
                        })
                    
                    st.dataframe(summary_data, use_container_width=True)
                
                # Display processed video
                st.subheader("📹 Processed Video")
                st.video(out_path)
                
                # Cleanup
                os.unlink(tfile.name)
                # Note: keeping out_path for video display

# Real-time camera mode
elif mode == "📹 Real-Time Camera":
    st.subheader("Real-Time Camera Detection")
    
    class YOLOProcessor:
        def __init__(self):
            self.frame_count = 0
            self.fps_counter = 0
            self.last_time = time.time()
            
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            self.fps_counter += 1
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                st.session_state.fps = self.fps_counter / (current_time - self.last_time)
                self.fps_counter = 0
                self.last_time = current_time
            
            # Skip frames for performance
            if self.frame_count % frame_skip != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Run detection
            try:
                annotated, results, error = run_inference(st.session_state.model, img, confidence)
                
                if error:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # Store detections
                detections = []
                boxes = getattr(results[0], 'boxes', None)
                if boxes is not None and hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                    for conf, cls in zip(boxes.conf, boxes.cls):
                        class_name = st.session_state.model.names.get(int(cls), f"Class_{int(cls)}")
                        detections.append(f"{class_name}: {conf:.2f}")
                
                st.session_state.detections = detections
                
                # Add FPS counter
                if show_fps:
                    cv2.putText(annotated, f"FPS: {st.session_state.fps:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                
            except Exception as e:
                print(f"Detection error: {e}")
                return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # WebRTC configuration
    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })
    
    # Camera stream
    webrtc_ctx = webrtc_streamer(
        key="yolo-trash-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Status display
    col1, col2 = st.columns(2)
    
    with col1:
        if webrtc_ctx.state.playing:
            st.success("🟢 Camera Active")
            if show_fps:
                fps_placeholder = st.empty()
                fps_placeholder.markdown(f'<div class="fps-counter">FPS: {st.session_state.fps:.1f}</div>', 
                                       unsafe_allow_html=True)
        else:
            st.info("🔴 Camera Inactive - Click START to begin")
    
    with col2:
        if st.session_state.detections:
            st.subheader("🎯 Live Detections")
            for detection in st.session_state.detections[:5]:
                st.write(f"• {detection}")
            if len(st.session_state.detections) > 5:
                st.write(f"• ... and {len(st.session_state.detections) - 5} more")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🗑️ <strong>YOLO Trash Detector</strong> - Professional AI-Powered Waste Detection</p>
    <p>Built with Streamlit • YOLOv8 • PyTorch</p>
</div>
""", unsafe_allow_html=True)
