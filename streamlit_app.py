import streamlit as st
import torch
import numpy as np
import tempfile
import os
import time
import base64
from PIL import Image
import gdown
import io

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

# Custom CSS for better UI and camera functionality
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
    .camera-container {
        text-align: center;
        padding: 20px;
        border: 2px dashed #ccc;
        border-radius: 10px;
        margin: 20px 0;
    }
    .upload-container {
        padding: 20px;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        text-align: center;
        background-color: #f9f9f9;
    }
    .detection-result {
        background-color: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        color: #856404;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        color: #155724;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        color: #0c5460;
        margin: 10px 0;
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
if 'processing_video' not in st.session_state:
    st.session_state.processing_video = False

# Check dependencies
if not CV2_AVAILABLE or not YOLO_AVAILABLE:
    st.stop()

# Title with styling
st.markdown('<div class="main-header"><h1>🗑️ YOLO Trash Detector</h1><p>Cloud-Friendly AI-Powered Waste Detection System</p></div>', unsafe_allow_html=True)

# Check for device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.markdown('<div class="warning-box">⚠️ Running on CPU. For faster processing, consider using a CUDA-enabled environment.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="success-box">✅ CUDA available. GPU acceleration enabled for optimal performance.</div>', unsafe_allow_html=True)

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

def process_image_from_bytes(image_bytes, model, confidence):
    """Process image from bytes data"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run inference
        annotated, results, error = run_inference(model, img_array, confidence)
        
        if error:
            return None, None, error
            
        return annotated, results, None
        
    except Exception as e:
        return None, None, str(e)

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model setup section
    st.subheader("📁 Model Setup")
    
    model_file = st.file_uploader("Upload YOLO Model (.pt)", type=["pt"])
    
    # Default model option
    use_default = st.checkbox("Use YOLOv8n (will download automatically)", value=True)
    
    drive_url = st.text_input("Or paste Google Drive link to custom model")
    
    model_path = None
    
    if use_default and not model_file and not drive_url:
        model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
        if not os.path.exists(model_path):
            st.info("Default YOLOv8n model will be downloaded on first use")
    elif model_file is not None:
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
    if model_path:
        if st.button("🚀 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                if use_default and not os.path.exists(model_path):
                    # Download default YOLOv8n model
                    try:
                        model = YOLO('yolov8n.pt')  # This will auto-download
                        if device == "cuda":
                            model.to("cuda")
                        st.session_state.model = model
                        st.success("✅ YOLOv8n model loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load default model: {e}")
                else:
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
    st.markdown('<div class="warning-box">⚠️ Please load a YOLO model first. You can use the default YOLOv8n model or upload your own.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Use the sidebar to select and load a model. The default YOLOv8n model works well for general object detection.</div>', unsafe_allow_html=True)
    st.stop()

# Mode selection
st.subheader("🎯 Detection Mode")
mode = st.selectbox("Choose detection method:", 
                   ["📸 Upload Image", "📹 Camera Capture", "🎥 Upload Video"],
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
            
            with st.spinner("🔍 Detecting objects..."):
                annotated, results, error = process_image_from_bytes(
                    uploaded_img.getvalue(), st.session_state.model, confidence
                )
                
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
                        st.markdown(f'<div class="detection-result">✅ Found {len(detection_data)} objects</div>', unsafe_allow_html=True)
                    else:
                        st.info("No objects detected. Try lowering the confidence threshold.")

# Camera capture mode (Cloud-friendly)
elif mode == "📹 Camera Capture":
    st.subheader("Camera Capture")
    
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    
    # HTML5 Camera capture component
    camera_html = """
    <div style="text-align: center;">
        <video id="video" width="640" height="480" autoplay style="border: 2px solid #4CAF50; border-radius: 10px;"></video>
        <br><br>
        <button id="capture" onclick="captureImage()" style="
            background-color: #4CAF50; 
            color: white; 
            padding: 15px 32px; 
            text-align: center; 
            text-decoration: none; 
            display: inline-block; 
            font-size: 16px; 
            margin: 4px 2px; 
            cursor: pointer; 
            border: none; 
            border-radius: 5px;
        ">📸 Capture Photo</button>
        <br><br>
        <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
        <img id="captured" style="max-width: 100%; height: auto; border: 2px solid #4CAF50; border-radius: 10px; display: none;">
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const capturedImg = document.getElementById('captured');
        const ctx = canvas.getContext('2d');
        
        // Access camera
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error('Error accessing camera: ', err);
                alert('Error accessing camera. Please make sure you have granted camera permissions.');
            });
        
        function captureImage() {
            // Draw video frame to canvas
            ctx.drawImage(video, 0, 0, 640, 480);
            
            // Convert to base64
            const dataURL = canvas.toDataURL('image/jpeg');
            
            // Show captured image
            capturedImg.src = dataURL;
            capturedImg.style.display = 'block';
            
            // Send to Streamlit (you'll need to handle this part)
            window.parent.postMessage({
                type: 'camera-capture',
                data: dataURL
            }, '*');
        }
    </script>
    """
    
    # Display camera interface
    st.components.v1.html(camera_html, height=600)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Alternative: Simple file upload for mobile users
    st.markdown("---")
    st.subheader("📱 Alternative: Upload from Device Camera")
    st.markdown('<div class="info-box">On mobile devices, you can use the camera directly by uploading an image below:</div>', unsafe_allow_html=True)
    
    camera_upload = st.file_uploader(
        "Take/Upload a photo", 
        type=["jpg", "jpeg", "png"],
        help="On mobile, this will open your camera app"
    )
    
    if camera_upload:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            img = Image.open(camera_upload)
            st.image(img, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            with st.spinner("🔍 Analyzing image..."):
                annotated, results, error = process_image_from_bytes(
                    camera_upload.getvalue(), st.session_state.model, confidence
                )
                
                if error:
                    st.error(f"Detection failed: {error}")
                elif annotated is not None:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # Display detection details
                    boxes = getattr(results[0], 'boxes', None)
                    if boxes is not None and hasattr(boxes, 'conf') and len(boxes.conf) > 0:
                        st.subheader("🎯 Detected Objects")
                        detection_data = []
                        for conf, cls in zip(boxes.conf, boxes.cls):
                            class_name = st.session_state.model.names.get(int(cls), f"Class_{int(cls)}")
                            detection_data.append({
                                "Object": class_name,
                                "Confidence": f"{conf:.2f}",
                                "Score": f"{conf*100:.1f}%"
                            })
                        
                        st.dataframe(detection_data, use_container_width=True)
                        st.markdown(f'<div class="detection-result">✅ Detected {len(detection_data)} objects with AI analysis</div>', unsafe_allow_html=True)
                    else:
                        st.info("No objects detected. Try adjusting the confidence threshold or ensure the image is clear.")

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
            
            st.markdown(f'<div class="info-box">📹 Video Properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames</div>', unsafe_allow_html=True)
            
            # Processing options
            col1, col2 = st.columns(2)
            with col1:
                frame_skip = st.selectbox("Process every Nth frame", [1, 2, 3, 5, 10], index=2, help="Higher values = faster processing, fewer detections")
            with col2:
                max_frames = st.number_input("Max frames to process", min_value=10, max_value=min(1000, total_frames), value=min(300, total_frames))
            
            if st.button("🚀 Process Video", type="primary"):
                if not st.session_state.processing_video:
                    st.session_state.processing_video = True
                    
                    # Create output video
                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    frame_count = 0
                    processed_frames = 0
                    detections = []
                    start_time = time.time()
                    
                    while cap.isOpened() and frame_count < max_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        progress = min(frame_count / min(max_frames, total_frames), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{min(max_frames, total_frames)}")
                        
                        # Process every nth frame
                        if frame_count % frame_skip == 0:
                            annotated, results, error = run_inference(st.session_state.model, frame, confidence)
                            if annotated is not None:
                                processed_frames += 1
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
                    st.session_state.processing_video = False
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")
                    
                    # Show results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Processing Time", f"{processing_time:.1f}s")
                    with col2:
                        st.metric("Frames Processed", processed_frames)
                    with col3:
                        st.metric("Total Detections", len(detections))
                    
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
                        st.markdown(f'<div class="detection-result">🎯 Analysis complete! Found {len(detection_summary)} different object types across {len(detections)} detections.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ No objects detected in the video. Try lowering the confidence threshold or using a different model.</div>', unsafe_allow_html=True)
                    
                    # Display processed video
                    st.subheader("📹 Processed Video")
                    st.video(out_path)
                    
                    # Cleanup
                    os.unlink(tfile.name)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🗑️ <strong>YOLO Trash Detector</strong> - Cloud-Friendly AI Detection System</p>
    <p>✨ Optimized for Streamlit Cloud Deployment</p>
    <p>Built with Streamlit • YOLOv8 • OpenCV • PyTorch</p>
</div>
""", unsafe_allow_html=True)
