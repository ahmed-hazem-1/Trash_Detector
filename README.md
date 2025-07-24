# Trash Detector

## Overview
Trash Detector is a machine learning project aimed at identifying and classifying various types of waste in images using the YOLOv8 model. The dataset, sourced from Roboflow, contains 22,746 images split into training (89%), validation (7%), and test (3%) sets. The model is designed to detect objects such as plastic bottles, cans, straws, and other waste items, contributing to automated waste management solutions. Additionally, a Streamlit web application has been developed to demonstrate the model's capabilities interactively.

## Features
- **YOLOv8 Model**: Utilizes the YOLOv8s model fine-tuned on a custom waste detection dataset with 22 classes.
- **Data Preprocessing**: Images are resized to 244x244, auto-oriented, and augmented with techniques like horizontal flips, 90° rotations, saturation adjustments (±33%), brightness adjustments (±25%), and noise addition (up to 0.81% of pixels).
- **Streamlit App**: An interactive web interface to upload images and visualize waste detection results.
- **Performance**: Achieves high precision, recall, and mAP scores for various waste classes (e.g., plastic bottles, straws, snack bags).

## Installation
To run the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ahmed-hazem-1/Trash_Detector.git
   cd Trash_Detector
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.11 installed. Then, install the required packages:
   ```bash
   pip install roboflow ultralytics opencv-python matplotlib pillow numpy torch
   ```

3. **Download the Dataset**:
   Use the Roboflow API to download the waste detection dataset. Replace `<YOUR_API_KEY>` with your Roboflow API key:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="<YOUR_API_KEY>")
   project = rf.workspace().project("waste-detection-3")
   dataset = project.version(1).download("yolov8")
   ```

4. **Download the Trained Model**:
   The trained model weights are available in the repository or can be downloaded from Google Drive (see the notebook for details).

## Usage
1. **Training the Model**:
   Run the training script using the YOLOv8 model:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8s.pt")
   results = model.train(
       data="path/to/waste-detection-3/data.yaml",
       epochs=200,
       batch=250,
       save=True,
       save_period=1,
       patience=5,
       project="runs",
       name="exp"
   )
   ```

2. **Validating the Model**:
   Validate the trained model to check performance metrics:
   ```python
   results = model.val()
   print(results.box_map)
   ```

3. **Visualizing Results**:
   Use the provided scripts to visualize sample images or validation results:
   ```python
   import matplotlib.pyplot as plt
   import cv2
   import glob
   import os
   val_images_folder = 'runs/val'
   image_files = glob.glob(os.path.join(val_images_folder, '*.jpg'))
   for img_path in image_files:
       img = cv2.imread(img_path)
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       plt.imshow(img)
       plt.title(os.path.basename(img_path))
       plt.axis('off')
       plt.show()
   ```

4. **Streamlit App**:
   Try the interactive web app to upload images and see real-time waste detection:
   [Trash Detector Streamlit App](https://trash--detector.streamlit.app/)

## Dataset
- **Source**: Roboflow
- **Size**: 22,746 images
- **Split**: 
  - Training: 20,352 images (89%)
  - Validation: 1,607 images (7%)
  - Test: 787 images (3%)
- **Classes**: 22 waste types, including plastic bottles, cans, cardboard, snack bags, straws, etc.

## Model Performance
The model was trained for 200 epochs with a batch size of 250. Key metrics include:
- **mAP**: ~0.75 (varies by class)
- **Precision/Recall**: High for classes like plastic bottles (~0.97 precision, ~0.86 recall) and straws (~0.99 precision, ~0.99 recall).
- **Speed**: ~0.7ms preprocess, ~3.6ms inference per image on a Tesla T4 GPU.

## Repository
- **GitHub**: [https://github.com/ahmed-hazem-1/Trash_Detector](https://github.com/ahmed-hazem-1/Trash_Detector)

## Streamlit App
Explore the model through the interactive Streamlit app: [https://trash--detector.streamlit.app/](https://trash--detector.streamlit.app/)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.

