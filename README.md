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
