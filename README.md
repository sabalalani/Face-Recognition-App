# 🚶 Person Detection App

A real-time person detection web application built with Streamlit and OpenCV, featuring multiple detection methods optimized for CPU performance.

![Person Detection](https://img.shields.io/badge/Person-Detection-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5-blue)

## ✨ Features

- **Multiple Detection Methods**: Choose between YOLOv3-tiny and Haar Cascade
- **CPU Optimized**: Specifically designed for cloud deployment without GPUs
- **Real-time Processing**: Fast detection with configurable parameters
- **User-Friendly Interface**: Intuitive Streamlit web interface
- **Export Results**: Download processed images with bounding boxes
- **Cross-Platform**: Works on any device with a web browser

## 🛠️ Detection Methods

### 1. YOLOv3-tiny (Recommended)
- **Accuracy**: Good balance of speed and accuracy
- **Speed**: ~1-3 seconds per image
- **Best For**: General purpose detection with decent accuracy

### 2. Haar Cascade (Fastest)
- **Accuracy**: Basic detection for clear frontal views
- **Speed**: < 1 second per image
- **Best For**: Maximum speed and simple use cases

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/person-detection-app.git
   cd person-detection-app
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run app.py
Open your browser

text
Local URL: http://localhost:8501
Network URL: http://your-ip:8501
Streamlit Cloud Deployment
Fork this repository to your GitHub account

Go to Streamlit Cloud and sign in

Click "New app" and configure:

Repository: your-username/person-detection-app

Branch: main

Main file path: app.py

Click "Deploy" - your app will be live in minutes!

# 📁 Project Structure
text
person-detection-app/

├── app.py                 # Main application file

├── requirements.txt       # Python dependencies

├── README.md             # Project documentation

├── .gitignore           # Git ignore rules
└── .streamlit/          # Streamlit configuration (optional)

    └── config.toml

# 🎯 Usage Guide
1. Upload Image
Click "Browse files" in the sidebar

Supported formats: JPG, JPEG, PNG

Optimal size: 500KB - 2MB

2. Configure Settings
Detection Method: Choose YOLOv3-tiny or Haar Cascade

Confidence Threshold: Adjust detection sensitivity (0.1-0.9)

Max Image Size: Set processing dimension (400-800px)

3. View Results
Detection results with bounding boxes

Confidence scores for each detection

Option to download processed image

4. Performance Tips
Use images with clear, visible people

Start with lower confidence thresholds

Choose smaller image sizes for faster processing

JPG format typically processes faster than PNG

# 🔧 Technical Details
Models Used
YOLOv3-tiny

Weights: 33.7MB

Classes: 80 object categories

Input size: 320x320 (optimized for CPU)

Download: Automatic on first run

Haar Cascade

Model: OpenCV's haarcascade_fullbody.xml

Input: Grayscale images

Speed: Very fast, less accurate


# 🌐 API Reference
The application provides a simple web interface with the following components:

Input Parameters
image_file: Uploaded image file

detection_method: Algorithm selection

confidence_threshold: Detection sensitivity

max_image_size: Processing dimension

Output
Processed image with bounding boxes

Detection count and confidence scores

Downloadable result image

# 🐛 Troubleshooting
Common Issues
Slow Processing

Reduce image size in settings

Use Haar Cascade method

Upload smaller image files

No Detections

Lower confidence threshold

Try different detection method

Ensure good image quality and lighting

Model Download Failures

Check internet connection

Restart the application

Manual download option available

Memory Errors

Use smaller images

Restart the application

Check available system resources

Logs and Debugging
Enable detailed logging by setting:

python
logging.basicConfig(level=logging.DEBUG)
# 🤝 Contributing
I welcome contributions! Please follow these steps:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Development Setup
bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt


# 🙏 Acknowledgments
OpenCV for computer vision libraries

Streamlit for the amazing web framework

YOLO authors for the object detection models

COCO dataset for training data


# 🔄 Changelog
v1.0.0 (2024-01-01)
Initial release

YOLOv3-tiny and Haar Cascade support

Streamlit web interface

CPU optimization

<div align="center">
Made by Saba Bashir with ❤️ using Streamlit and OpenCV

https://static.streamlit.io/badges/streamlit_badge_black_white.svg
https://img.shields.io/badge/OpenCV-5.3.1-green.svg

</div> ```