import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import requests
from io import BytesIO
import logging

# Try to import OpenCV with fallback
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    OPENCV_AVAILABLE = False
except Exception as e:
    st.warning(f"OpenCV initialization issue: {e}")
    OPENCV_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Smart Face Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Smart Face Detector")
st.markdown("Reliable face detection using optimized computer vision models")


class SmartFaceDetector:
    def __init__(self):
        self.face_detector = None
        self.load_model()

    def load_model(self):
        """Load face detection model with proper error handling"""
        if not OPENCV_AVAILABLE:
            st.error("❌ OpenCV not available. Using basic image processing.")
            return False

        try:
            st.info("🔄 Loading face detection model...")

            # Try to use OpenCV's DNN face detector
            proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

            self.face_detector = self.download_and_load_model(proto_url, model_url, "face_detector")

            if self.face_detector is not None:
                st.success("✅ Face detection model loaded successfully!")
                return True
            else:
                st.warning("⚠️ Using alternative detection method")
                return self.load_alternative_detector()

        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            return self.load_alternative_detector()

    def load_alternative_detector(self):
        """Load alternative detection method"""
        try:
            # Try Haar cascades
            cascade_path = self.download_haar_cascade()
            if cascade_path and os.path.exists(cascade_path):
                self.face_detector = "haar"
                st.info("✅ Using Haar cascade for face detection")
                return True
            else:
                st.warning("⚠️ Using basic image analysis")
                self.face_detector = "basic"
                return True
        except Exception as e:
            st.error(f"❌ Alternative detector failed: {e}")
            self.face_detector = "basic"
            return True

    def download_haar_cascade(self):
        """Download Haar cascade file"""
        try:
            cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            cascade_path = os.path.join(tempfile.gettempdir(), "haarcascade_frontalface_default.xml")

            if not os.path.exists(cascade_path):
                response = requests.get(cascade_url)
                with open(cascade_path, 'wb') as f:
                    f.write(response.content)

            return cascade_path
        except Exception as e:
            logging.error(f"Haar cascade download failed: {e}")
            return None

    def download_and_load_model(self, proto_url, model_url, model_name):
        """Download and load DNN model"""
        try:
            temp_dir = tempfile.gettempdir()
            proto_path = os.path.join(temp_dir, f"{model_name}.prototxt")
            model_path = os.path.join(temp_dir, f"{model_name}.caffemodel")

            # Download files if needed
            for url, path in [(proto_url, proto_path), (model_url, model_path)]:
                if not os.path.exists(path):
                    response = requests.get(url, timeout=60)
                    with open(path, 'wb') as f:
                        f.write(response.content)

            # Load model
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            return net

        except Exception as e:
            logging.error(f"DNN model loading failed: {e}")
            return None

    def detect_faces(self, image, confidence_threshold=0.5):
        """Main face detection function"""
        if not OPENCV_AVAILABLE or self.face_detector == "basic":
            return self.detect_faces_basic(image)
        elif self.face_detector == "haar":
            return self.detect_faces_haar(image)
        else:
            return self.detect_faces_dnn(image, confidence_threshold)

    def detect_faces_dnn(self, image, confidence_threshold):
        """DNN-based face detection"""
        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()

            faces = []
            confidences = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)

                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        faces.append([x1, y1, x2, y2])
                        confidences.append(confidence)

            return faces, confidences

        except Exception as e:
            logging.error(f"DNN detection failed: {e}")
            return [], []

    def detect_faces_haar(self, image):
        """Haar cascade face detection"""
        try:
            cascade_path = self.download_haar_cascade()
            face_cascade = cv2.CascadeClassifier(cascade_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            converted_faces = []
            for (x, y, w, h) in faces:
                converted_faces.append([x, y, x + w, y + h])

            return converted_faces, [0.8] * len(faces)

        except Exception as e:
            logging.error(f"Haar detection failed: {e}")
            return [], []

    def detect_faces_basic(self, image):
        """Basic face detection using image analysis"""
        # This is a simple fallback that looks for face-like patterns
        # In a real app, you might use a different library or API
        try:
            # Convert to PIL for basic processing
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image

            # Simple face-like region detection (very basic)
            width, height = pil_image.size
            faces = []

            # Look for bright regions that might be faces (very simplistic)
            # This is just a placeholder - in production, use a proper face detection service
            gray_image = pil_image.convert('L')
            pixels = np.array(gray_image)

            # Simple threshold-based detection
            bright_regions = pixels > 150  # Adjust threshold as needed

            # Find connected components (very basic)
            from scipy import ndimage
            labeled, num_features = ndimage.label(bright_regions)

            for i in range(1, num_features + 1):
                region = labeled == i
                if np.sum(region) > 1000:  # Minimum size
                    coords = np.where(region)
                    y1, x1 = np.min(coords[0]), np.min(coords[1])
                    y2, x2 = np.max(coords[0]), np.max(coords[1])

                    if (x2 - x1) > 50 and (y2 - y1) > 50:  # Reasonable face size
                        faces.append([x1, y1, x2, y2])

            return faces, [0.5] * len(faces)  # Low confidence for basic detection

        except Exception as e:
            logging.error(f"Basic detection failed: {e}")
            return [], []

    def analyze_image(self, image, confidence_threshold=0.5):
        """Analyze image and return results"""
        if not OPENCV_AVAILABLE:
            # Use PIL for basic image processing
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            return {
                'faces_detected': 0,
                'face_data': [],
                'processed_image': pil_image,
                'total_confidence': 0.0,
                'method': 'basic'
            }

        # Convert PIL to OpenCV format if needed
        if isinstance(image, Image.Image):
            image_cv = np.array(image)
            if len(image_cv.shape) == 3:
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image.copy()

        faces, confidences = self.detect_faces(image_cv, confidence_threshold)

        results = {
            'faces_detected': len(faces),
            'face_data': [],
            'processed_image': image_cv.copy(),
            'total_confidence': 0.0,
            'method': self.face_detector if self.face_detector != "basic" else "basic"
        }

        for i, (x1, y1, x2, y2) in enumerate(faces):
            confidence = confidences[i] if i < len(confidences) else 0.5

            # Calculate face info
            face_width = x2 - x1
            face_height = y2 - y1

            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw bounding box
            cv2.rectangle(results['processed_image'], (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"Face {i + 1}: {confidence:.1%}"
            cv2.putText(results['processed_image'], label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            results['face_data'].append({
                'face_id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'width': face_width,
                'height': face_height
            })

        if results['faces_detected'] > 0:
            results['total_confidence'] = sum(confidences) / len(confidences)

        return results


def main():
    # Initialize detector
    detector = SmartFaceDetector()

    # Sidebar
    st.sidebar.header("⚙️ Detection Settings")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Higher values = fewer but more reliable detections"
    )

    # Model info
    st.sidebar.header("🔧 System Info")
    if not OPENCV_AVAILABLE:
        st.sidebar.error("OpenCV not available")
        st.sidebar.info("Using basic image processing")
    else:
        st.sidebar.success("OpenCV loaded successfully")
        if detector.face_detector == "haar":
            st.sidebar.info("Using: **Haar Cascade**")
        elif detector.face_detector == "basic":
            st.sidebar.warning("Using: **Basic Analysis**")
        else:
            st.sidebar.info("Using: **DNN Model**")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image for face detection",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            try:
                # Load image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

                # Analyze image
                with st.spinner("🔍 Analyzing image..."):
                    results = detector.analyze_image(image, confidence_threshold)

                # Display results
                st.subheader("📊 Detection Results")

                if results['faces_detected'] > 0:
                    st.success(f"✅ **{results['faces_detected']} face(s) detected**")
                    if results['total_confidence'] > 0:
                        st.metric("Average Confidence", f"{results['total_confidence']:.1%}")

                    for face in results['face_data']:
                        with st.expander(f"Face {face['face_id']} ({face['confidence']:.1%} confidence)"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Width", f"{face['width']}px")
                            with col_b:
                                st.metric("Height", f"{face['height']}px")

                            # Fixed progress bar
                            confidence_percent = int(face['confidence'] * 100)
                            st.progress(confidence_percent, text=f"Confidence: {face['confidence']:.1%}")

                else:
                    st.warning("❌ No faces detected")
                    st.info("""
                    **Tips for better detection:**
                    - Try lowering the confidence threshold
                    - Use clear, front-facing photos
                    - Ensure good lighting
                    - Avoid heavily rotated faces
                    """)

            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

    with col2:
        st.subheader("🎯 Detection Output")
        if uploaded_file is not None and 'results' in locals():
            if results['faces_detected'] > 0:
                # Display processed image
                if OPENCV_AVAILABLE and results['method'] != 'basic':
                    processed_rgb = cv2.cvtColor(results['processed_image'], cv2.COLOR_BGR2RGB)
                    processed_pil = Image.fromarray(processed_rgb)
                else:
                    processed_pil = results['processed_image']

                st.image(processed_pil, use_container_width=True,
                         caption=f"Detected {results['faces_detected']} face(s)")

                # Download button
                buf = BytesIO()
                processed_pil.save(buf, format="JPEG", quality=95)

                st.download_button(
                    label="📥 Download Result",
                    data=buf.getvalue(),
                    file_name="face_detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            else:
                st.info("👆 No faces detected in the image")
        else:
            st.info("""
            ## 🎯 How to Use:

            1. **Upload** an image using the file uploader
            2. **Adjust** confidence threshold if needed
            3. **View** detection results and analysis
            4. **Download** the processed image

            ### 💡 Best Practices:
            - Clear, well-lit images work best
            - Front-facing faces detected most accurately
            - Multiple faces supported
            - Various image formats accepted
            """)

    # Features
    st.markdown("---")
    st.subheader("🚀 Features")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown("""
        ### 🔍 Smart Detection
        - Multiple detection methods
        - Automatic fallback systems
        - Confidence scoring
        - Robust error handling
        """)

    with col4:
        st.markdown("""
        ### 🎨 Visualization
        - Color-coded confidence
        - Bounding box annotations
        - Detailed face information
        - Downloadable results
        """)

    with col5:
        st.markdown("""
        ### ⚡ Performance
        - Streamlit Cloud optimized
        - Fast processing
        - Low memory usage
        - Mobile friendly
        """)


if __name__ == "__main__":
    main()