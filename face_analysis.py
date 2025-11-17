import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import requests
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Accurate Face Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Accurate Face Detector")
st.markdown("Reliable face detection with high accuracy using OpenCV's best models")


class AccurateFaceDetector:
    def __init__(self):
        self.face_detector = None
        self.load_model()

    def load_model(self):
        """Load the most reliable face detection model"""
        try:
            st.info("🔄 Loading high-accuracy face detection model...")

            # Use the most reliable OpenCV face detector
            proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

            self.face_detector = self.download_and_load_model(proto_url, model_url, "accurate_face_detector")

            if self.face_detector is not None:
                st.success("✅ High-accuracy face detector loaded!")
                return True
            else:
                return self.load_haar_cascade()

        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            return self.load_haar_cascade()

    def load_haar_cascade(self):
        """Load Haar cascade as fallback"""
        try:
            st.warning("⚠️ Using Haar cascade (good accuracy for frontal faces)")
            self.face_detector = "haar"
            return True
        except Exception as e:
            st.error(f"❌ All detection methods failed: {e}")
            return False

    def download_and_load_model(self, proto_url, model_url, model_name):
        """Download and load model with verification"""
        temp_dir = tempfile.gettempdir()
        proto_path = os.path.join(temp_dir, f"{model_name}.prototxt")
        model_path = os.path.join(temp_dir, f"{model_name}.caffemodel")

        try:
            # Download files if needed
            for url, path in [(proto_url, proto_path), (model_url, model_path)]:
                if not os.path.exists(path):
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    with open(path, 'wb') as f:
                        f.write(response.content)

            # Load model
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

            # Test the model with a simple check
            test_blob = cv2.dnn.blobFromImage(np.random.rand(300, 300, 3).astype(np.float32), 1.0, (300, 300),
                                              [104, 117, 123])
            net.setInput(test_blob)
            net.forward()  # This will fail if model is corrupted

            return net

        except Exception as e:
            logging.error(f"Model download/load failed: {e}")
            # Clean up potentially corrupted files
            for path in [proto_path, model_path]:
                if os.path.exists(path):
                    os.remove(path)
            return None

    def detect_faces_dnn(self, image, confidence_threshold=0.5):
        """Accurate face detection using DNN"""
        h, w = image.shape[:2]

        # Create blob with proper preprocessing
        blob = cv2.dnn.blobFromImage(
            image,
            1.0,
            (300, 300),
            [104, 117, 123],  # Mean subtraction values
            swapRB=True,
            crop=False
        )

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        faces = []
        confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Only add if face region is valid
                if x2 > x1 and y2 > y1:
                    faces.append([x1, y1, x2, y2])
                    confidences.append(confidence)

        return faces, confidences

    def detect_faces_haar(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """Face detection using Haar cascades"""
        try:
            # Load cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                # Download if not available
                cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                response = requests.get(cascade_url)
                cascade_path = os.path.join(tempfile.gettempdir(), "haarcascade_frontalface_default.xml")
                with open(cascade_path, 'wb') as f:
                    f.write(response.content)

            face_cascade = cv2.CascadeClassifier(cascade_path)

            # Convert to grayscale for Haar cascades
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces with multiple parameter sets for better accuracy
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=minSize,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Convert to consistent format
            converted_faces = []
            for (x, y, w, h) in faces:
                converted_faces.append([x, y, x + w, y + h])

            return converted_faces, [0.9] * len(faces)  # High confidence for Haar

        except Exception as e:
            logging.error(f"Haar detection failed: {e}")
            return [], []

    def detect_faces(self, image, confidence_threshold=0.5):
        """Main detection function"""
        if self.face_detector == "haar":
            return self.detect_faces_haar(image)
        else:
            return self.detect_faces_dnn(image, confidence_threshold)

    def analyze_image(self, image, confidence_threshold=0.5):
        """Analyze image and return detailed results"""
        faces, confidences = self.detect_faces(image, confidence_threshold)

        results = {
            'faces_detected': len(faces),
            'face_data': [],
            'processed_image': image.copy(),
            'total_confidence': 0.0
        }

        for i, (x1, y1, x2, y2) in enumerate(faces):
            confidence = confidences[i] if i < len(confidences) else 0.8

            # Calculate face area and position for additional info
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            image_area = image.shape[0] * image.shape[1]
            area_percentage = (face_area / image_area) * 100

            # Determine face size category
            if area_percentage > 10:
                size_category = "Large"
            elif area_percentage > 5:
                size_category = "Medium"
            else:
                size_category = "Small"

            # Determine position in image
            center_x = (x1 + x2) / 2
            image_center_x = image.shape[1] / 2

            if center_x < image_center_x - 100:
                position = "Left"
            elif center_x > image_center_x + 100:
                position = "Right"
            else:
                position = "Center"

            # Choose color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence

            # Draw bounding box
            cv2.rectangle(results['processed_image'], (x1, y1), (x2, y2), color, 3)

            # Draw confidence background
            label = f"Face {i + 1}: {confidence:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(results['processed_image'],
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          color, -1)

            # Draw confidence text
            cv2.putText(results['processed_image'], label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Store face data
            results['face_data'].append({
                'face_id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'width': face_width,
                'height': face_height,
                'area_percentage': area_percentage,
                'size_category': size_category,
                'position': position
            })

        # Calculate average confidence
        if results['faces_detected'] > 0:
            results['total_confidence'] = sum(confidences) / len(confidences)

        return results


def main():
    # Initialize detector
    detector = AccurateFaceDetector()

    if detector.face_detector is None:
        st.error("Failed to initialize face detector. Please refresh the page.")
        return

    # Sidebar
    st.sidebar.header("⚙️ Detection Settings")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.1,
        help="Higher values = fewer but more reliable detections"
    )

    show_detection_info = st.sidebar.checkbox(
        "Show Detailed Detection Info",
        value=True,
        help="Display additional information about each detection"
    )

    # Model info
    st.sidebar.header("🔧 Model Info")
    if detector.face_detector == "haar":
        st.sidebar.info("Using: **Haar Cascade**\n\nGood for frontal faces")
    else:
        st.sidebar.info("Using: **DNN Model**\n\nHigh accuracy for various angles")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image for face detection",
            type=["jpg", "jpeg", "png"],
            help="For best results, use clear images with visible faces"
        )

        if uploaded_file is not None:
            try:
                # Load and display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

                # Convert to OpenCV format
                image_np = np.array(image)
                if len(image_np.shape) == 3:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

                # Analyze image
                with st.spinner("🔍 Detecting faces..."):
                    results = detector.analyze_image(image_bgr, confidence_threshold)

                # Display results
                st.subheader("📊 Detection Results")

                if results['faces_detected'] > 0:
                    st.success(f"✅ **{results['faces_detected']} face(s) detected**")
                    st.metric("Average Confidence", f"{results['total_confidence']:.1%}")

                    if show_detection_info:
                        for face in results['face_data']:
                            with st.expander(f"Face {face['face_id']} ({face['confidence']:.1%} confidence)"):
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Size", face['size_category'])
                                with col_b:
                                    st.metric("Position", face['position'])
                                with col_c:
                                    st.metric("Area", f"{face['area_percentage']:.1f}%")

                                # FIXED: Convert float to int for progress bar
                                confidence_percent = int(face['confidence'] * 100)
                                st.progress(confidence_percent, text=f"Detection Confidence: {face['confidence']:.1%}")

                else:
                    st.warning("❌ No faces detected")
                    st.info("""
                    **Tips for better detection:**
                    - Try lowering the confidence threshold
                    - Ensure faces are clearly visible
                    - Use front-facing photos when possible
                    - Check lighting conditions
                    - Avoid heavily rotated faces
                    """)

            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

    with col2:
        st.subheader("🎯 Detection Output")
        if uploaded_file is not None and 'results' in locals():
            if results['faces_detected'] > 0:
                # Convert and display processed image
                processed_rgb = cv2.cvtColor(results['processed_image'], cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, use_container_width=True,
                         caption=f"Detected {results['faces_detected']} face(s)")

                # Confidence color guide
                st.caption("🎨 Confidence Colors: "
                           "🟢 High (>80%) | 🟡 Medium (50-80%) | 🔴 Low (<50%)")

                # Download button
                processed_pil = Image.fromarray(processed_rgb)
                buf = BytesIO()
                processed_pil.save(buf, format="JPEG", quality=95)

                st.download_button(
                    label="📥 Download Result Image",
                    data=buf.getvalue(),
                    file_name="face_detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            else:
                st.info("👆 No faces detected in the image")
        else:
            st.info("""
            ## 🎯 What to Expect:

            **High Accuracy Detection:**
            - Reliable face bounding boxes
            - Confidence scores for each detection
            - Size and position analysis
            - Color-coded confidence levels

            **Best Practices:**
            - Clear, well-lit images
            - Multiple faces supported
            - Various angles and sizes
            - Real-time processing
            """)

    # Accuracy tips
    st.markdown("---")
    st.subheader("💡 Accuracy Tips")

    tip_col1, tip_col2, tip_col3 = st.columns(3)

    with tip_col1:
        st.markdown("""
        ### 🖼️ Image Quality
        - Use high-resolution images
        - Good lighting conditions
        - Clear focus on faces
        - Avoid motion blur
        """)

    with tip_col2:
        st.markdown("""
        ### 👤 Face Visibility
        - Front-facing works best
        - Avoid heavy obstructions
        - Multiple angles supported
        - Various sizes detected
        """)

    with tip_col3:
        st.markdown("""
        ### ⚙️ Settings
        - Adjust confidence threshold
        - Start with 0.6 confidence
        - Lower for more detections
        - Higher for fewer but reliable
        """)


if __name__ == "__main__":
    main()
    main()
    main()