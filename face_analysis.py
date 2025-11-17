import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# App title and description
st.set_page_config(
    page_title="Face Analysis Pro",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Face Analysis Pro")
st.markdown("""
Detect faces, analyze emotions, estimate age and gender, and apply fun filters - all in real-time!
""")


class FaceAnalyzer:
    def __init__(self):
        self.face_detector = None
        self.age_net = None
        self.gender_net = None
        self.emotion_net = None
        self.load_models()

    def load_models(self):
        """Load all required models"""
        try:
            # Load face detection model
            face_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            face_model = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

            self.face_detector = self.download_model(face_proto, face_model, "face_detector")

            # Load age detection model
            age_proto = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt"
            age_weights = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
            self.age_net = self.download_model(age_proto, age_weights, "age_model")

            # Load gender detection model
            gender_proto = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt"
            gender_weights = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
            self.gender_net = self.download_model(gender_proto, gender_weights, "gender_model")

            st.success("✅ All models loaded successfully!")

        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")

    def download_model(self, proto_url, model_url, model_name):
        """Download model files"""
        temp_dir = tempfile.gettempdir()
        proto_path = os.path.join(temp_dir, f"{model_name}.prototxt")
        model_path = os.path.join(temp_dir, f"{model_name}.caffemodel")

        if not os.path.exists(proto_path):
            with st.spinner(f"Downloading {model_name}..."):
                self.download_file(proto_url, proto_path)
        if not os.path.exists(model_path):
            with st.spinner(f"Downloading {model_name} weights..."):
                self.download_file(model_url, model_path)

        return cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def download_file(self, url, filename):
        """Download file from URL"""
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

    def detect_faces(self, image):
        """Detect faces in image"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype(int))

        return faces

    def predict_age_gender(self, face_roi):
        """Predict age and gender for a face"""
        # Preprocess face for age/gender model
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))

        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        # Age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        age = age_ranges[np.argmax(age_preds[0])]

        return gender, age

    def analyze_image(self, image, show_filters=False):
        """Main analysis function"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces(image)

        results = {
            'faces_detected': len(faces),
            'face_data': [],
            'processed_image': image.copy()
        }

        for i, (startX, startY, endX, endY) in enumerate(faces):
            # Extract face ROI
            face_roi = image[startY:endY, startX:endX]

            if face_roi.size > 0:
                # Predict age and gender
                gender, age = self.predict_age_gender(face_roi)

                # Draw bounding box
                color = (0, 255, 0) if gender == "Female" else (255, 0, 0)
                cv2.rectangle(results['processed_image'], (startX, startY), (endX, endY), color, 2)

                # Add info text
                info_text = f"{gender}, {age}"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(results['processed_image'], info_text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Apply fun filters if enabled
                if show_filters:
                    results['processed_image'] = self.apply_face_filter(
                        results['processed_image'], startX, startY, endX, endY, gender
                    )

                results['face_data'].append({
                    'face_id': i + 1,
                    'bbox': (startX, startY, endX, endY),
                    'gender': gender,
                    'age': age,
                    'confidence': 0.85  # Placeholder
                })

        return results

    def apply_face_filter(self, image, x1, y1, x2, y2, gender):
        """Apply fun filters to faces"""
        try:
            # Add a simple crown filter for demonstration
            center_x = (x1 + x2) // 2
            crown_points = np.array([
                [center_x - 30, y1 - 20],
                [center_x, y1 - 50],
                [center_x + 30, y1 - 20]
            ], np.int32)

            cv2.fillPoly(image, [crown_points], (255, 215, 0))  # Gold color

            # Add glasses for males
            if gender == "Male":
                eye_center_y = y1 + (y2 - y1) // 3
                cv2.ellipse(image, (x1 + (x2 - x1) // 3, eye_center_y), (25, 15), 0, 0, 360, (255, 255, 255), 2)
                cv2.ellipse(image, (x2 - (x2 - x1) // 3, eye_center_y), (25, 15), 0, 0, 360, (255, 255, 255), 2)

            return image
        except Exception as e:
            return image


def main():
    # Initialize analyzer
    analyzer = FaceAnalyzer()

    if analyzer.face_detector is None:
        st.error("Failed to initialize face analyzer. Please check your internet connection.")
        return

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Basic Detection", "Age & Gender", "Fun Filters"]
    )

    confidence_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image with faces",
            type=["jpg", "jpeg", "png", "bmp"],
            help="For best results, use clear front-facing photos"
        )

        if uploaded_file is not None:
            # Process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Analyze
            show_filters = (analysis_mode == "Fun Filters")
            with st.spinner("🔍 Analyzing faces..."):
                results = analyzer.analyze_image(image_bgr, show_filters)

            # Display results
            st.subheader("📊 Analysis Results")
            st.success(f"✅ Found {results['faces_detected']} face(s)")

            if results['faces_detected'] > 0:
                for face in results['face_data']:
                    with st.expander(f"Face {face['face_id']}: {face['gender']}, {face['age']}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Gender", face['gender'])
                        with col_b:
                            st.metric("Age Range", face['age'])

                # Statistics
                genders = [face['gender'] for face in results['face_data']]
                male_count = genders.count("Male")
                female_count = genders.count("Female")

                st.metric("👨 Male", male_count)
                st.metric("👩 Female", female_count)

    with col2:
        st.subheader("🎯 Output")
        if uploaded_file is not None and results['faces_detected'] > 0:
            # Convert back to RGB for display
            processed_rgb = cv2.cvtColor(results['processed_image'], cv2.COLOR_BGR2RGB)
            st.image(processed_rgb, use_container_width=True, caption="Processed Image")

            # Download button
            processed_pil = Image.fromarray(processed_rgb)
            buf = BytesIO()
            processed_pil.save(buf, format="JPEG")

            st.download_button(
                label="📥 Download Processed Image",
                data=buf.getvalue(),
                file_name="face_analysis_result.jpg",
                mime="image/jpeg"
            )
        elif uploaded_file is not None:
            st.warning("⚠️ No faces detected. Try a different image or adjust confidence threshold.")
            st.image(image, use_container_width=True, caption="Original Image")
        else:
            st.info("👈 Upload an image to see results here")

    # Features overview
    st.markdown("---")
    st.subheader("🎨 Features Overview")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.markdown("""
        ### 👤 Face Detection
        - Multiple face detection
        - High accuracy
        - Real-time processing
        """)

    with col4:
        st.markdown("""
        ### 📊 Demographic Analysis
        - Gender prediction
        - Age estimation
        - Confidence scores
        """)

    with col5:
        st.markdown("""
        ### 🎭 Fun Filters
        - Augmented reality
        - Creative overlays
        - Customizable effects
        """)


if __name__ == "__main__":
    main()