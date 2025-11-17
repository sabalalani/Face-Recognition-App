import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import requests
from io import BytesIO
import tempfile
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Pure Python Face Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Pure Python Face Detector")
st.markdown("Face detection using pure Python libraries - No OpenCV required!")


class PurePythonFaceDetector:
    def __init__(self):
        self.face_model = None
        st.success("✅ Pure Python detector ready!")

    def detect_faces_python(self, image, confidence_threshold=0.5):
        """
        Simple face detection using image processing techniques
        This is a basic implementation - for production, consider using a proper face detection API
        """
        try:
            # Convert to grayscale for processing
            gray_image = image.convert('L')
            width, height = gray_image.size

            # Simple skin tone detection (very basic approach)
            skin_regions = self.detect_skin_regions(image)

            # Look for oval-like shapes that could be faces
            face_candidates = self.find_face_candidates(skin_regions, width, height)

            # Filter candidates based on size and aspect ratio
            valid_faces = self.filter_face_candidates(face_candidates, width, height)

            # Convert to bounding boxes with confidence scores
            faces = []
            confidences = []

            for candidate in valid_faces:
                x, y, w, h = candidate
                confidence = self.calculate_confidence(image, x, y, w, h)

                if confidence >= confidence_threshold:
                    faces.append([x, y, x + w, y + h])
                    confidences.append(confidence)

            return faces, confidences

        except Exception as e:
            logging.error(f"Python face detection failed: {e}")
            return [], []

    def detect_skin_regions(self, image):
        """Very basic skin tone detection"""
        try:
            # Convert to numpy array for processing
            img_array = np.array(image)

            # Simple RGB-based skin detection
            # These are very approximate ranges for demonstration
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

            # Basic skin color conditions (can be improved)
            skin_mask = (
                    (r > 95) & (g > 40) & (b > 20) &
                    ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 15) &
                    (np.abs(r - g) > 15) & (r > g) & (r > b)
            )

            return skin_mask

        except Exception as e:
            logging.error(f"Skin detection failed: {e}")
            return np.zeros((image.height, image.width), dtype=bool)

    def find_face_candidates(self, skin_mask, width, height):
        """Find potential face regions"""
        try:
            from scipy import ndimage

            # Label connected components
            labeled_mask, num_features = ndimage.label(skin_mask)

            candidates = []

            for i in range(1, num_features + 1):
                region = (labeled_mask == i)

                if np.sum(region) > 1000:  # Minimum area
                    coords = np.where(region)
                    y_coords, x_coords = coords[0], coords[1]

                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)

                    w = x_max - x_min
                    h = y_max - y_min

                    # Basic face-like aspect ratio
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.6 <= aspect_ratio <= 1.4:  # Roughly square to slightly rectangular
                        candidates.append((x_min, y_min, w, h))

            return candidates

        except Exception as e:
            logging.error(f"Face candidate finding failed: {e}")
            return []

    def filter_face_candidates(self, candidates, image_width, image_height):
        """Filter candidates based on reasonable face properties"""
        filtered = []

        for x, y, w, h in candidates:
            # Reasonable size constraints
            min_face_size = max(50, image_width * 0.05, image_height * 0.05)
            max_face_size = min(image_width * 0.8, image_height * 0.8)

            if min_face_size <= w <= max_face_size and min_face_size <= h <= max_face_size:
                # Not too close to edges
                margin = 10
                if (x >= margin and y >= margin and
                        x + w <= image_width - margin and
                        y + h <= image_height - margin):
                    filtered.append((x, y, w, h))

        return filtered

    def calculate_confidence(self, image, x, y, w, h):
        """Calculate confidence score for a candidate region"""
        try:
            confidence = 0.5  # Base confidence

            # Check if region contains potential facial features
            face_region = image.crop((x, y, x + w, y + h))

            # Convert to grayscale for analysis
            gray_face = face_region.convert('L')
            face_array = np.array(gray_face)

            # Simple symmetry check (faces are roughly symmetrical)
            if w > 0:
                left_half = face_array[:, :w // 2]
                right_half = face_array[:, w // 2:]
                right_half_flipped = np.fliplr(right_half)

                # Compare halves (very basic symmetry measure)
                if left_half.shape == right_half_flipped.shape:
                    diff = np.mean(np.abs(left_half - right_half_flipped))
                    max_diff = 255  # Maximum possible difference
                    symmetry_score = 1 - (diff / max_diff)
                    confidence += symmetry_score * 0.3

            # Size-based confidence
            area_ratio = (w * h) / (image.width * image.height)
            if 0.02 <= area_ratio <= 0.3:  # Reasonable face size range
                confidence += 0.2

            return min(confidence, 1.0)  # Cap at 1.0

        except Exception as e:
            logging.error(f"Confidence calculation failed: {e}")
            return 0.3

    def analyze_image(self, image, confidence_threshold=0.3):
        """Main analysis function"""
        try:
            # Detect faces
            faces, confidences = self.detect_faces_python(image, confidence_threshold)

            # Create processed image with annotations
            processed_image = image.copy()
            draw = ImageDraw.Draw(processed_image)

            results = {
                'faces_detected': len(faces),
                'face_data': [],
                'processed_image': processed_image,
                'total_confidence': 0.0,
                'method': 'pure_python'
            }

            for i, (x1, y1, x2, y2) in enumerate(faces):
                confidence = confidences[i] if i < len(confidences) else 0.5

                # Choose color based on confidence
                if confidence > 0.7:
                    color = "green"
                elif confidence > 0.4:
                    color = "orange"
                else:
                    color = "red"

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Draw label
                label = f"Face {i + 1}: {confidence:.1%}"
                draw.text((x1, y1 - 25), label, fill=color)

                # Calculate face info
                face_width = x2 - x1
                face_height = y2 - y1
                area_percentage = (face_width * face_height) / (image.width * image.height) * 100

                # Determine size category
                if area_percentage > 10:
                    size_category = "Large"
                elif area_percentage > 5:
                    size_category = "Medium"
                else:
                    size_category = "Small"

                # Determine position
                center_x = (x1 + x2) / 2
                if center_x < image.width * 0.4:
                    position = "Left"
                elif center_x > image.width * 0.6:
                    position = "Right"
                else:
                    position = "Center"

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

            if results['faces_detected'] > 0:
                results['total_confidence'] = sum(confidences) / len(confidences)

            return results

        except Exception as e:
            logging.error(f"Image analysis failed: {e}")
            return {
                'faces_detected': 0,
                'face_data': [],
                'processed_image': image,
                'total_confidence': 0.0,
                'method': 'error'
            }


def main():
    # Initialize detector
    detector = PurePythonFaceDetector()

    # Sidebar
    st.sidebar.header("⚙️ Detection Settings")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,  # Lower default for basic detection
        step=0.1,
        help="Higher values = fewer but more confident detections"
    )

    show_detection_info = st.sidebar.checkbox(
        "Show Detailed Detection Info",
        value=True,
        help="Display additional information about each detection"
    )

    # System info
    st.sidebar.header("🔧 System Info")
    st.sidebar.success("✅ Pure Python Implementation")
    st.sidebar.info("""
    **No OpenCV Required**

    Using:
    - PIL for image processing
    - NumPy for calculations
    - Basic computer vision algorithms
    """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image for face detection",
            type=["jpg", "jpeg", "png"],
            help="For best results, use clear front-facing photos with good lighting"
        )

        if uploaded_file is not None:
            try:
                # Load and display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

                # Display file info
                file_size = len(uploaded_file.getvalue()) / 1024
                st.caption(f"Image size: {image.width} × {image.height} pixels • {file_size:.1f} KB")

                # Analyze image
                with st.spinner("🔍 Detecting faces with pure Python..."):
                    results = detector.analyze_image(image, confidence_threshold)

                # Display results
                st.subheader("📊 Detection Results")

                if results['faces_detected'] > 0:
                    st.success(f"✅ **{results['faces_detected']} face(s) detected**")

                    if results['total_confidence'] > 0:
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

                                # Progress bar with integer value
                                confidence_percent = int(face['confidence'] * 100)
                                st.progress(confidence_percent, text=f"Detection Confidence: {face['confidence']:.1%}")

                else:
                    st.warning("❌ No faces detected")
                    st.info("""
                    **Tips for better detection:**
                    - Try lowering the confidence threshold (0.1-0.3)
                    - Use clear, front-facing photos
                    - Ensure good, even lighting
                    - Avoid heavily rotated or obscured faces
                    - Make sure faces are clearly visible
                    """)

            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                st.info("Please try a different image file.")

    with col2:
        st.subheader("🎯 Detection Output")
        if uploaded_file is not None and 'results' in locals():
            if results['faces_detected'] > 0:
                # Display processed image
                st.image(results['processed_image'], use_container_width=True,
                         caption=f"Detected {results['faces_detected']} face(s) - Pure Python")

                # Confidence color guide
                st.caption("🎨 Confidence Colors: 🟢 High (>70%) | 🟡 Medium (40-70%) | 🔴 Low (<40%)")

                # Download button
                buf = BytesIO()
                results['processed_image'].save(buf, format="JPEG", quality=95)

                st.download_button(
                    label="📥 Download Result Image",
                    data=buf.getvalue(),
                    file_name="python_face_detection_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            else:
                st.info("👆 No faces detected in the image")
                if uploaded_file is not None:
                    st.image(image, use_container_width=True, caption="Original Image")
        else:
            st.info("""
            ## 🎯 Pure Python Face Detection

            **How it works:**
            - Uses basic image processing algorithms
            - No external dependencies beyond Python
            - Skin tone and pattern detection
            - Symmetry analysis
            - Size and position filtering

            **Best Practices:**
            - Clear, well-lit frontal faces
            - Good contrast between face and background
            - Reasonable image quality
            - Multiple faces supported

            **Note:** This is a basic implementation. For production use, consider integrating with cloud-based face detection APIs.
            """)

    # Features and limitations
    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("✅ Features")
        st.markdown("""
        - **Pure Python**: No OpenCV or external dependencies
        - **Streamlit Cloud Compatible**: Works anywhere
        - **Multiple Face Detection**: Handles multiple faces
        - **Confidence Scoring**: Color-coded confidence levels
        - **Detailed Analysis**: Size, position, and area information
        - **Download Results**: Save processed images
        - **Mobile Friendly**: Responsive design
        - **Fast Processing**: Optimized algorithms
        """)

    with col4:
        st.subheader("⚠️ Limitations")
        st.markdown("""
        - **Basic Algorithm**: Simple computer vision approach
        - **Accuracy**: Lower than professional face detectors
        - **Lighting Sensitive**: Works best with good lighting
        - **Frontal Faces**: Optimized for front-facing photos
        - **Simple Scenes**: Best with clear backgrounds
        - **No Advanced Features**: Basic detection only

        *For high-accuracy needs, consider cloud APIs like:*
        - Google Vision API
        - Amazon Rekognition  
        - Microsoft Face API
        - OpenCV with proper setup
        """)


if __name__ == "__main__":
    main()