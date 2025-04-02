import streamlit as st  # ✅ Move set_page_config before all imports
st.set_page_config(page_title="MRI Stroke Detection", page_icon="🧠", layout="wide")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

IMAGE_SIZE = (128, 128)  # Matches training size
TIME_STEPS = 5
MODEL_PATH = 'stroke_detection_model.keras'
image_path = 'brain.png'

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Define label encoder with the correct class names
label_encoder = LabelEncoder()
label_encoder.fit(['Normal', 'Stroke'])

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    if img is None:
        return None

    img = cv2.resize(img, IMAGE_SIZE)  # Resize to match model input size
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)   # Add channel dimension
    
    return img

# Function to create a sequence from a single image (replicate image for TIME_STEPS)
def create_sequence(image):
    sequence = np.repeat(np.expand_dims(image, axis=0), TIME_STEPS, axis=0)
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    return sequence

# ---------------------------
# Streamlit UI Design
# ---------------------------

# Sidebar for Navigation
with st.sidebar:
    st.image(image_path, width=200)
    st.title("MRI Stroke Detection")
    st.write("This Tool assists in stroke detection from MRI scans.")

    st.subheader("About this tool")
    st.write("""
    This deep learning model analyzes MRI scans and predicts whether the brain shows signs of a stroke.
    
    - Uses **Convolutional Neural Networks (CNN)** & **LSTM (Long Short-Term Memory)**
    - Trained on **MRI brain scans** for accurate detection.
    - Provides a **confidence score** for better decision-making.
    """)
    st.markdown("---")
    st.write("Developed for clinical use. **For medical professionals only.**")

# Main Layout
st.title("🧠 Stroke Detection from MRI Scans")
st.write("Upload an MRI scan to get an instant stroke detection result.")

# Upload Image
uploaded_file = st.file_uploader("Upload MRI Scan (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)

    if image is None:
        st.error("Error processing the image. Please upload a valid MRI scan.")
    else:
        # Display Uploaded Image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Uploaded MRI Scan")
            st.image(image.squeeze(), caption="MRI Scan", use_column_width=True)

        # Make Prediction
        sequence_input = create_sequence(image)
        st.write("🔍 **Analyzing MRI scan...**")
        prediction = model.predict(sequence_input)

        # Get Predicted Class & Confidence Percentage
        pred_class = np.argmax(prediction, axis=1)
        predicted_label = label_encoder.inverse_transform(pred_class)[0]
        confidence_percentage = np.max(prediction) * 100  # Get highest probability

        # Display Prediction Result
        with col2:
            st.subheader("Prediction Results")
            if predicted_label == "Stroke":
                st.error(f"🚨 **Stroke Detected** with {confidence_percentage:.2f}% confidence.")
            else:
                st.success(f"✅ **No Stroke Detected** with {confidence_percentage:.2f}% confidence.")

        # Confidence Score Visualization
        st.subheader("Prediction Confidence Breakdown")
        fig, ax = plt.subplots(figsize=(6, 3))  # Reduce size further
        sns.barplot(x=label_encoder.classes_, y=prediction[0], palette=["blue", "red"], ax=ax)
        ax.set_xlabel("Class", fontsize=10)  # Reduce font size
        ax.set_ylabel("Prediction Probability", fontsize=10)
        ax.set_title("Confidence Scores", fontsize=12)
        ax.tick_params(axis='both', labelsize=8)  # Reduce tick label sizes
        st.pyplot(fig)

        # Additional Report for Clinicians
        st.subheader("📊 Clinical Report")
        st.write(f"- **Predicted Condition:** {predicted_label}")
        st.write(f"- **Confidence Score:** {confidence_percentage:.2f}%")
        st.write("- **Model Used:** CNN + LSTM Deep Learning")
        st.write("- **Intended Use:** Assist radiologists in stroke diagnosis from MRI scans.")
        
        st.markdown("---")
        st.write("🔬 **Note:** This Model is for **clinical decision support only**. **Final diagnosis must be made by a doctor.**")
