"""
MNIST Digit Recognition Web Application
A Streamlit app to test the trained CNN model for handwritten digit recognition.
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
import cv2

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained MNIST CNN model."""
    try:
        model = keras.models.load_model('mnist_cnn_model.keras')
        return model
    except:
        try:
            model = keras.models.load_model('mnist_cnn_model.h5')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

def preprocess_image(image):
    """
    Preprocess the input image for model prediction.
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixel values
    - Reshape for model input
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Invert colors if background is white (MNIST has black background)
    if np.mean(image) > 127:
        image = 255 - image
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    # Reshape for model input (1, 28, 28, 1)
    image = np.expand_dims(image, axis=(0, -1))
    
    return image

def predict_digit(model, image):
    """Make prediction on the preprocessed image."""
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_class, confidence, prediction[0]

# Main app
def main():
    st.title("🔢 MNIST Digit Recognition")
    st.markdown("**Draw a digit (0-9) or upload an image to test the trained CNN model.**")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure the model file exists.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["✏️ Draw Digit", "📤 Upload Image"])
    
    with tab1:
        st.markdown("### Draw a digit below:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a canvas for drawing
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        
        with col2:
            st.markdown("**Instructions:**")
            st.markdown("1. Draw a digit (0-9)")
            st.markdown("2. Click 'Predict'")
            st.markdown("3. Use sidebar to clear")
        
        if st.button("🔮 Predict", key="predict_draw"):
            if canvas_result.image_data is not None:
                # Get the drawn image
                image = canvas_result.image_data.astype(np.uint8)
                
                # Check if something was drawn
                if np.sum(image[:, :, :3]) > 0:
                    # Preprocess and predict
                    processed = preprocess_image(image)
                    predicted_class, confidence, probabilities = predict_digit(model, processed)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Digit", predicted_class)
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col3:
                        # Show preprocessed image
                        st.image(processed[0, :, :, 0], caption="Processed (28x28)", width=100)
                    
                    # Show probability distribution
                    st.markdown("#### Probability Distribution")
                    chart_data = {str(i): prob * 100 for i, prob in enumerate(probabilities)}
                    st.bar_chart(chart_data)
                else:
                    st.warning("Please draw a digit first!")
            else:
                st.warning("Please draw a digit first!")
    
    with tab2:
        st.markdown("### Upload an image of a handwritten digit:")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="uploader"
        )
        
        if uploaded_file is not None:
            # Read and display the uploaded image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", width=200)
            
            if st.button("🔮 Predict", key="predict_upload"):
                # Preprocess and predict
                processed = preprocess_image(image_array)
                predicted_class, confidence, probabilities = predict_digit(model, processed)
                
                with col2:
                    st.image(processed[0, :, :, 0], caption="Processed (28x28)", width=100)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Digit", predicted_class)
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Show probability distribution
                st.markdown("#### Probability Distribution")
                chart_data = {str(i): prob * 100 for i, prob in enumerate(probabilities)}
                st.bar_chart(chart_data)
    
    # Sidebar with model info
    with st.sidebar:
        st.markdown("## 📋 Model Information")
        st.markdown("""
        **Architecture:** CNN
        - 3 Convolutional layers
        - MaxPooling layers
        - Dense layer with Dropout
        - Softmax output (10 classes)
        
        **Input:** 28x28 grayscale image
        
        **Output:** Digit (0-9)
        
        **Training:** MNIST Dataset
        - 60,000 training images
        - 10,000 test images
        """)
        
        st.markdown("---")
        st.markdown("### 🛠️ Tips")
        st.markdown("""
        - Draw digits clearly
        - Center the digit
        - Use thick strokes
        - Fill most of the canvas
        """)

if __name__ == "__main__":
    main()
