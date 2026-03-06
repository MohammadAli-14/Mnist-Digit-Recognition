"""
MNIST Digit Recognition Web Application
A Streamlit app to test the trained CNN model for handwritten digit recognition.
"""

import os
import sys
import logging
import traceback
import importlib
from datetime import datetime

# Configure logging before imports
log_file = f"mnist_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*50)
logger.info("MNIST App Starting...")
logger.info(f"Log file: {log_file}")

try:
    import streamlit as st
    logger.info(f"Streamlit version: {st.__version__}")
except ImportError as e:
    print(f"ERROR: Failed to import streamlit: {e}")
    raise

try:
    import numpy as np
    logger.info(f"NumPy version: {np.__version__}")
except ImportError as e:
    logger.error(f"Failed to import numpy: {e}")
    raise

try:
    from PIL import Image
    logger.info("Pillow imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PIL: {e}")
    raise

try:
    keras = importlib.import_module("keras")
    logger.info(f"Keras version: {keras.__version__}")
except ImportError as e:
    logger.error(f"Failed to import keras: {e}")
    raise

try:
    from streamlit_drawable_canvas import st_canvas
    logger.info("streamlit_drawable_canvas imported successfully")
except ImportError as e:
    logger.error(f"Failed to import streamlit_drawable_canvas: {e}")
    raise

try:
    import cv2
    logger.info(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    logger.error(f"Failed to import cv2: {e}")
    raise

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Model file paths
MODEL_PATHS = [
    'mnist_cnn_model.keras',
    'mnist_cnn_model.h5'
]

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained MNIST CNN model with proper error handling."""
    logger.info("Attempting to load model...")
    
    # Check which model files exist
    existing_models = []
    for path in MODEL_PATHS:
        if os.path.exists(path):
            existing_models.append(path)
            logger.info(f"Found model file: {path}")
        else:
            logger.warning(f"Model file not found: {path}")
    
    if not existing_models:
        error_msg = "No model files found. Please ensure mnist_cnn_model.keras or mnist_cnn_model.h5 exists in the working directory."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    # Try loading each available model
    for model_path in existing_models:
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            
            if model_path.endswith('.keras'):
                model = keras.models.load_model(model_path)
            elif model_path.endswith('.h5'):
                model = keras.models.load_model(model_path)
            else:
                continue
            
            # Verify model is loaded correctly
            if model is None:
                raise ValueError("Model loaded as None")
            
            # Test model with dummy input
            logger.info("Testing model with dummy input...")
            dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
            test_prediction = model.predict(dummy_input, verbose=0)
            
            if test_prediction.shape != (1, 10):
                raise ValueError(f"Unexpected output shape: {test_prediction.shape}, expected (1, 10)")
            
            logger.info(f"[OK] Model loaded successfully from {model_path}")
            logger.info(f"  Model input shape: {model.input_shape}")
            logger.info(f"  Model output shape: {model.output_shape}")
            logger.info(f"  Model summary logged to file")
            
            # Log model summary
            try:
                model.summary(print_fn=lambda x: logger.info(x))
            except Exception as e:
                logger.warning(f"Could not log model summary: {e}")
            
            return model
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {model_path} - {e}")
            continue
        except ValueError as e:
            logger.error(f"Model validation error for {model_path}: {e}")
            continue
        except RuntimeError as e:
            logger.error(f"Runtime error loading model {model_path}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_path}: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # If we get here, all models failed to load
    error_msg = "Failed to load any model file. Check logs for details."
    logger.error(error_msg)
    st.error(error_msg)
    return None

def preprocess_image(image):
    """
    Preprocess the input image for model prediction.
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixel values
    - Reshape for model input
    """
    logger.debug(f"Preprocessing image with shape: {image.shape}, dtype: {image.dtype}")
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                logger.debug("Converted RGBA to grayscale")
            elif image.shape[2] == 3:  # RGB/BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                logger.debug("Converted BGR to grayscale")
            else:
                image = image[:, :, 0]  # Take first channel
                logger.debug("Took first channel for grayscale")
        
        # Resize to 28x28
        original_shape = image.shape
        image = cv2.resize(image, (28, 28))
        logger.debug(f"Resized image from {original_shape} to {image.shape}")
        
        # Invert colors if background is white (MNIST has black background)
        mean_value = np.mean(image)
        logger.debug(f"Image mean value: {mean_value}")
        
        if mean_value > 127:
            image = 255 - image
            logger.debug("Inverted colors (white background detected)")
        
        # Normalize to [0, 1]
        image = image.astype('float32') / 255.0
        logger.debug(f"Normalized image: min={image.min():.4f}, max={image.max():.4f}")
        
        # Reshape for model input (1, 28, 28, 1)
        image = np.expand_dims(image, axis=(0, -1))
        logger.debug(f"Final image shape: {image.shape}")
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        logger.error(traceback.format_exc())
        raise

def predict_digit(model, image):
    """Make prediction on the preprocessed image."""
    logger.info("Making prediction...")
    
    try:
        prediction = model.predict(image, verbose=0)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        
        logger.info(f"Prediction: class={predicted_class}, confidence={confidence:.2f}%")
        logger.debug(f"Full prediction probabilities: {prediction[0]}")
        
        return predicted_class, confidence, prediction[0]
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.error(traceback.format_exc())
        raise

# Main app
def main():
    logger.info("Starting main app...")
    
    st.title("MNIST Digit Recognition")
    st.markdown("**Draw a digit (0-9) or upload an image to test the trained CNN model.**")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("""
        Failed to load the model. 
        
        Please ensure the model file exists in the working directory:
        - mnist_cnn_model.keras (preferred)
        - mnist_cnn_model.h5 (fallback)
        
        Check the log file for details: {log_file}
        """)
        logger.error("App terminated: Model failed to load")
        return
    
    st.success("Model loaded successfully!")
    logger.info("Model loaded, proceeding with app...")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])
    
    with tab1:
        st.markdown("### Draw a digit below:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a canvas for drawing
            try:
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
                logger.debug(f"Canvas created: {type(canvas_result)}")
            except Exception as e:
                logger.error(f"Error creating canvas: {e}")
                st.error(f"Error creating drawing canvas: {e}")
                return
        
        with col2:
            st.markdown("**Instructions:**")
            st.markdown("1. Draw a digit (0-9)")
            st.markdown("2. Click 'Predict'")
            st.markdown("3. Use sidebar to clear")
        
        if st.button("Predict", key="predict_draw"):
            logger.info("Predict button clicked (draw)")
            
            if canvas_result.image_data is not None:
                try:
                    # Get the drawn image
                    image = canvas_result.image_data.astype(np.uint8)
                    logger.info(f"Canvas image shape: {image.shape}")
                    
                    # Check if something was drawn
                    pixel_sum = np.sum(image[:, :, :3])
                    logger.info(f"Total pixel sum: {pixel_sum}")
                    
                    if pixel_sum > 0:
                        with st.spinner("Processing..."):
                            # Preprocess and predict
                            processed = preprocess_image(image)
                            predicted_class, confidence, probabilities = predict_digit(model, processed)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### Prediction Results")
                            
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
                            
                        logger.info("Prediction displayed successfully")
                    else:
                        st.warning("Please draw a digit first!")
                        logger.warning("Predict clicked but canvas is empty")
                        
                except Exception as e:
                    error_msg = f"Error processing drawing: {e}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    st.error(error_msg)
            else:
                st.warning("Please draw a digit first!")
                logger.warning("Predict clicked but canvas_result.image_data is None")
    
    with tab2:
        st.markdown("### Upload an image of a handwritten digit:")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="uploader"
        )
        
        if uploaded_file is not None:
            logger.info(f"File uploaded: {uploaded_file.name}, type: {uploaded_file.type}")
            
            try:
                # Read and display the uploaded image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                logger.info(f"Uploaded image shape: {image_array.shape}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", width=200)
                
                if st.button("Predict", key="predict_upload"):
                    logger.info("Predict button clicked (upload)")
                    
                    try:
                        with st.spinner("Processing..."):
                            # Preprocess and predict
                            processed = preprocess_image(image_array)
                            predicted_class, confidence, probabilities = predict_digit(model, processed)
                            
                            with col2:
                                st.image(processed[0, :, :, 0], caption="Processed (28x28)", width=100)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### Prediction Results")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Digit", predicted_class)
                            with col2:
                                st.metric("Confidence", f"{confidence:.1f}%")
                            
                            # Show probability distribution
                            st.markdown("#### Probability Distribution")
                            chart_data = {str(i): prob * 100 for i, prob in enumerate(probabilities)}
                            st.bar_chart(chart_data)
                            
                        logger.info("Prediction displayed successfully")
                        
                    except Exception as e:
                        error_msg = f"Error processing uploaded image: {e}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        st.error(error_msg)
                        
            except Exception as e:
                error_msg = f"Error loading uploaded image: {e}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(error_msg)
    
    # Sidebar with model info
    with st.sidebar:
        st.markdown("## Model Information")
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
        - 99%+ accuracy
        
        **Files:**
        - Model: mnist_cnn_model.keras
        - Logs: mnist_app_*.log
        """)
        
        # Show log file location
        st.markdown(f"**Log file**: `{log_file}`")
        
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            logger.info("Cache cleared by user")
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        logger.critical(traceback.format_exc())
        st.error(f"Application error: {e}")
        raise
