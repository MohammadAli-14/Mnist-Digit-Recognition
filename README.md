# MNIST Digit Recognition

Interactive Streamlit app to test a trained CNN on the MNIST handwritten digit dataset. Draw a digit or upload an image to get the predicted class, confidence, and full probability distribution.

## Project Contents
- `app.py` – Streamlit UI for drawing/uploading digits and running predictions
- `mnist_cnn_model.keras` – Trained TensorFlow/Keras model (preferred format)
- `mnist_cnn_model.h5` – H5 fallback for broader compatibility
- `mnist_cnn_weights.weights.h5` – Model weights (not required by the app)
- `requirements.txt` – Python dependencies
- `mnist-ml-task.ipynb` – Notebook used to train the model

## Prerequisites
- Python 3.9+ (tested with TensorFlow 2.x)
- A working virtual environment (recommended)
- Model file in the project root: `mnist_cnn_model.keras` or `mnist_cnn_model.h5`

## Setup
```bash
# (Optional) create and activate a venv
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Run the App
```bash
streamlit run app.py
```
Then open the provided local URL (default: http://localhost:8501).

## Using the App
- **Draw Digit** tab: draw a digit (0–9) on the canvas, click **Predict**.
- **Upload Image** tab: upload a PNG/JPG/JPEG/BMP of a single digit, click **Predict**.
- The app shows predicted digit, confidence, and a bar chart of class probabilities.

## Notes
- The app first tries to load `mnist_cnn_model.keras`, then falls back to `mnist_cnn_model.h5`.
- If you re-train the model, place the new file in the project root with the same name.
- Input is converted to grayscale, resized to 28×28, normalized to [0,1], and inverted if the background is white.

## Troubleshooting
- **Model load error**: ensure `mnist_cnn_model.keras` or `.h5` exists in the project root and matches TensorFlow 2.x.
- **Blank predictions on canvas**: make sure you draw with visible strokes; the app ignores empty canvases.
- **GPU issues**: TensorFlow will run on CPU if no GPU is available.
