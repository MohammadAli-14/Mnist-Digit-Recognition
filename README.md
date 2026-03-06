# MNIST Digit Recognition Web App

An end-to-end handwritten digit recognition project using a Convolutional Neural Network (CNN) trained on MNIST and deployed with Streamlit.

The project includes:
- `mnist-ml-task.ipynb` for data prep, training, evaluation, and model export
- `app.py` for interactive inference through drawing and image upload

## Key Features
- Real-time digit prediction from a drawing canvas
- Image upload support (`png`, `jpg`, `jpeg`, `bmp`)
- Confidence score and full probability distribution for classes `0-9`
- Robust model loading with `.keras` (primary) and `.h5` (fallback)
- Detailed runtime logging for easier debugging and reproducibility

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- OpenCV
- Pillow

## Repository Structure
- `app.py`: Streamlit inference app
- `mnist-ml-task.ipynb`: training and evaluation notebook
- `mnist_cnn_model.keras`: trained model (recommended format)
- `mnist_cnn_model.h5`: trained model fallback format
- `mnist_cnn_weights.weights.h5`: model weights only
- `requirements.txt`: pinned dependencies

## Model Pipeline
1. Load MNIST dataset
2. Normalize pixel values to `[0, 1]`
3. Reshape images to `(28, 28, 1)`
4. Train CNN model
5. Evaluate on test set
6. Save model for deployment

## Local Setup
```bash
python -m venv .venv
```

Windows PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the App
From the project root (important):
```bash
streamlit run app.py
```

If you run from inside `.venv/`, Streamlit will not find `app.py`.

## How to Use
1. Open the **Draw Digit** tab and draw a digit with your mouse.
2. Click **Predict** to see class, confidence, and probability chart.
3. Or use **Upload Image** to test a digit image file.

## Notebook + App Workflow
- Train and validate model in `mnist-ml-task.ipynb`
- Export model files (`.keras` / `.h5`)
- Use `app.py` to serve predictions in a web UI

## Troubleshooting
- `File does not exist: app.py`: run Streamlit from project root, not `.venv`.
- `No module named ...`: activate virtual environment and reinstall requirements.
- Model load errors: ensure `mnist_cnn_model.keras` or `mnist_cnn_model.h5` is present in project root.
- TensorFlow CPU info/warnings in terminal: informational only, app can still run correctly.

## Future Improvements
- Add model versioning and experiment tracking
- Add optional preprocessing preview controls in UI
- Add automated tests for inference and preprocessing steps

## Author
Ali
