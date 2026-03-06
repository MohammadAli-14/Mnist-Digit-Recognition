# 🔢 MNIST Digit Recognition Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mnist-digit-recognition-final-version.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning application for handwritten digit recognition using deep learning. This project includes complete model training, evaluation, and deployment through an interactive web interface.
### Training Curves

<img src="/TrainingCurve.png" alt="Training Curves" width="800"/>


### Confusion Matrix

<img src="/ConfusionMatrix.png" alt="Confusion Matrix" width="600"/>

## 🌟 Live Demo

**Try it now:** [https://mnist-digit-recognition-final-version.streamlit.app/](https://mnist-digit-recognition-final-version.streamlit.app/)

Draw digits directly on the canvas or upload your own images to see real-time predictions!

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

- **🎨 Interactive Canvas Drawing**: Draw digits directly in the browser with an intuitive canvas interface
- **📁 Image Upload Support**: Upload your own handwritten digit images (PNG, JPG, JPEG)
- **🤖 Real-Time Predictions**: Instant digit recognition using a trained CNN model
- **📊 Confidence Scoring**: View prediction probabilities for all 10 digits (0-9)
- **📈 Probability Visualization**: Interactive bar charts showing model confidence distribution
- **🔄 Robust Model Loading**: Primary `.keras` format with `.h5` fallback for compatibility
- **⚡ Fast Performance**: Optimized preprocessing and inference pipeline
- **🌐 Production Deployment**: Fully deployed on Streamlit Cloud

## 🛠️ Tech Stack

### Machine Learning
- **TensorFlow/Keras** - Deep learning framework for model training and inference
- **NumPy** - Numerical computing for array operations

### Web Application
- **Streamlit** - Interactive web app framework
- **streamlit-drawable-canvas** - Canvas drawing component

### Image Processing
- **OpenCV (cv2)** - Advanced image preprocessing
- **Pillow (PIL)** - Image file handling
- **scikit-image** - Additional image utilities

### Development Tools
- **Jupyter Notebook** - Model training and experimentation
- **Matplotlib/Seaborn** - Data visualization

## 📁 Project Structure

```
Mnist-Digit-Recognition/
│
├── app.py                          # Streamlit web application
├── mnist-ml-task.ipynb            # Model training notebook
├── mnist_cnn_model.keras          # Trained model (primary format)
├── mnist_cnn_model.h5             # Trained model (backup format)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── assets/                         # Demo images and GIFs
│   ├── demo-canvas.gif
│   └── demo-upload.gif
│
├── sample_images/                  # Test digit images
│   ├── digit_0.png
│   ├── digit_1.png
│   └── ...
│
└── .streamlit/                     # Streamlit configuration
    └── config.toml
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/MohammadAli-14/Mnist-Digit-Recognition.git
cd Mnist-Digit-Recognition
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 💻 Usage

### Running the Web App Locally

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

### Training Your Own Model

1. Open `mnist-ml-task.ipynb` in Jupyter Notebook or JupyterLab:
```bash
jupyter notebook mnist-ml-task.ipynb
```

2. Run all cells to:
   - Load and preprocess MNIST dataset
   - Build CNN architecture
   - Train the model
   - Evaluate performance
   - Save the trained model

3. The trained model will be saved as:
   - `mnist_cnn_model.keras` (recommended)
   - `mnist_cnn_model.h5` (backup)

### Using the App

#### Canvas Drawing Mode:
1. Select "Draw on Canvas" tab
2. Use your mouse/touchpad to draw a digit (0-9)
3. Click "Predict Digit" button
4. View the predicted digit and confidence scores

#### Image Upload Mode:
1. Select "Upload Image" tab
2. Click "Choose an image..." and select a digit image
3. Supported formats: PNG, JPG, JPEG
4. Click "Predict Digit" button
5. View results with probability distribution

## 🧠 Model Architecture

### Convolutional Neural Network (CNN)

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Conv2D                      (None, 26, 26, 32)        320       
MaxPooling2D                (None, 13, 13, 32)        0         
Conv2D                      (None, 11, 11, 64)        18,496    
MaxPooling2D                (None, 5, 5, 64)          0         
Conv2D                      (None, 3, 3, 64)          36,928    
Flatten                     (None, 576)               0         
Dense                       (None, 64)                36,928    
Dropout                     (None, 64)                0         
Dense                       (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

### Key Components:
- **Input Layer**: 28x28 grayscale images
- **3 Convolutional Blocks**: Feature extraction with ReLU activation
- **MaxPooling Layers**: Spatial dimension reduction
- **Dropout Layer**: Regularization (50% dropout rate)
- **Output Layer**: 10 neurons with softmax activation (digits 0-9)

### Hyperparameters:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 10
- **Learning Rate**: Default (0.001)

## 📊 Results

### Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 99.2% | 98.9% | 98.7% |
| **Loss** | 0.024 | 0.038 | 0.041 |


## 🌐 Deployment

### Streamlit Cloud Deployment

This app is deployed on Streamlit Cloud for free hosting.

**Steps to deploy your own:**

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click "Deploy"

### Alternative Deployment Options

- **Heroku**: Use `setup.sh` and `Procfile` for deployment
- **AWS EC2**: Deploy on cloud VM with nginx reverse proxy
- **Google Cloud Run**: Containerize with Docker and deploy
- **Azure App Service**: Deploy Python web apps

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 5
enableCORS = false
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions:
- 🎨 Improve UI/UX design
- 🚀 Add model performance enhancements
- 📱 Mobile responsiveness improvements
- 🧪 Additional test cases
- 📝 Documentation improvements
- 🌍 Multi-language support

## 🐛 Known Issues & Roadmap

### Current Limitations:
- Model only recognizes single digits (not multi-digit numbers)
- Canvas drawing may be sensitive to stroke thickness
- Best results with centered, well-formed digits

### Future Enhancements:
- [ ] Add support for multi-digit number recognition
- [ ] Implement data augmentation for better generalization
- [ ] Add model explanation/visualization features (Grad-CAM)
- [ ] Create REST API for integration
- [ ] Add batch prediction capability
- [ ] Implement model versioning and A/B testing
- [ ] Add user authentication and prediction history

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohammad Ali**

- GitHub: [@MohammadAli-14](https://github.com/MohammadAli-14)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit**: For making ML deployment accessible
- **Community**: Thanks to all contributors and users!

## 📚 References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Keras API Reference](https://keras.io/api/)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ and Python**