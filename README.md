# 🩺 Skin Disease Detection with Deep Learning

This is a deep learning-based web application that classifies images of skin diseases using a Convolutional Neural Network (CNN). Users can upload an image through a simple web interface, and the model will predict the most likely skin condition.

---

## 📷 Demo

Upload an image of a skin condition, and the app will display:

- The predicted disease class (e.g., Acne, Eczema, Psoriasis, etc.)
- The model's confidence score
- A preview of the uploaded image

---

## 📁 Project Structure

```
SkinDiseaseDetection/
├── app.py                      # Flask web application
├── model/
│   └── skin_disease_model.h5   # Trained Keras model
├── scripts/
│   ├── train.py                # Script to train the model
│   ├── test.py                 # Script to test the model
│   └── predict.py              # Script to predict single image from command line
├── static/
│   └── uploads/                # Folder for uploaded images
├── templates/
│   └── index.html              # Frontend HTML
├── data/                       # Dataset (not included)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🧠 Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Input Size:** 224 x 224 RGB images
- **Output:** 9 skin disease classes
- **Classes:**
  - Basal Cell Carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Acitinic Keratosis
  - Pigmented Benign Keratosis
  - Seborrheic Keratosis
  - Squamous Cell Carcinoma
  - Vascular Lesion
#### Dataset taken from kaggle - https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification

---

## ⚙️ Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SkinDiseaseDetection.git
cd SkinDiseaseDetection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧾 Dataset Structure

> ⚠️ Dataset is not included in the repository due to size constraints.

You must manually download a skin disease dataset and arrange it like this:

```
data/
├── train/
│   ├── Acne/
│   ├── Eczema/
│   └── ...
├── test/
│   ├── Acne/
│   ├── Eczema/
│   └── ...
```

---

## 🚀 Running the App

### Option 1: Command Line Prediction

```bash
python scripts/predict.py
```

You will be prompted to enter the path to the image.

### Option 2: Flask Web Application

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---
