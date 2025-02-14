# Chest-XRay-Pneumonia-Detection

## Overview
This project focuses on the automatic classification of chest X-ray images to detect pneumonia. The model is based on deep learning techniques, utilizing **transfer learning** with **ResNet-18** for classification.

## Dataset
You can download the dataset from [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
The dataset consists of **5,863 chest X-ray images** categorized into:
- **Normal**: Healthy lung X-rays.
- **Pneumonia**: X-rays indicating pneumonia infection.

The dataset is **highly imbalanced**, with more pneumonia cases than normal cases. To address this, **data augmentation** and **class balancing techniques** are applied.

## Training the Model
### **1. Environment Setup**
Ensure you have Python installed, then install dependencies:
```
pip install -r requirements.txt
```

### **2. Training**
Run the training script to fine-tune ResNet-18 on the dataset:
```
python src/models/train.py
```
This script:
- Loads and preprocesses the dataset
- Applies data augmentation
- Trains the model using **CrossEntropyLoss** and **Adam optimizer**
- Saves the trained model in the `models/` directory

### **3. Model Evaluation**
To evaluate the model:
```
python src/models/evaluate.py
```
This script computes:

- **Accuracy, Precision, Recall, F1-score**
- **AUC-ROC Curve**
- **Confusion Matrix**
- **Grad-CAM Visualization** for model interpretability

## Model Deployment
The trained model is deployed using FastAPI.

### **1. Running the API**
#### **Option 1: API for Postman or External Clients**
To start the API server for receiving JSON responses:
```
uvicorn src.deploy.app_postman:app --reload
```
- **Endpoint:** `POST /predict/`
- **Input:** Chest X-ray image (as a file)
- **Output:** JSON with prediction (`Normal` or `Pneumonia`), probability, and base64-encoded images

#### **Option 2: Web Interface for Browsers**
To start a web interface with visualization:
```
uvicorn src.deploy.app_browser:app --reload
```
- The webpage displays the **original and processed image** along with the **classification result**

### **2. Testing with Postman**
- Open **Postman**
- Use **POST** request to `http://127.0.0.1:8000/predict/`
- Upload an X-ray image
- The API returns a JSON response with the classification result


