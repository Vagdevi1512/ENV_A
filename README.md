
# Coral Classification using MobileNetV2
### Deep Learning Model for Coral Health Detection (TensorFlow/Keras)

A deep-learning project to classify coral images into **two categories** using **MobileNetV2 transfer learning**.  
The project includes data preprocessing, augmentation, training, evaluation, visualization, and exporting the model in both **H5** and **TFLite** formats for mobile/edge deployment.

---

## Dataset Structure

Each subfolder must contain images downloaded from:

Dataset Link: https://www.kaggle.com/datasets/aneeshdighe/corals-classification
Here the Training, Validation, Testing should be extracted from downloads and keep these separately in the folder where remaining code is present
```
Training/
    ├── bleached_corals/
    └── healthy_corals/

Validation/
    ├── bleached_corals/
    └── healthy_corals/

Testing/
    ├── bleached_corals/
    └── healthy_corals/
```
---

## Features

- Transfer Learning with **MobileNetV2**
- Strong Data Augmentation  
- Automatic Class Weighting  
- EarlyStopping & ReduceLROnPlateau  
- Confusion Matrix + Classification Report  
- Saves `.h5` and `.tflite` models  

---

## Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

GPU (optional):

```bash
pip install tensorflow-gpu
```

---

## Model Architecture

```
MobileNetV2 (pretrained, frozen)
↓
GlobalAveragePooling2D
↓
Dense(128, ReLU)
↓
Dropout(0.5)
↓
Dense(2, Softmax)
```

---

## Run the Training Script

```bash
streamlit run app_Mobile_Net.py
```

---

## Evaluation Output

- Accuracy/Loss curves  
- Confusion matrix  
- Classification report  

---

## Output Files

```
CNN_model_MobileNet_CoralsClassification.h5
CNN_model_MobileNet_CoralsClassification.tflite
```

