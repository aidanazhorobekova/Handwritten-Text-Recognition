# Handwritten Text Recognition

## Overview
This project demonstrates an end-to-end pipeline for Handwritten Text Recognition (HTR) using Optical Character Recognition (OCR) techniques. The goal is to extract text from images containing handwritten characters, providing an accurate method for recognizing and transcribing handwritten words into machine-readable text.

## Dataset
This project uses the "Handwritten Text Recognition" dataset, which is available on Kaggle. To access it, please visit:
[Link to dataset on Kaggle](https://www.kaggle.com/datasets/aidanazhorobekova/handwritten-text)
The dataset consists of images containing handwritten text along with corresponding labels. The images are in `.jpg` or `.png` format, and the labels are stored in `.csv` files. The goal is to recognize the handwritten text in each image and convert it into a machine-readable format.

## Model Architecture
We use a combination of CNN (for feature extraction) and RNN (for sequence learning). The CNN layers help extract relevant features from images, while the RNN layers handle the sequential nature of text recognition.

## Usage
1. Clone the repository and install the required dependencies from `requirements.txt`.
2. Download the dataset (images and labels) and preprocess the data.
3. Train the model on your local machine or Kaggle notebook.
4. Use the trained model to predict handwritten text from images.

### Requirements
* TensorFlow
* Pandas
* NumPy
* OpenCV
* Matplotlib

### Example Code Snippet for Text Recognition:
```python
import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('htr_model.h5')

# Load an image for prediction
image = cv2.imread('path_to_image.jpg', 0)
image = cv2.resize(image, (128, 32))  # Resize to match model input size
image = np.expand_dims(image, axis=-1)
image = image / 255.0  # Normalize

# Make prediction
pred = model.predict(np.expand_dims(image, axis=0))
print(f"Extracted Text: {decode_prediction(pred)}")
