MNIST Digit Classification using Artificial Neural Network
Project Overview

This project aims to classify handwritten digits (0–9) from grayscale images using an Artificial Neural Network (ANN).

The problem is a multi-class classification task, where the target variable represents digits from 0 to 9.

Objective

To build a Deep Learning model that can accurately recognize handwritten digits using the MNIST dataset.

Dataset Information

Total Training Images: 60,000

Total Testing Images: 10,000

Image Size: 28 × 28 pixels

Image Type: Grayscale

Number of Classes: 10 (Digits 0–9)

The MNIST dataset is a benchmark dataset widely used for image classification tasks.

Technologies Used

Python

NumPy

Matplotlib

TensorFlow

Keras

Scikit-learn

Seaborn

Data Preprocessing

The following preprocessing steps were performed:

Normalized pixel values from 0–255 to 0–1

Flattened 28×28 images into 784-length vectors

Used predefined train-test split

Artificial Neural Network Model
ANN Architecture

Input Layer: 784 neurons

Hidden Layer: 128 neurons (ReLU activation)

Output Layer: 10 neurons (Softmax activation)

The Softmax function outputs probability scores for each digit class.

Model Performance

Training Accuracy: ~98%

Test Accuracy: 97.8%

Evaluation Metrics:

Accuracy

Confusion Matrix

The model demonstrates strong performance in classifying handwritten digits.

How to Run the Project

Clone the repository

Install required libraries:

pip install -r requirements.txt

Open the Jupyter Notebook

Run all cells sequentially

Future Improvements

Implement Convolutional Neural Networks (CNN)

Perform hyperparameter tuning

Deploy the model using Flask or Streamlit

Repository Structure
MNIST-Digit-Classification-ANN/
│
├── MNIST_Digit_Classification_Using_Neural_Networks.ipynb
├── MNIST_Digit_Classification_Using_ANN_Presentation.pptx
├── PROJECT REPORT.docx
├── requirements.txt
└── README.md
Conclusion

This project demonstrates the practical implementation of Artificial Neural Networks for image classification tasks.
The ANN model successfully classifies handwritten digits with high accuracy.
