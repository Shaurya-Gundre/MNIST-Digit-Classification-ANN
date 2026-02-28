# MNIST Digit Classification using Artificial Neural Network (ANN)

## Project Overview
This project aims to classify handwritten digits (0–9) from the MNIST dataset using an Artificial Neural Network (ANN).

The problem is a **multi-class classification task**, where the target variable represents digits from 0 to 9.

---

## Objective
To build a deep learning model that can accurately recognize handwritten digits using neural networks.

---

## Dataset Information
- Dataset: MNIST Handwritten Digits
- Total Training Images: 60,000
- Total Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Number of Classes: 10 (Digits 0–9)

---

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- Scikit-learn

---

## Data Preprocessing
The following preprocessing steps were performed:

- Normalized pixel values (scaled between 0 and 1)
- Flattened 28×28 images into 784-dimensional vectors
- Converted labels into categorical format (One-Hot Encoding)

---

## Model Architecture
Artificial Neural Network (ANN):

- Input Layer: 784 neurons
- Hidden Layer 1: 128 neurons (ReLU activation)
- Hidden Layer 2: 64 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)

Loss Function: Categorical Crossentropy  
Optimizer: Adam  
Evaluation Metric: Accuracy  

---

## Model Performance
- Training Accuracy: ~98%
- Testing Accuracy: ~97%

Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Classification Report

The model performs well in recognizing handwritten digits with high accuracy.

---

## How to Run the Project

1. Clone the repository
2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook
4. Run all cells sequentially

---

## Future Improvements
- Add Convolutional Neural Network (CNN) for higher accuracy
- Implement Dropout for regularization
- Deploy as a web application
- Use real-time digit recognition

---

## Repository Structure

```
MNIST-Digit-Classification-ANN/
│── MNIST_Digit_Classification.ipynb
│── requirements.txt
│── README.md
```

---

## Conclusion
This project demonstrates how Artificial Neural Networks can effectively classify handwritten digits. The model achieves high accuracy and showcases the power of deep learning in image classification tasks.
