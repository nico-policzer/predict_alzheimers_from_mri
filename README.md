
# **MRI Classification Using Deep Learning**

This project focuses on building a deep learning model to classify MRI scans into four categories for early detection and diagnosis of brain conditions, leveraging computer vision techniques and transfer learning.

## **Project Overview**

Medical imaging is a critical domain for applying deep learning, particularly for analyzing MRI scans. This project implements a **convolutional neural network (CNN)** using a pretrained **VGG16** model to classify MRI scans into four classes, enhancing the accuracy and efficiency of disease detection.

---

## **Key Features**
- **Model Architecture**:
  - VGG16 pretrained on ImageNet as the base model.
  - Custom dense head with L2 regularization to improve performance.
- **Data Preprocessing**:
  - MRI scans normalized to match VGG16 input requirements.
- **Evaluation Metrics**:
  - Accuracy, precision, recall, F1-score, and confusion matrix.
- **Tools and Frameworks**:
  - Python, TensorFlow/Keras, scikit-learn, NumPy, and Matplotlib.

---

## **Setup and Installation**

### Prerequisites
- Python 3.8 or later
- TensorFlow 2.x
- Required Python libraries: NumPy, pandas, scikit-learn, matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nico-policzer/predict_alzheimers_from_mri.git
   cd mri-predict_alzheimers_from_mri
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **How to Use**

### 1. **Data Preprocessing**
- Place MRI scan data in the `data/` directory.
- Normalize and resize images using the preprocessing pipeline in the notebook.

### 2. **Training the Model**
- Run the `data_analysis.ipynb` notebook to train the model.
- Use the `model.fit()` call to initiate training with data augmentation, and L2 regularization.

### 3. **Evaluation**
- Evaluate the model on validation/test data.
- Metrics like accuracy, precision, recall, and F1-score will be printed to the console.

---

## **Results**
- **Model Performance**:
  - Validation accuracy: **77%**
  - Precision/Recall: **73% / 74%**
- **Confusion Matrix**:
  Visualized in the notebook to assess class-wise performance.

---
## **Limitations**
- Dataset had very few mild Alzheimers examples, so model did never predicted mild

## **File Structure**
```
mri-classification/
├── data_analysis.ipynb           # Jupyter Notebook with model implementation
├── requirements.txt              # List of Python dependencies
├── README.md                     # Project documentation
└── model/                        # Directory to save trained models
```

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- Dataset provided by [https://huggingface.co/datasets/Falah/Alzheimer_MRI](#).
- Model architecture inspired by **VGG16** and transfer learning techniques.
