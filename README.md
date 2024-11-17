
# **MRI Classification Using Deep Learning**

This project focuses on building a deep learning model to classify MRI scans into four categories for early detection and diagnosis of brain conditions, leveraging computer vision techniques and transfer learning.

## **Project Overview**

Medical imaging is a critical domain for applying deep learning, particularly for analyzing MRI scans. This project implements a **convolutional neural network (CNN)** using a pretrained **VGG16** model to classify MRI scans into four classes, enhancing the accuracy and efficiency of disease detection.

---

## **Key Features**
- **Model Architecture**:
  - VGG16 pretrained on ImageNet as the base model.
  - Custom dense head with Dropout and L2 regularization to improve performance.
- **Data Preprocessing**:
  - MRI scans normalized to match VGG16 input requirements.
  - Augmentation applied to simulate real-world variability (brightness shifts, rotations, zoom).
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
- Use the `model.fit()` call to initiate training with data augmentation, Dropout, and L2 regularization.

### 3. **Evaluation**
- Evaluate the model on validation/test data.
- Metrics like accuracy, precision, recall, and F1-score will be printed to the console.

---

## **Results**
- **Model Performance**:
  - Validation accuracy: **X%**
  - Precision/Recall: **X% / X%**
- **Confusion Matrix**:
  Visualized in the notebook to assess class-wise performance.

---

## **File Structure**
```
mri-classification/
├── data/                         # Directory for MRI scan data
├── data_analysis.ipynb           # Jupyter Notebook with model implementation
├── requirements.txt              # List of Python dependencies
├── README.md                     # Project documentation
└── model/                        # Directory to save trained models
```

---

## **Future Work**
- Explore other pretrained architectures like ResNet or EfficientNet for comparison.
- Fine-tune the VGG16 base layers to capture domain-specific features.
- Experiment with larger MRI datasets and unsupervised pretraining methods.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- Dataset provided by [dataset source, e.g., Kaggle/OASIS](#).
- Model architecture inspired by **VGG16** and transfer learning techniques.
- References:
  - [Keras Applications Documentation](https://keras.io/api/applications/)
  - [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
