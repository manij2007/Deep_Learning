# 🧠 Handwritten Digit Classification Using Convolutional Neural Network

A deep learning project that implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

This project demonstrates the fundamental concepts of computer vision and image classification, including image preprocessing, convolutional layers, pooling, model training, performance evaluation, and custom image prediction.

---

## 📌 Project Overview

The goal of this project is to build a CNN model capable of recognizing handwritten digits ranging from **0 to 9**.

The model is trained on the MNIST dataset and uses convolutional and pooling layers to extract visual features from grayscale images before classifying them into one of ten digit classes.

The project also includes a custom image prediction pipeline that allows the trained model to predict a digit from an external image.

---

## 🎯 Objectives

* Understand the fundamental architecture of Convolutional Neural Networks.
* Work with the MNIST handwritten digit dataset.
* Preprocess grayscale images for CNN input.
* Apply convolution and max-pooling operations.
* Build and train an image classification model using Keras.
* Evaluate the model using accuracy, loss, and classification metrics.
* Predict handwritten digits from custom images.

---

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**, which is included in Keras.

### Dataset Information

| Property          | Value                           |
| ----------------- | ------------------------------- |
| Dataset           | MNIST                           |
| Task              | Multiclass Image Classification |
| Training Samples  | 60,000                          |
| Testing Samples   | 10,000                          |
| Image Dimensions  | 28 × 28                         |
| Image Channels    | 1 (Grayscale)                   |
| Number of Classes | 10                              |
| Classes           | Digits 0–9                      |

Each image represents a handwritten digit and contains pixel intensity values ranging from 0 to 255 before normalization.

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied before training the model.

### 1. Reshaping Images

The original MNIST images have the shape:

```text
(60000, 28, 28)
(10000, 28, 28)
```

The images were reshaped to include a single grayscale channel:

```python
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
```

Final image shapes:

```text
Training Data: (60000, 28, 28, 1)
Testing Data:  (10000, 28, 28, 1)
```

This format is compatible with the expected TensorFlow/Keras image input format:

```text
(batch_size, height, width, channels)
```

### 2. Pixel Normalization

The image data was converted to `float32` and normalized by dividing pixel values by `255.0`.

```python
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = X_train / 255.0
X_test = X_test / 255.0
```

This scales pixel values to the range:

```text
0.0 – 1.0
```

### 3. One-Hot Encoding

The target labels were converted into one-hot encoded vectors containing ten classes.

```python
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

Resulting target shapes:

```text
Training Labels: (60000, 10)
Testing Labels:  (10000, 10)
```

---

## 🏗️ CNN Architecture

The model was implemented using the Keras Sequential API.

### Model Structure

```text
Input
  ↓
Conv2D (32 filters, 3×3 kernel, ReLU)
  ↓
MaxPooling2D (2×2)
  ↓
Conv2D (32 filters, 3×3 kernel, ReLU)
  ↓
MaxPooling2D (2×2)
  ↓
Flatten
  ↓
Dense (100 units, ReLU)
  ↓
Dropout (0.2)
  ↓
Dense (10 units, Softmax)
```

### Architecture Details

| Layer        | Configuration                       |
| ------------ | ----------------------------------- |
| Input        | 28 × 28 × 1                         |
| Conv2D       | 32 filters, 3 × 3 kernel, ReLU      |
| MaxPooling2D | 2 × 2 pool size                     |
| Conv2D       | 32 filters, 3 × 3 kernel, ReLU      |
| MaxPooling2D | 2 × 2 pool size                     |
| Flatten      | Converts feature maps into a vector |
| Dense        | 100 units, ReLU                     |
| Dropout      | Rate = 0.2                          |
| Output       | 10 units, Softmax                   |

### Model Parameters

| Metric                   |  Value |
| ------------------------ | -----: |
| Total Parameters         | 90,678 |
| Trainable Parameters     | 90,678 |
| Non-Trainable Parameters |      0 |

---

## ⚙️ Model Compilation

The model was compiled using the following configuration:

```python
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

| Configuration     | Value                    |
| ----------------- | ------------------------ |
| Optimizer         | Adam                     |
| Loss Function     | Categorical Crossentropy |
| Evaluation Metric | Accuracy                 |
| Output Activation | Softmax                  |

Since this is a multiclass classification problem with one-hot encoded labels, `categorical_crossentropy` is used as the loss function.

---

## 🚀 Model Training

The model was trained using the following settings:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)
```

### Training Configuration

| Parameter          | Value         |
| ------------------ | ------------- |
| Epochs             | 10            |
| Validation Data    | Test Dataset  |
| Training Samples   | 60,000        |
| Validation Samples | 10,000        |
| Optimizer          | Adam          |
| Batch Size         | Keras Default |

> **Note:** The notebook uses the test dataset as validation data during training. For a more rigorous machine learning workflow, a separate validation set should be created and reserved independently from the final test set.

---

## 📈 Training Visualization

The notebook includes visualizations for monitoring model performance during training.

### Accuracy Curve

The training and validation accuracy are plotted across epochs to observe how classification performance changes during training.

### Loss Curve

The training and validation loss are plotted to analyze the model's optimization process and generalization behavior.

These visualizations can help identify potential issues such as overfitting or unstable training.

---

## 📊 Model Evaluation

The trained model was evaluated using the MNIST test dataset.

### Final Evaluation Results

| Metric             |     Result |
| ------------------ | ---------: |
| Test Accuracy      | **99.18%** |
| Test Loss          | **0.0363** |
| Evaluation Samples |     10,000 |

The model achieved approximately **99.18% accuracy on the test dataset used in the notebook**.

```text
Accuracy: 0.9918000102043152
Loss:     0.03627542406320572
```

### Classification Report

The classification report shows strong performance across all ten digit classes.

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| 0     |      0.99 |   1.00 |     1.00 |
| 1     |      0.99 |   1.00 |     0.99 |
| 2     |      0.99 |   0.99 |     0.99 |
| 3     |      0.99 |   0.99 |     0.99 |
| 4     |      0.99 |   0.99 |     0.99 |
| 5     |      0.98 |   0.99 |     0.99 |
| 6     |      0.99 |   0.99 |     0.99 |
| 7     |      0.99 |   0.99 |     0.99 |
| 8     |      0.99 |   0.99 |     0.99 |
| 9     |      0.99 |   0.98 |     0.99 |

The model achieved an overall accuracy of approximately **99% across 10,000 test images**.

---

## 🔍 Custom Image Prediction

In addition to evaluating the model on the MNIST test dataset, the notebook includes a custom image prediction pipeline.

The process includes:

1. Loading an external image using Pillow.
2. Converting the image to grayscale.
3. Resizing the image to `28 × 28`.
4. Checking the image background and inverting pixel values when necessary.
5. Normalizing pixel values to the range `0–1`.
6. Reshaping the image into the expected CNN input format.
7. Generating a prediction using the trained model.

### Prediction Pipeline

```python
original_img = Image.open(image_path)

img = original_img.convert("L")
img = img.resize((28, 28))

img_array = np.array(img)

if img_array.mean() > 127:
    img_array = 255 - img_array

img_array = img_array.astype("float32")
img_array = img_array / 255.0

input_image = img_array.reshape(1, 28, 28, 1)

prediction = model.predict(input_image, verbose=0)
predicted_class = np.argmax(prediction)
```

### Example Prediction

The custom image included in the notebook was predicted as:

```text
Prediction: 5
```

The notebook also displays the original image alongside the processed image and predicted class.

> **Note:** Custom image performance can depend on image quality, background, digit positioning, stroke thickness, and similarity to the MNIST training data.

---

## 🧰 Technologies & Libraries

* **Python** — Programming language
* **TensorFlow/Keras** — Deep learning model development
* **NumPy** — Numerical operations and array manipulation
* **Matplotlib** — Data and training visualization
* **Pillow (PIL)** — Image loading and preprocessing
* **Scikit-learn** — Classification report and evaluation metrics

### Main Imports

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten
)

from keras.utils import to_categorical
from keras.datasets import mnist

from sklearn.metrics import classification_report
```

---

## 📁 Project Structure

```text
Convolutional_Neural_Network/
│
├── Conveloutional_Neurol_Network_1.ipynb
│
└── README.md
```

---

## ⚡ Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Convolutional_Neural_Network
```

### 3. Install Dependencies

```bash
pip install numpy matplotlib pillow scikit-learn tensorflow
```

### 4. Run the Notebook

Open the notebook using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open:

```text
Conveloutional_Neurol_Network_1.ipynb
```

Run the cells sequentially to load the dataset, preprocess the images, train the CNN model, evaluate its performance, and test custom image prediction.

---

## 🧠 Key Concepts Covered

This project provides practical experience with the following deep learning concepts:

* Convolutional Neural Networks (CNNs)
* Image classification
* Grayscale image processing
* Image reshaping and tensor dimensions
* Pixel normalization
* One-hot encoding
* Convolutional layers
* Max pooling
* Feature extraction
* Flattening feature maps
* Fully connected layers
* Dropout regularization
* Softmax classification
* Categorical crossentropy
* Model training and validation
* Classification reports
* Custom image inference

---

## 📚 Learning Outcomes

By completing this project, you can gain a better understanding of:

1. How CNNs process image data.
2. How convolutional filters extract visual patterns.
3. How pooling layers reduce spatial dimensions.
4. How feature maps are transformed into classification features.
5. How to prepare image data for TensorFlow/Keras.
6. How to train a multiclass image classification model.
7. How to evaluate classification performance using multiple metrics.
8. How to prepare and classify an external image using a trained model.

---

## 🔧 Possible Improvements

The current implementation can be improved through several extensions:

* Create a separate validation set instead of using the test set during training.
* Add data augmentation to improve robustness.
* Experiment with different convolutional filter sizes and numbers.
* Add Batch Normalization layers.
* Use Early Stopping to prevent unnecessary training.
* Tune the learning rate and batch size.
* Add a confusion matrix for more detailed error analysis.
* Display incorrectly classified images.
* Save the trained model for later inference.
* Improve preprocessing for custom handwritten images.
* Experiment with deeper CNN architectures.

---

## 📝 Conclusion

This project demonstrates how a Convolutional Neural Network can be used to classify handwritten digits from the MNIST dataset.

By combining convolutional layers, max-pooling layers, fully connected layers, and dropout regularization, the model achieved approximately **99.18% test accuracy** on the evaluated MNIST test set.

The project also extends beyond standard dataset evaluation by implementing a custom image prediction pipeline, providing practical experience with applying trained deep learning models to external images.

Overall, this project serves as a practical introduction to **Computer Vision, CNN architectures, and image classification using TensorFlow/Keras**.

---

## ⭐ Author

**Mani**

If you found this project useful, feel free to explore the repository and follow the learning journey through other machine learning and deep learning projects.

