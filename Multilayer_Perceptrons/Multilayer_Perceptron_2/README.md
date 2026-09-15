# 🌸 Iris Flower Classification with Multilayer Perceptron

A deep learning classification project that uses a **Multilayer Perceptron (MLP)** neural network to classify Iris flowers into three different species based on their morphological measurements.

This project demonstrates a complete workflow for applying a neural network to a small structured dataset, including **data preparation, feature scaling, model construction, training, visualization, and evaluation**.

---

## 🚀 Project Overview

The **Iris dataset** is one of the most well-known datasets in machine learning and pattern recognition.

In this project, a **Multilayer Perceptron (MLP)** is implemented using **Keras** to classify Iris flowers into three species:

* 🌱 Iris Setosa
* 🌱 Iris Versicolor
* 🌱 Iris Virginica

### 🎯 Objective

> **Predict the species of an Iris flower from its four measured features.**

The model receives four numerical measurements as input and produces a probability distribution over the three possible classes.

---

## 📊 Dataset

The project uses the built-in **Iris dataset** provided by `scikit-learn`.

The dataset contains:

* **150 samples**
* **4 numerical features**
* **3 classes**
* **50 samples per class**
* No missing values

### Features

| Feature      | Description                 |
| ------------ | --------------------------- |
| Sepal Length | Sepal length in centimeters |
| Sepal Width  | Sepal width in centimeters  |
| Petal Length | Petal length in centimeters |
| Petal Width  | Petal width in centimeters  |

### Target Classes

| Class           | Samples |
| --------------- | ------: |
| Iris Setosa     |      50 |
| Iris Versicolor |      50 |
| Iris Virginica  |      50 |

The dataset is therefore evenly distributed across the three target classes.

---

## 🧠 Multilayer Perceptron Architecture

The project uses a compact fully connected neural network designed for the four-dimensional input space.

### Architecture

```text
Input Layer
    │
    ▼
Dense (10) + ReLU
    │
    ▼
Dense (10) + ReLU
    │
    ▼
Dropout (30%)
    │
    ▼
Dense (3) + Softmax
    │
    ▼
3-Class Prediction
```

### Model Parameters

The network contains:

**193 trainable parameters**

Architecture from the notebook:

```python
model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(units=10, activation="relu"))
model.add(Dense(units=10, activation="relu"))
model.add(Dropout(rate=0.3))
model.add(Dense(units=3, activation="softmax"))
```

---

## 🔄 Machine Learning Pipeline

The project follows the following workflow:

```text
Iris Dataset
     ↓
Data Loading
     ↓
Feature / Target Separation
     ↓
Train / Test Split
     ↓
Min-Max Scaling
     ↓
One-Hot Encoding
     ↓
MLP Construction
     ↓
Model Compilation
     ↓
Model Training
     ↓
Training History Visualization
     ↓
Model Evaluation
     ↓
Classification Report
     ↓
Confusion Matrix
```

---

## ⚙️ Data Preparation

### 1. Feature & Target Separation

The Iris dataset is separated into:

```python
X = iris["data"]
y = iris["target"]
```

The resulting shapes are:

```text
X: (150, 4)
y: (150,)
```

---

### 2. Train/Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=101
)
```

Result:

```text
Training samples: 120
Testing samples: 30
Features: 4
```

---

### 3. Feature Scaling

A **Min-Max Scaler** is applied to normalize the input features:

```python
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

This scales the features before they are passed into the neural network.

---

### 4. One-Hot Encoding

The three target classes are converted into a one-hot representation:

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

This matches the three-neuron softmax output layer used by the model.

---

## ⚙️ Model Configuration

The network is compiled using:

| Component         | Configuration            |
| ----------------- | ------------------------ |
| Optimizer         | Adam                     |
| Learning Rate     | 0.01                     |
| Loss Function     | Categorical Crossentropy |
| Metric            | Accuracy                 |
| Hidden Activation | ReLU                     |
| Output Activation | Softmax                  |
| Dropout Rate      | 0.30                     |

The exact compilation configuration used in the notebook is:

```python
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.01),
    metrics=["accuracy"]
)
```

---

## 🏋️ Training

The model is trained for **100 epochs** using the test set as validation data:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_test, y_test)
)
```

The notebook records both training and validation:

* Accuracy
* Loss

during the training process.

---

## 📈 Training Visualization

Training history is visualized using Matplotlib.

### Accuracy

The notebook plots:

```python
history.history["accuracy"]
history.history["val_accuracy"]
```

### Loss

It also plots:

```python
history.history["loss"]
history.history["val_loss"]
```

These visualizations provide a view of how the model's training and validation performance changes across epochs.

---

## 📊 Model Evaluation

After training, the model is evaluated on the 30-sample test set.

### Test Results

The recorded evaluation result is:

| Metric   |      Result |
| -------- | ----------: |
| Accuracy |    **100%** |
| Loss     | **0.04843** |

```text
Accuracy : 1.0
Loss     : 0.048434533178806305
```

### Classification Report

The classification report recorded in the notebook is:

| Class            | Precision |   Recall | F1-Score | Support |
| ---------------- | --------: | -------: | -------: | ------: |
| Class 0          |      1.00 |     1.00 |     1.00 |      10 |
| Class 1          |      1.00 |     1.00 |     1.00 |      12 |
| Class 2          |      1.00 |     1.00 |     1.00 |       8 |
| **Accuracy**     |           |          | **1.00** |  **30** |
| **Macro Avg**    |  **1.00** | **1.00** | **1.00** |  **30** |
| **Weighted Avg** |  **1.00** | **1.00** | **1.00** |  **30** |

---

## 🔲 Confusion Matrix

A confusion matrix is also generated to visualize the model's classification results:

```python
ConfusionMatrixDisplay.from_predictions(
    y_test.argmax(axis=1),
    model.predict(X_test).argmax(axis=1)
)
```

The notebook includes a confusion-matrix visualization as part of the final evaluation stage.

---

## 🛠️ Technologies & Libraries

This project was developed using Python and the following libraries:

* **Python**
* **NumPy** — Numerical computation
* **Pandas** — Data manipulation
* **Matplotlib** — Visualization
* **Seaborn** — Data visualization
* **Scikit-learn** — Dataset, preprocessing, and evaluation
* **Keras** — Neural network implementation
* **TensorFlow** — Deep learning backend

### Main Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, RMSprop, SGD
```

---

## 📂 Project Structure

```text
.
├── Multilayer_Perceptron_2.ipynb
└── README.md
```

The dataset does not need to be downloaded separately because it is loaded directly through `sklearn.datasets.load_iris()`.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Multilayer_Perceptron_2.ipynb
```

The notebook can also be executed using **Google Colab**.

---

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Multilayer Perceptrons
* Dense layers
* ReLU activation
* Softmax activation
* Dropout regularization
* Min-Max feature scaling
* Train/Test splitting
* One-hot encoding
* Categorical cross-entropy
* Adam optimizer
* Neural network training
* Training history visualization
* Classification reports
* Precision
* Recall
* F1-score
* Confusion matrices
* Multi-class classification

---

## 🎯 Learning Objectives

The main purpose of this project is to understand how a **Multilayer Perceptron can be applied to a small tabular multi-class classification problem**.

The project provides hands-on practice with the complete deep learning workflow:

```text
Prepare Data
     ↓
Scale Features
     ↓
Encode Targets
     ↓
Build MLP
     ↓
Compile Model
     ↓
Train Model
     ↓
Analyze Learning Curves
     ↓
Evaluate Predictions
```

---

## 🔮 Possible Improvements

Future iterations of the project could explore:

* Hyperparameter tuning
* Different hidden-layer sizes
* Different learning rates
* Alternative optimizers such as SGD or RMSprop
* Different dropout rates
* Cross-validation
* More detailed model comparison
* ROC-AUC analysis
* Precision-Recall analysis
* Comparison with traditional machine learning algorithms

---

## 📌 Conclusion

This project demonstrates an end-to-end **Iris flower classification pipeline using a Multilayer Perceptron**.

Starting from the original four numerical features, the data is split, normalized, encoded, and passed through a compact neural network consisting of fully connected layers and dropout regularization.

The recorded experiment achieved:

```text
Test Accuracy: 100%
Test Loss:     0.04843
```

on the 30-sample test split used in the notebook.

The project serves as a practical introduction to applying **deep learning techniques to structured/tabular data** and provides a foundation for experimenting with more advanced MLP architectures and training strategies.

---

## ⭐ Project Focus

**Deep Learning • Multilayer Perceptron • Multi-Class Classification • Keras • TensorFlow • Scikit-learn • Iris Dataset**

