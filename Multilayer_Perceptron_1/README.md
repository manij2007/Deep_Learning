# 📊 Telco Customer Churn Prediction with Multilayer Perceptron

A deep learning classification project that uses a **Multilayer Perceptron (MLP)** neural network to predict whether a telecommunications customer is likely to **churn**.

The project covers the complete machine learning workflow, from **exploratory data analysis and preprocessing** to **neural network construction, training, visualization, and evaluation**.

---

## 🚀 Project Overview

Customer churn is a major challenge for telecommunications companies. Being able to identify customers who are likely to leave can help businesses take preventive actions and improve customer retention.

In this project, a **Multilayer Perceptron (MLP)** built with **Keras/TensorFlow** is trained on the Telco Customer Churn dataset to perform binary classification.

The model learns from customer demographic, service, contract, and billing information and predicts the customer's churn status.

### 🎯 Objective

> **Predict whether a customer will churn based on their available customer and service information.**

---

## 📁 Dataset

The project uses the **Telco Customer Churn** dataset.

The original dataset contains:

* **7,043 customers**
* **21 columns**
* Demographic information
* Account information
* Service subscriptions
* Billing information
* Churn status

After data cleaning, the dataset contains **7,032 usable samples**.

### Target Variable

The target variable is:

```text
Churn
```

with two possible classes:

* `No` → Customer stays
* `Yes` → Customer churns

The dataset contains:

| Class    | Samples |
| -------- | ------: |
| No Churn |   5,163 |
| Churn    |   1,869 |

This also shows that the target variable is somewhat imbalanced, making metrics beyond accuracy important when evaluating the model.

---

## 🧠 Model

The project uses a fully connected **Multilayer Perceptron (MLP)** implemented with Keras.

### Architecture

```text
Input Layer
     ↓
Dense (128) + ReLU
     ↓
Dropout (0.20)
     ↓
Dense (64) + ReLU
     ↓
Dropout (0.20)
     ↓
Dense (32) + ReLU
     ↓
Dropout (0.20)
     ↓
Dense (16) + ReLU
     ↓
Dropout (0.20)
     ↓
Dense (2) + Softmax
```

The network contains **14,866 trainable parameters**.

Dropout layers are used throughout the hidden layers to help reduce overfitting.

---

## 🔄 Machine Learning Pipeline

The project follows a complete end-to-end workflow:

```text
Raw Dataset
     ↓
Data Loading
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Feature / Target Separation
     ↓
Categorical Feature Encoding
     ↓
Train / Test Split
     ↓
Min-Max Scaling
     ↓
Target One-Hot Encoding
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
```

---

## 🔍 Exploratory Data Analysis

Several checks are performed before training the model:

* Dataset dimensions
* Column inspection
* Duplicate detection
* Missing-value detection
* Data types
* Numerical feature statistics
* Categorical feature statistics
* Target distribution

The dataset initially contains **7,043 rows and 21 columns**.

No duplicate rows were found, and the initial missing-value check returned zero missing values. The `TotalCharges` column, however, required conversion from object/string format to numeric values, after which invalid values were removed.

---

## 🧹 Data Preprocessing

### 1. Removing Identifier

The `customerID` column is removed because it is an identifier rather than a meaningful predictive feature.

### 2. Converting `TotalCharges`

`TotalCharges` is converted into a numeric feature:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

Rows containing invalid values are then removed.

### 3. Encoding Categorical Features

Categorical features are handled according to their cardinality:

* Binary categorical features → `LabelEncoder`
* Multi-class categorical features → One-Hot Encoding using `pd.get_dummies()`

This transforms the original feature space into **30 numerical features**.

### 4. Train/Test Split

The dataset is divided into:

* **78% training data**
* **22% test data**

Stratified splitting is used to preserve the target-class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.22,
    stratify=y,
    random_state=101
)
```

Result:

```text
Training samples: 5484
Testing samples: 1548
Features: 30
```

### 5. Feature Scaling

A `MinMaxScaler` is applied to normalize the numerical feature space:

```python
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

### 6. Target Encoding

The binary target is converted into a two-class one-hot representation:

```python
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
```

---

## ⚙️ Model Configuration

The neural network is compiled using:

| Component         | Configuration            |
| ----------------- | ------------------------ |
| Optimizer         | Adam                     |
| Loss Function     | Categorical Crossentropy |
| Metric            | Accuracy                 |
| Output Activation | Softmax                  |
| Hidden Activation | ReLU                     |
| Dropout Rate      | 0.20                     |

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## 🏋️ Training

The model is configured for up to **100 epochs** with:

* Batch size: `32`
* Validation split: `20%`
* Early stopping
* Patience: `5`
* Best weights restored

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

Early stopping helps prevent unnecessary training once validation performance stops improving.

---

## 📈 Training Visualization

Training history is visualized using Matplotlib.

The notebook tracks:

### Accuracy

```python
history.history["accuracy"]
history.history["val_accuracy"]
```

### Loss

```python
history.history["loss"]
history.history["val_loss"]
```

These plots make it possible to visually inspect the model's learning behavior and identify potential overfitting or underfitting.

---

## 📊 Evaluation

The trained model is evaluated on the previously unseen test set.

### Test Performance

```text
Accuracy: 80.88%
Loss:     0.4247
```

The classification report provides a more detailed view of performance:

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| 0 — No Churn         |      0.85 |   0.90 |     0.87 |
| 1 — Churn            |      0.67 |   0.55 |     0.60 |
| **Overall Accuracy** |           |        | **0.81** |

The model performs considerably better at identifying customers who do **not** churn than customers who do churn. This highlights why looking only at accuracy would not provide the full picture of model performance.

---

## 🛠️ Technologies & Libraries

The project is implemented in Python using:

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical operations
* **Matplotlib** — Visualization
* **Scikit-learn** — Preprocessing and evaluation
* **Keras / TensorFlow** — Neural network implementation

### Main Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
```

---

## 📂 Project Structure

```text
.
├── Multilayer_Perceptron_1.ipynb
├── Telco_customer_churn.csv
├── Telco_customer_churn.zip
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Multilayer_Perceptron_1.ipynb
```

Alternatively, the notebook can be executed directly in **Google Colab**.

---

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of several important machine learning and deep learning concepts:

* Exploratory Data Analysis (EDA)
* Data cleaning
* Feature engineering
* Categorical encoding
* One-hot encoding
* Train/test splitting
* Stratified sampling
* Feature scaling
* Multilayer Perceptrons
* Dense layers
* ReLU activation
* Softmax activation
* Dropout regularization
* Adam optimizer
* Categorical cross-entropy
* Early stopping
* Validation data
* Training history visualization
* Classification reports
* Precision, Recall, and F1-score

---

## 🎯 What I Learned

Through this project, I practiced building a neural network classification pipeline from raw tabular data to final model evaluation.

The main focus was understanding how an **MLP can be applied to structured/tabular data**, including the importance of preprocessing categorical features and scaling numerical inputs before training.

I also explored how different evaluation metrics can provide more meaningful insights than accuracy alone, especially when the target classes are not perfectly balanced.

---

## 🔮 Possible Improvements

Future versions of this project could explore:

* Hyperparameter tuning
* Different MLP architectures
* Learning-rate optimization
* Batch normalization
* Alternative regularization strategies
* Class weighting
* ROC-AUC evaluation
* Precision-Recall curves
* Confusion matrix visualization
* Cross-validation
* Comparison with traditional machine learning models
* More advanced feature engineering

---

## 📌 Conclusion

This project demonstrates an end-to-end approach to **customer churn prediction using a Multilayer Perceptron**.

Rather than focusing only on model construction, the notebook covers the complete workflow:

**Data → EDA → Cleaning → Preprocessing → Scaling → Neural Network → Training → Evaluation**

The final model achieves approximately **80.9% test accuracy**, while the detailed classification report provides a clearer understanding of its strengths and limitations across the two churn classes.

---

## ⭐ If you find this project useful

Feel free to explore the notebook, experiment with the architecture, and improve the model.

**Machine Learning is not just about training a model — it's about understanding the entire pipeline.**
