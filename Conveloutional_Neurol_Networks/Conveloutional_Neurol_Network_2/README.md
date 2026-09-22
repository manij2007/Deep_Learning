# 🚗🏍️ Car vs Bike Image Classification Using CNN

A deep learning image classification project that uses a **Convolutional Neural Network (CNN)** to distinguish between **cars and bikes** from RGB images.

The model is built from scratch using **TensorFlow/Keras** and demonstrates a complete computer vision workflow, including image loading, normalization, one-hot encoding, convolutional feature extraction, batch normalization, regularization, model evaluation, and prediction on individual images.

---

## 📌 Project Overview

The goal of this project is to develop a CNN capable of classifying an input image into one of two categories:

* 🏍️ **Bike**
* 🚗 **Car**

Unlike basic grayscale image classification tasks, this project works with real-world **RGB images** resized to `224 × 224 × 3`.

The model uses multiple convolutional blocks with increasing numbers of filters to progressively learn more complex visual features.

---

## 🎯 Objectives

* Build a CNN for real-world RGB image classification.
* Load images directly from a directory structure using Keras.
* Normalize RGB pixel values.
* Apply one-hot encoding to binary class labels.
* Learn hierarchical image features using convolutional layers.
* Use Batch Normalization to stabilize training.
* Apply Max Pooling for spatial dimensionality reduction.
* Reduce overfitting using Dropout.
* Evaluate the trained model on unseen test images.
* Perform predictions on individual external images.

---

## 📊 Dataset

The project uses a **Car vs Bike image dataset** organized into separate training and testing directories.

### Dataset Structure

```text
Car_Bike_dataset/
│
├── train/
│   ├── Bike/
│   └── Car/
│
└── test/
    ├── Bike/
    └── Car/
```

### Dataset Information

| Property          | Value                       |
| ----------------- | --------------------------- |
| Task              | Binary Image Classification |
| Classes           | Bike, Car                   |
| Number of Classes | 2                           |
| Training Images   | 1,400                       |
| Testing Images    | 600                         |
| Image Size        | 224 × 224                   |
| Color Mode        | RGB                         |
| Channels          | 3                           |
| Batch Size        | 32                          |

The class order detected by Keras is:

```python
['Bike', 'Car']
```

Therefore:

```text
0 → Bike
1 → Car
```

---

## 📥 Loading the Images

Images are loaded using Keras `image_dataset_from_directory`.

```python
train = image_dataset_from_directory(
    directory="/content/Car_Bike_dataset/train",
    image_size=(224, 224),
    color_mode="rgb",
    batch_size=32
)

test = image_dataset_from_directory(
    directory="/content/Car_Bike_dataset/test",
    image_size=(224, 224),
    color_mode="rgb",
    batch_size=32
)
```

Keras automatically detects the classes based on the directory names.

The dataset contains:

```text
Found 1400 files belonging to 2 classes.
Found 600 files belonging to 2 classes.

Classes: ['Bike', 'Car']
```

---

## 🛠️ Data Preprocessing

### 1. Image Resizing

All images are automatically resized to:

```text
224 × 224
```

Since the images are loaded in RGB mode, the final input shape for the CNN is:

```text
224 × 224 × 3
```

---

### 2. Pixel Normalization

Original image pixel values are in the range:

```text
0 – 255
```

A Keras `Rescaling` layer is used to normalize them:

```python
normalization_layer = Rescaling(1.0 / 255.0)

train = train.map(
    lambda x, y: (normalization_layer(x), y)
)

test = test.map(
    lambda x, y: (normalization_layer(x), y)
)
```

After normalization, pixel values are approximately in the range:

```text
0.0 – 1.0
```

Normalization helps make the optimization process more stable.

---

### 3. One-Hot Encoding

The original labels are integer encoded.

Since the output layer contains two neurons with a Softmax activation function, the labels are converted to one-hot encoded vectors.

```python
def one_hot_encoder(x, y):
    y = tf.one_hot(y, depth=2)
    return x, y

train = train.map(one_hot_encoder)
test = test.map(one_hot_encoder)
```

Conceptually:

```text
Bike → [1, 0]
Car  → [0, 1]
```

---

# 🧠 CNN Architecture

The network is built from scratch using the Keras Sequential API.

It contains three major convolutional blocks followed by fully connected layers.

## Convolutional Block 1

```text
Input: 224 × 224 × 3

Conv2D
32 Filters
3 × 3 Kernel
ReLU
He Uniform
Same Padding
      ↓
Batch Normalization
      ↓
Conv2D
32 Filters
3 × 3 Kernel
ReLU
      ↓
Batch Normalization
      ↓
MaxPooling2D (2 × 2)
      ↓
Dropout (0.2)
```

---

## Convolutional Block 2

```text
Conv2D (64 Filters, 3 × 3, ReLU)
      ↓
Batch Normalization
      ↓
Conv2D (64 Filters, 3 × 3, ReLU)
      ↓
Batch Normalization
      ↓
MaxPooling2D (2 × 2)
      ↓
Dropout (0.2)
```

---

## Convolutional Block 3

```text
Conv2D (128 Filters, 3 × 3, ReLU)
      ↓
Batch Normalization
      ↓
Conv2D (128 Filters, 3 × 3, ReLU)
      ↓
Batch Normalization
      ↓
MaxPooling2D (2 × 2)
      ↓
Dropout (0.2)
```

The number of filters increases from:

```text
32 → 64 → 128
```

This allows deeper layers to learn increasingly complex visual features.

---

## Fully Connected Classifier

After feature extraction, the feature maps are flattened and passed through several Dense layers.

```text
Flatten
   ↓
Dense (512, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense (256, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense (128, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense (64, ReLU)
   ↓
Dropout (0.3)
   ↓
Dense (2, Softmax)
```

The final two neurons represent the two possible classes:

```text
Bike
Car
```

---

## 🏗️ Complete Architecture

```text
Input (224 × 224 × 3)
        ↓
Conv2D (32)
        ↓
BatchNormalization
        ↓
Conv2D (32)
        ↓
BatchNormalization
        ↓
MaxPooling2D
        ↓
Dropout (0.2)
        ↓
Conv2D (64)
        ↓
BatchNormalization
        ↓
Conv2D (64)
        ↓
BatchNormalization
        ↓
MaxPooling2D
        ↓
Dropout (0.2)
        ↓
Conv2D (128)
        ↓
BatchNormalization
        ↓
Conv2D (128)
        ↓
BatchNormalization
        ↓
MaxPooling2D
        ↓
Dropout (0.2)
        ↓
Flatten
        ↓
Dense (512)
        ↓
Dropout (0.3)
        ↓
Dense (256)
        ↓
Dropout (0.3)
        ↓
Dense (128)
        ↓
Dropout (0.3)
        ↓
Dense (64)
        ↓
Dropout (0.3)
        ↓
Dense (2, Softmax)
```

---

## 🔬 Important Architecture Choices

### Conv2D

Convolutional layers extract spatial features from the input images.

Earlier layers typically learn relatively simple patterns, while deeper layers can represent more complex visual structures.

### Batch Normalization

Batch Normalization is applied after every convolutional layer in this architecture.

It helps stabilize the activations during training and can improve optimization behavior.

### Max Pooling

`MaxPooling2D` reduces the spatial dimensions of the feature maps.

```python
MaxPooling2D((2, 2))
```

This reduces computational complexity while retaining important learned features.

### Dropout

Dropout is used throughout the network as a regularization technique.

Convolutional blocks use:

```text
Dropout = 0.2
```

Fully connected layers use:

```text
Dropout = 0.3
```

### He Uniform Initialization

The convolutional layers use:

```python
kernel_initializer="he_uniform"
```

This initialization strategy is commonly paired with ReLU-based neural networks.

---

# ⚙️ Model Compilation

The model is compiled using:

```python
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=RMSprop(learning_rate=0.001),
    metrics=["accuracy"]
)
```

### Configuration

| Parameter         | Value                    |
| ----------------- | ------------------------ |
| Optimizer         | RMSprop                  |
| Learning Rate     | 0.001                    |
| Loss Function     | Categorical Crossentropy |
| Metric            | Accuracy                 |
| Output Activation | Softmax                  |

Because the targets are one-hot encoded and the network has two output neurons, `CategoricalCrossentropy` is used as the loss function.

---

# 🚀 Model Training

The model is trained for **20 epochs**.

```python
history = model.fit(
    train,
    validation_data=test,
    epochs=20
)
```

### Training Configuration

| Parameter         | Value   |
| ----------------- | ------- |
| Epochs            | 20      |
| Batch Size        | 32      |
| Training Images   | 1,400   |
| Validation Images | 600     |
| Optimizer         | RMSprop |
| Learning Rate     | 0.001   |

> **Note:** In this notebook, the test dataset is also used as `validation_data` during training. For a stricter evaluation workflow, the dataset should ideally be divided into separate training, validation, and test sets.

---

# 📈 Training Performance

The notebook tracks both training and validation performance during the 20 epochs.

Training accuracy increased substantially during training, reaching approximately:

```text
98.50%
```

at the final epoch.

The validation accuracy at the final epoch was:

```text
88.33%
```

---

## 📊 Accuracy Visualization

Training and validation accuracy are visualized using Matplotlib:

```python
plt.figure(figsize=(8, 6))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Test"])

plt.show()
```

This plot makes it easier to compare the model's performance on training and unseen images.

---

## 📉 Loss Visualization

The notebook also visualizes training and validation loss:

```python
plt.figure(figsize=(8, 6))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Test"])

plt.show()
```

These curves can be useful for identifying generalization problems and overfitting.

---

# 📊 Model Evaluation

The model is evaluated using:

```python
result = model.evaluate(test, verbose=0)

print(
    f"Accuracy : {result[1]} , Loss : {result[0]}"
)
```

The notebook reports:

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy | **88.33%** |
| Test Loss     | **1.6290** |
| Test Images   |        600 |

Exact notebook output:

```text
Accuracy : 0.8833333253860474
Loss     : 1.6290075778961182
```

Therefore, the CNN correctly classified approximately **88.33% of the 600 test images** in this evaluation.

---

## ⚠️ Generalization Observation

At the final epoch, the model reached approximately:

```text
Training Accuracy   ≈ 98.50%
Validation Accuracy ≈ 88.33%
```

while the final validation loss was considerably higher than the training loss.

This gap suggests that the current model may be **overfitting the training dataset**.

The result is useful because it demonstrates an important deep learning concept: achieving very high training accuracy does not automatically mean that a model generalizes equally well to unseen images.

---

# 🔍 Single Image Prediction

The notebook also implements inference on an individual image.

An image is loaded using:

```python
img = load_img(
    img_path,
    target_size=(224, 224),
    color_mode="rgb"
)
```

It is then converted into a NumPy array:

```python
img = img_to_array(img)
```

The image is normalized:

```python
img = img / 255.0
```

Since the model expects a batch of images, an additional batch dimension is added:

```python
img = np.expand_dims(img, axis=0)
```

The resulting shape is conceptually:

```text
(1, 224, 224, 3)
```

---

## 🎯 Generating the Prediction

Prediction probabilities are generated using:

```python
prediction = model.predict(img)
```

The class with the highest probability is selected:

```python
predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
```

The notebook tested the model using a bike image.

The final prediction was:

```text
Prediction: Bike
```

The model therefore correctly classified the tested image as a **Bike**.

---

# 🧰 Technologies & Libraries

This project uses:

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Matplotlib**
* **Google Colab**

Important Keras components include:

```python
Sequential
Input
Conv2D
MaxPooling2D
BatchNormalization
Dropout
Flatten
Dense
Rescaling
image_dataset_from_directory
load_img
img_to_array
RMSprop
CategoricalCrossentropy
```

---

# 📁 Project Structure

```text
Convolutional_Neural_Network/
│
├── Conveloutional_Neurol_Network_2.ipynb
│
└── README.md
```

The dataset used by the notebook follows this structure:

```text
Car_Bike_dataset/
│
├── train/
│   ├── Bike/
│   └── Car/
│
└── test/
    ├── Bike/
    └── Car/
```

---

# ⚡ Getting Started

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

## 2. Navigate to the Project

```bash
cd Convolutional_Neural_Network
```

## 3. Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

## 4. Prepare the Dataset

Organize the images into the following structure:

```text
Car_Bike_dataset/
├── train/
│   ├── Bike/
│   └── Car/
└── test/
    ├── Bike/
    └── Car/
```

Update the dataset paths in the notebook if necessary.

## 5. Run the Notebook

Open:

```text
Conveloutional_Neurol_Network_2.ipynb
```

Then execute the cells sequentially.

---

# 🧠 Key Concepts Covered

This project demonstrates several important computer vision and deep learning concepts:

* Convolutional Neural Networks
* Binary image classification
* RGB image processing
* Image resizing
* Image normalization
* One-hot encoding
* Convolutional feature extraction
* Multiple convolutional blocks
* ReLU activation
* He Uniform initialization
* Batch Normalization
* Max Pooling
* Dropout regularization
* Fully connected neural networks
* Softmax classification
* Categorical Crossentropy
* RMSprop optimization
* Training and validation monitoring
* Overfitting analysis
* Image inference and prediction

---

# 📚 Learning Outcomes

After completing this project, you can gain practical experience with:

1. Loading real-world image datasets from directories.
2. Preparing RGB images for CNN models.
3. Understanding image tensor dimensions.
4. Normalizing image pixel values.
5. Encoding class labels for Softmax classification.
6. Building deeper CNN architectures.
7. Increasing convolutional filters across network depth.
8. Applying Batch Normalization.
9. Using Dropout for regularization.
10. Evaluating generalization performance.
11. Recognizing signs of overfitting.
12. Preparing individual images for inference.
13. Converting neural network outputs into human-readable class predictions.

---

# 🔧 Possible Improvements

The current model can be improved in several ways:

* Create a separate validation dataset instead of using the test set during training.
* Add **data augmentation** such as random flipping, rotation, zoom, and translation.
* Add **EarlyStopping** to stop training when validation performance stops improving.
* Use **ReduceLROnPlateau** for adaptive learning-rate reduction.
* Experiment with a smaller fully connected classifier.
* Tune Dropout rates.
* Experiment with Adam and other optimizers.
* Save the best model using `ModelCheckpoint`.
* Add a confusion matrix.
* Add precision, recall, and F1-score evaluation.
* Visualize incorrectly classified images.
* Experiment with transfer learning models such as MobileNet, EfficientNet, or ResNet.

---

# 📝 Conclusion

This project implements a complete **Car vs Bike image classification system** using a custom Convolutional Neural Network built with TensorFlow/Keras.

The CNN processes `224 × 224` RGB images through multiple convolutional blocks containing **Conv2D, Batch Normalization, Max Pooling, and Dropout layers**, followed by a fully connected classifier.

After 20 epochs, the model achieved:

```text
Test Accuracy: 88.33%
Test Loss:     1.6290
```

The notebook also demonstrates how to preprocess and classify an individual image, successfully predicting the tested sample as:

```text
Bike
```

Beyond achieving a classification result, the project demonstrates an important practical challenge in deep learning: the difference between strong training performance and generalization to unseen data.

It serves as a practical step from simple image datasets toward **real-world Computer Vision and CNN-based image classification**.

---

## 👨‍💻 Author

**Mani**

Machine Learning & Deep Learning Learning Journey

⭐ If you find this project useful, consider starring the repository.

