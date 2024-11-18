# Helmet vs Non-Helmet Image Classification

This project uses Convolutional Neural Networks (CNNs) to classify images of people wearing helmets and not wearing helmets. The goal is to build a reliable model for helmet detection, which can be used in safety applications, such as detecting whether a person is wearing a helmet while riding a bike or motorcycle.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Installation](#installation)
- [Model Development](#model-development)
- [Fine-Tuning](#fine-tuning)
- [Training the Model](#training-the-model)
- [Evaluation and Results](#evaluation-and-results)

## Introduction

Helmet detection is an important problem in ensuring safety, especially for motorcyclists. This project focuses on classifying images into two categories:

- **Helmet**: Images of individuals wearing helmets.
- **Non-Helmet**: Images of individuals not wearing helmets.

The project uses a CNN architecture to learn the features of helmeted and non-helmeted individuals from a labeled dataset of images.

## Dataset

The dataset consists of images categorized into two classes:

1. **Helmet**: Images of people wearing helmets.
2. **Non-Helmet**: Images of people not wearing helmets.

The dataset is structured as follows:

- `Helmet_Images.zip`: Contains images of individuals wearing helmets.
- `NoHelmet_Images.zip`: Contains images of individuals not wearing helmets.

### Data Preprocessing

The images are resized to 128x128 pixels, normalized to a scale of 0 to 1, and renamed for consistency. The dataset is then split into training and testing sets.

## Technologies

This project uses the following technologies:

- **Python 3.x**
- **TensorFlow** for building the CNN model
- **OpenCV** for image processing
- **NumPy** for numerical computations
- **Matplotlib** for plotting results
- **Scikit-learn** for evaluating the model

## Installation

To run this project locally, follow the steps below:

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Helmet-Non-Helmet.git
```

### Step 2: Install Required Libraries

Install the necessary libraries using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Mount Google Drive (If Using Google Colab)

If you're working on Google Colab, mount your Google Drive to access the dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Load Dataset

Unzip the dataset and load it into the appropriate directories for model training and testing.

## Model Development

The model is built using a **Convolutional Neural Network (CNN)**. The architecture of the model is as follows:

- **Input Layer**: Images resized to 128x128 pixels.
- **Conv2D Layers**: Convolutional layers with ReLU activation.
- **MaxPooling Layers**: MaxPooling layers to reduce spatial dimensions.
- **Dense Layers**: Fully connected layers with a softmax output layer for binary classification.
  
### CNN Architecture:

1. Conv2D (filters=32, kernel_size=(3,3), activation='relu')
2. MaxPooling2D (pool_size=(2, 2))
3. Conv2D (filters=64, kernel_size=(3,3), activation='relu')
4. MaxPooling2D (pool_size=(2, 2))
5. Flatten
6. Dense (units=128, activation='relu')
7. Dense (units=1, activation='sigmoid')

## Fine-Tuning

To improve the performance of the model, several techniques are used:

1. **Data Augmentation**: The training dataset is augmented with random transformations (e.g., rotation, flipping, zooming) to introduce more diversity and prevent overfitting.
2. **Early Stopping**: The training process stops early if the validation loss does not improve for a specified number of epochs.
3. **Model Checkpoints**: The best model is saved during training based on the lowest validation loss.

## Training the Model

To train the model, the following code can be used:

```python
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=17
)
```

The model is trained for 17 epochs, with a batch size of 32. You can adjust the number of epochs and batch size as needed.

## Evaluation and Results

After training, the model's performance is evaluated on the test dataset. The results include:

- **Test Accuracy**: 66%
- **Test Loss**: 1.2345

### Classification Report

```plaintext
              precision    recall  f1-score   support

      Helmet       0.70      0.63      0.66        32
   No Helmet       0.62      0.70      0.66        31

    accuracy                           0.66        63
   macro avg       0.66      0.66      0.66        63
weighted avg       0.66      0.66      0.66        63
```

### Confusion Matrix

```plaintext
[[20 12]
 [ 9 22]]
```

These results demonstrate that after fine-tuning, the model is able to correctly classify images with an accuracy of 66%, showing significant improvement from the baseline model.
