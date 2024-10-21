# Pneumonia Detection

This project implements a deep learning-based approach for detecting pneumonia from chest X-ray images. It uses Convolutional Neural Networks (CNN) to classify chest X-rays into two categories: **Normal** and **Pneumonia**. The dataset is split into training and testing sets, and the model is trained to differentiate between normal and pneumonia-affected chest X-rays.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project aims to automate the detection of pneumonia using chest X-ray images. It utilizes TensorFlow and Keras to train a convolutional neural network model that can classify medical images. The model processes the images, learns the features associated with pneumonia, and predicts whether the patient has pneumonia or not.

## Features

* **Image Preprocessing:** The images are resized, normalized, and split into training and testing sets.
* **Convolutional Neural Network (CNN):** A deep learning model built using TensorFlow and Keras for image classification.
* **Training and Evaluation:** The model is trained on the dataset with a 20% validation split, and performance is evaluated on a test set.
* **Data Visualization:** The project includes functions to visualize the dataset and predictions.

## System Requirements

+ **Python:** 3.6 or higher
+ **TensorFlow:** 2.x
+ **Matplotlib:** 3.x
+ **Pillow:** 8.x or higher
+ **Google Colab:** Optional for running the notebook online


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Guruttamsv/Pneumonia-Detection.git
cd Pneumonia-Detection
```
2. Set up a virtual environment (optional but recommended):
```bash
# If using Conda
# Using conda
conda create -n pneumonia-detection python=3.8
conda activate pneumonia-detection
```
3. Install the required dependencies:
```bash
pip install tensorflow matplotlib numpy pillow
```

## Dataset

### Option 1: Download via URL
The dataset is downloaded from the following link and extracted directly into the working directory:
```bash
dataset_url = 'http://vision.roboslang.org/open_datasets/pneumonia_dataset.zip'
```

### Option 2: Manual Upload
If the download link is slow or unavailable, the dataset can be manually uploaded to Google Colab via the file upload interface.

The dataset contains two folders for training and testing:

+ **train:** Images of normal and pneumonia X-rays for training
+ **test:** Images of normal and pneumonia X-rays for testing

## Model Architecture

The model is based on a simple Convolutional Neural Network (CNN) with two convolutional layers followed by pooling and dropout layers to prevent overfitting. The architecture is as follows:
```python

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu'),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer for binary classification
])
```

## Training and Evaluation

### Training
The model is trained using the Adam optimizer with sparse categorical cross-entropy loss. The training process includes 5 epochs, with performance being evaluated based on accuracy and loss.

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the model
model.fit(train_ds, epochs=5)

```

### Evaluation
```python
# Evaluate model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.2f}")

```

## Results

+ **Training Accuracy:** Achieved after 5 epochs (can be improved with further training).
+ **Test Accuracy:** Model performance evaluated on the test set, typically around 90% or higher with the current architecture.

Model performance can be improved by fine-tuning hyperparameters, adding more layers, or experimenting with different optimizers.

## Limitations and Future Work

### Limitations:
+ **Data Quality:** The accuracy of the model depends on the quality and size of the dataset. Larger datasets might be needed for better generalization.
+ **Model Complexity:** The model is relatively simple and may not capture all intricate features in complex images.

### Future Work
* Improve model architecture to boost accuracy, especially for the test set.
* Incorporate more advanced techniques like **data augmentation** to prevent overfitting.
* Apply the model to real-world X-ray images to assess its practical utility.

## Acknowledgements

* **TensorFlow:** For providing the deep learning framework used in this project.
* **Pillow:** For the image processing support.
* **Google Colab:** For providing an accessible platform to run Jupyter notebooks with GPU support.

