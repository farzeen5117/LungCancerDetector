import os
from zipfile import ZipFile
from pathlib import Path

# Image processing tools for loading and preparing image data
from PIL import Image

# Data handling and analysis libraries used for dataset inspection and metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Deep learning libraries for building and training the image classification model
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing import image_dataset_from_directory as image_dataset_from_directory
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom training callback for early stopping 
# when validation accuracy is sufficiently high (90%)
from callback import customCallback

# Suppress warnings to keep training output cleaner and easier to read
import warnings
warnings.filterwarnings('ignore')

# Find the zip file anywhere in the repo
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "data.zip"
extract_path = project_root

# Extracting dataset
with ZipFile(data_path, "r") as zip_file:
    zip_file.extractall(extract_path)
    print("Dataset extracted successfully.")

# Visualize a few sample images from each class to understand the dataset better
path = 'data/images'
classes = ['lung_n', 'lung_aca', 'lung_scc']

for category in classes:
    image_dir = f'{path}/{category}'
    images = os.listdir(image_dir)

    # Create a small gallery with three random images from the current class
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {category} category', fontsize=20)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{path}/{category}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()

# Define the image size and batch size for the model input pipeline.
# These values control how the images are resized and grouped during training.
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10

# Split the dataset into training and validation subsets using the directory structure
# loads images directly from the class folders and labels them by folder name
train_ds = image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

validation_ds = image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Improve training efficiency by caching the dataset and prefetching batches.
# Caching keeps the images in memory after the first read, while prefetching overlaps
# data loading with model execution to reduce idle time.

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Build the convolutional neural network used for lung cancer classification.
# Each convolutional layer learns visual features from the input images, while the
# pooling layers reduce spatial dimensions and make the model more efficient.

model = keras.models.Sequential([  # Sequential stacks layers one after another.
    
    # First convolutional layer: detects low-level patterns such as edges and textures.
    layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Max pooling reduces the feature map size and helps keep the most important signals
    layers.MaxPooling2D(2, 2),

    # Second convolutional layer: learns more complex features from the reduced maps
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    # Third convolutional layer: captures higher-level patterns before classification
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    # Flatten converts the 2D feature maps into a single vector for dense layers
    layers.Flatten(),

    # Fully connected layer that combines the learned features into a decision space
    layers.Dense(256, activation='relu'),

    layers.BatchNormalization(), # stabilizes activations and accelerates training
    layers.Dense(128, activation='relu'),

    # Dropout randomly disables neurons during training to reduce overfitting
    layers.Dropout(0.3),
    layers.BatchNormalization(),

    # Final layer has one output unit per class and uses softmax for class probabilities
    layers.Dense(3, activation='softmax')
])

model.summary() # print the model structure

# Early stopping prevents overtraining by halting training when validation accuracy
# stops improving, and it restores the best weights seen during training.
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)

# lowers the learning rate when the validation loss plateaus,
# which helps the model converge more smoothly in later training stages.
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# Compile the model with the Adam optimizer and multi-class categorical cross-entropy.
# This configuration is appropriate for a classification problem with three labels.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model is trained using generator-based data
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=[es, lr, customCallback()]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

Y_pred = model.predict(validation_ds)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_true = np.concatenate([y for _, y in validation_ds], axis=0)
Y_true = np.argmax(Y_true, axis=1)

print(metrics.classification_report(
    Y_true,
    Y_pred_labels,
    target_names=classes
))