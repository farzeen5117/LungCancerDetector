import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory

from utils.config import BATCH_SIZE, CLASSES, IMG_SIZE, IMAGES_PATH, PROJECT_ROOT, DATA_ZIP_PATH

def extract_dataset():
    """Extract the archived dataset into the project root."""
    with ZipFile(DATA_ZIP_PATH, "r") as zip_file:
        zip_file.extractall(PROJECT_ROOT)
        print("Dataset extracted successfully.")

def visualize_dataset():
    """Display a small sample from each class to validate the dataset."""
    for category in CLASSES:
        image_dir = IMAGES_PATH / category
        images = os.listdir(image_dir)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Images for {category} category", fontsize=20)

        for i in range(3):
            idx = np.random.randint(0, len(images))
            img = np.array(Image.open(image_dir / images[idx]))
            ax[i].imshow(img)
            ax[i].axis("off")
        plt.show()

def load_datasets():
    """Create training and validation datasets from the image folders."""
    train_ds = image_dataset_from_directory(
        str(IMAGES_PATH),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    validation_ds = image_dataset_from_directory(
        str(IMAGES_PATH),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, validation_ds