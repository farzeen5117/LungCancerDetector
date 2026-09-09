import os
import random
from zipfile import ZipFile

# image processing
import cv2
from PIL import Image

# data handling and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# building and training MLMs
import tensorflow as tf
import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore')

# importing the dataset in images.zip
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "data.zip"
extract_path = project_root

with ZipFile(data_path, "r") as zip_file:
    zip_file.extractall(extract_path)
    print("Dataset extracted successfully.")

# visualize random images from each category to understand the dataset
global path
path = 'data/images'
classes = ['lung_n', 'lung_aca', 'lung_scc']

for category in classes:
    image_dir = f'{path}/{category}'
    images = os.listdir(image_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {category} category', fontsize=20)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{path}/{category}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()

IMG_SIZE = 128      
BATCH_SIZE = 16    
EPOCHS = 10

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)