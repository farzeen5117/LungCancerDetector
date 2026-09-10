# LungCancerDetector

A deep learning project for classifying lung cancer image types from histopathology-style medical images. The project loads image folders from the dataset, builds a convolutional neural network (CNN), and trains it to distinguish between the supported classes.

## Project goal

This project aims to detect and classify lung cancer categories from image data, using a simple Keras/TensorFlow pipeline. The current implementation is designed for a multi-class classification problem with the following classes:

- lung_n
- lung_aca
- lung_scc

## Repository structure

- [src/main.py](src/main.py) — main training and evaluation script
- [src/callback.py](src/callback.py) — custom Keras callback for early stopping
- [data/](data/) — dataset folder, including the extracted image data and zip archive
- [requirements.txt](requirements.txt) — Python dependencies

## Dataset

The project expects a zipped dataset at:

- [data/data.zip](data/data.zip)

When the script runs, it extracts the archive into the project root and then loads images from the extracted directory structure under [data/images](data/). The training pipeline uses the folder names as labels, so each class should be in its own directory.

Example folder structure:

```text
data/
  data.zip
  images/
    lung_n/
    lung_aca/
    lung_scc/
```

## Environment setup

1. Create and activate a virtual environment if desired.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the project

From the project root:

```bash
python src/main.py
```

This script will:

- extract the dataset archive
- show sample images from each class
- build the CNN model
- train the model on the dataset
- evaluate accuracy on the validation set
- print a classification report

## Model overview

The training script uses a TensorFlow/Keras CNN with:

- convolutional layers for feature extraction
- max pooling layers to reduce dimensionality
- a flatten layer to convert feature maps to a vector
- dense layers for classification
- batch normalization and dropout to improve generalization
- early stopping and learning rate reduction callbacks