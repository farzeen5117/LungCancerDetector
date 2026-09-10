import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from callback import customCallback
from config import CLASSES, EPOCHS

def compile_model(model):
    """Compile the model with the chosen optimizer and loss function."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(model, train_ds, validation_ds):
    """Train the model and return the training history."""
    early_stopping = EarlyStopping(
        patience=3,
        monitor="val_accuracy",
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.5,
    )

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, customCallback()],
    )
    return history

def plot_history(history):
    """Plot accuracy and validation accuracy over time."""
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ["accuracy", "val_accuracy"]].plot()
    plt.show()

def evaluate_model(model, validation_ds):
    """Print a classification report for the validation data."""
    predictions = model.predict(validation_ds, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = np.concatenate([y for _, y in validation_ds], axis=0)
    true_labels = np.argmax(true_labels, axis=1)

    print(metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASSES,
    ))
