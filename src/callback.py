import tensorflow as tf

class customCallback(tf.keras.callbacks.Callback):
    ''' Stops training when validation accuracy reaches 90% '''
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.90:
            print("Stopping early")
            self.model.stop_training = True