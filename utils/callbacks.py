import keras


class Modify_Input(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        print batch.shape
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return