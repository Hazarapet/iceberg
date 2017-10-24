from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    _model = Sequential()

    _model.add(BatchNormalization(axis=1, input_shape=(2, 75, 75)))

    for i in range(4):
        _model.add(Conv2D(8 * 2 ** i, kernel_size=(3, 3)))
        _model.add(MaxPooling2D((2, 2)))

    _model.add(GlobalMaxPooling2D())
    _model.add(Dropout(0.5))
    _model.add(Dense(4))

    _model.add(Dense(1, activation='sigmoid'))

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/model/structures/']

