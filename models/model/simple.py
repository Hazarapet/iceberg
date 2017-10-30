from keras import layers
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalMaxPooling2D, Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    act = "elu"
    dp = 0.5
    _input_1 = Input(shape=(3, 75, 75), name="image_input")
    _input_2 = Input(shape=(1,), name="angle_input")

    _img_1 = Conv2D(16, (3, 3), name="img_1_conv_1")(_input_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_1")(_img_1)
    _img_1 = Activation(act, name="img_1_act_1")(_img_1)

    _img_1 = Conv2D(16, (3, 3), name="img_1_conv_2")(_img_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_2")(_img_1)
    _img_1 = Activation(act, name="img_1_act_2")(_img_1)

    _img_1 = MaxPooling2D((2, 2), name="img_1_pool_1")(_img_1)
    _img_1 = Dropout(dp, name="img_1_dp_1")(_img_1)

    _img_1 = Conv2D(32, (3, 3), name="img_1_conv_3")(_img_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_3")(_img_1)
    _img_1 = Activation(act, name="img_1_act_3")(_img_1)

    _img_1 = Conv2D(32, (3, 3), name="img_1_conv_4")(_img_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_4")(_img_1)
    _img_1 = Activation(act, name="img_1_act_4")(_img_1)

    _img_1 = MaxPooling2D((2, 2), name="img_1_pool_2")(_img_1)
    _img_1 = Dropout(dp, name="img_1_dp_2")(_img_1)

    _img_1 = Conv2D(64, (3, 3), name="img_1_conv_5")(_img_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_5")(_img_1)
    _img_1 = Activation(act, name="img_1_act_5")(_img_1)

    _img_1 = Conv2D(64, (3, 3), name="img_1_conv_6")(_img_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_6")(_img_1)
    _img_1 = Activation(act, name="img_1_act_6")(_img_1)

    _img_1 = Conv2D(128, (3, 3), name="img_1_conv_7")(_img_1)
    _img_1 = BatchNormalization(axis=1, name="img_1_bn_7")(_img_1)
    _img_1 = Activation(act, name="img_1_act_7")(_img_1)

    _img_1 = MaxPooling2D((2, 2), name="img_1_pool_7")(_img_1)
    _img_1 = Dropout(dp, name="img_1_dp_3")(_img_1)

    _img_1 = GlobalMaxPooling2D()(_img_1)

    _img_2 = Conv2D(128, (3, 3), name="img_2_conv_1")(_input_1)
    _img_2 = BatchNormalization(axis=1, name="img_2_bn_1")(_img_2)
    _img_2 = Activation(act, name="img_2_act_1")(_img_2)

    _img_2 = MaxPooling2D((2, 2), name="img_2_pool_1")(_img_2)
    _img_2 = Dropout(dp, name="img_2_dp_1")(_img_2)

    _img_2 = GlobalMaxPooling2D()(_img_2)

    _concat_1 = layers.concatenate([_img_1, _img_2], name='model_concat_1')

    _dense_1 = Dense(128, kernel_regularizer=l2(1e-4), name="dense_1")(_concat_1)
    _dense_1 = BatchNormalization(axis=1, name="dense_1_bn_1")(_dense_1)
    _dense_1 = Activation(act, name="dense_1_act_1")(_dense_1)
    _dense_1 = Dropout(dp, name="dense_1_dp_1")(_dense_1)

    _dense_2 = Dense(32, kernel_regularizer=l2(1e-4), name="dense_2")(_dense_1)
    _dense_2 = BatchNormalization(axis=1, name="dense_2_bn_1")(_dense_2)
    _dense_2 = Activation(act, name="dense_2_act_1")(_dense_2)
    _dense_2 = Dropout(dp, name="dense_2_dp_1")(_dense_2)

    _value_1 = BatchNormalization(axis=1, name="value_1_bn_1")(_dense_2)
    _concat_2 = layers.concatenate([_dense_1, _value_1], name='model_concat_2')

    _output = Dense(1, activation='sigmoid', name="output")(_concat_2)

    _model = Model(inputs=[_input_1, _input_2], outputs=[_output])

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/model/structures/']

