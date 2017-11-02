from keras import layers
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalMaxPooling2D, Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def model(weights_path=None):
    act = "selu"
    dp = 0.4
    gain = 16
    _input_1 = Input(shape=(3, 75, 75), name="image_input")
    _input_2 = Input(shape=(1,), name="angle_input")

    def small_conv(filters, _input, count, block):
        for i in range(count):
            _input = BatchNormalization(axis=1, name="img_{}_bn_{}".format(block, i))(_input)
            _input = Conv2D(filters + gain * i, (3, 3), padding="same", name="img_{}_conv_{}".format(block, i))(_input)
            _input = Activation(act, name="img_{}_act_{}".format(block, i))(_input)

            _input = Dropout(dp, name="img_{}_dp_{}".format(block, i))(_input)

        return _input

    def small_res(filters, _input, block):
        _input = BatchNormalization(axis=1, name="res_{}_bn_{}".format(block, 1))(_input)
        _input = Conv2D(filters, (1, 1), padding="same", name="res_{}_conv_{}".format(block, 1))(_input)
        _input = Activation(act, name="res_{}_act_{}".format(block, 1))(_input)

        _input = BatchNormalization(axis=1, name="res_{}_bn_{}".format(block, 2))(_input)
        _input = Conv2D(filters * 2, (3, 3), padding="same", name="res_{}_conv_{}".format(block, 2))(_input)
        _input = Activation(act, name="res_{}_act_{}".format(block, 2))(_input)

        _input = BatchNormalization(axis=1, name="res_{}_bn_{}".format(block, 3))(_input)
        _input = Conv2D(filters, (1, 1), padding="same", name="res_{}_conv_{}".format(block, 3))(_input)
        _input = Activation(act, name="res_{}_act_{}".format(block, 3))(_input)

        _input = MaxPooling2D((2, 2), name="res_{}_pool".format(block))(_input)
        _input = Dropout(dp, name="res_{}_dp".format(block))(_input)

        return _input

    def small_bridge(filters, _input, block):
        _input = BatchNormalization(axis=1, name="bridge_{}_bn_{}".format(block, 1))(_input)
        _input = Conv2D(filters, (1, 1), padding="same", name="bridge_{}_conv_{}".format(block, 1))(_input)
        _input = Activation(act, name="bridge_{}_act_{}".format(block, 1))(_input)

        _input = MaxPooling2D((2, 2), name="bridge_{}_pool".format(block))(_input)
        _input = Dropout(dp, name="bridge_{}_dp".format(block))(_input)

        return _input

    _block_1 = small_conv(32, _input_1, count=3, block=1)
    _res_1_1 = small_res(32, _block_1, block=1.1)
    _bridge_1 = small_bridge(32, _input_1, block=1)
    _concat_1 = layers.concatenate([_res_1_1, _bridge_1], axis=1, name='concat_1')
    _res_1_2 = small_res(64, _concat_1, block=1.2)

    _block_2 = small_conv(64, _res_1_2, count=3, block=2)
    _res_2_1 = small_res(64, _block_2, block=2.1)
    _bridge_2 = small_bridge(64, _concat_1, block=2)
    _concat_2 = layers.concatenate([_res_2_1, _bridge_2], axis=1, name='concat_2')
    _res_2_2 = small_res(128, _concat_2, block=2.2)

    _glb_pool_1 = GlobalMaxPooling2D()(_res_2_2)

    _dense_1 = Dense(64, kernel_regularizer=l2(1e-4), name="dense_1")(_glb_pool_1)
    _dense_1 = BatchNormalization(axis=1, name="dense_1_bn_1")(_dense_1)
    _dense_1 = Activation(act, name="dense_1_act_1")(_dense_1)
    _dense_1 = Dropout(dp, name="dense_1_dp_1")(_dense_1)

    _dense_2 = Dense(32, kernel_regularizer=l2(1e-4), name="dense_2")(_dense_1)
    _dense_2 = BatchNormalization(axis=1, name="dense_2_bn_1")(_dense_2)
    _dense_2 = Activation(act, name="dense_2_act_1")(_dense_2)
    _dense_2 = Dropout(dp, name="dense_2_dp_1")(_dense_2)

    _value_1 = BatchNormalization(axis=1, name="angle_1_bn_1")(_input_2)
    _concat_2 = layers.concatenate([_dense_2, _value_1], name='model_concat_2')

    _output = Dense(1, activation='sigmoid', name="output")(_concat_2)

    _model = Model(inputs=[_input_1, _input_2], outputs=[_output])

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/model/structures/']

