import sys
from keras.models import Model
from keras.layers import Input
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Activation


def model(weights_path=None):
    _input = Input((3, 198, 198))
    _m = ResNet50(weights=None, include_top=False, input_tensor=_input, input_shape=(3, 198, 198))
    _m.load_weights('models/resnet50/structures/resnet50_weights_th_dim_ordering_th_kernels_notop.h5')

    # 175 layers
    for i, layer in enumerate(_m.layers):
        if i > 155:
            break

        layer.trainable = False

    x = _m.output
    x = Flatten(name='my_flatten')(x)

    # x = Dense(512, name='my_dense_1')(x)
    # x = BatchNormalization(axis=1, name='my_bn_1')(x)
    # x = Activation('relu', name='my_act_1')(x)
    x = Dropout(0.5, name='my_dp_1')(x)

    x = Dense(10, name='my_output_dense')(x)
    x = Activation('softmax', name='my_output')(x)

    _model = Model(inputs=_m.input, outputs=x)

    if weights_path:
        _model.load_weights(weights_path)

    return [_model, 'models/resnet50/structures/']

