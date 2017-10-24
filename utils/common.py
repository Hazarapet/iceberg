import cv2
import numpy as np


def iterate_minibatches(inputs, batchsize=10):

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt]

    if len(inputs) % batchsize != 0:
        yield np.array(inputs)[- (len(inputs) % batchsize):]


def aug(array, input):
    # input's shape (cn, w, h)
    rn1 = np.random.randint(0, 12)
    rn2 = np.random.randint(input.shape[1] - 12, input.shape[1])  # this is much better

    # rotate 90
    rt90 = np.rot90(input, 1, axes=(1, 2))
    array.append(rt90)

    # flip h
    flip_h = np.flip(input, 2)
    array.append(flip_h)

    # flip v
    # flip_v = np.flip(input, 1)
    # array.append(flip_v)

    # random crop with 32px shift
    # TODO Kind of overfiting technique
    crop = input.transpose((1, 2, 0))
    crop = cv2.resize(crop[rn1:rn2, rn1:rn2], (crop.shape[0], crop.shape[1]))
    crop = crop.transpose((2, 0, 1))
    array.append(crop)

    # rotate 90, flip v
    # rot90_flip_v = np.rot90(flip_v, 1, axes=(1, 2))
    # array.append(rot90_flip_v)

    # rotate 90, flip h
    rot90_flip_h = np.rot90(flip_h, 1, axes=(1, 2))
    array.append(rot90_flip_h)

    return array


def ensemble(array):
    new_array = []
    for cl in range(array.shape[1]):
        cn = list(array[:, cl]).count(1)
        all_cn = array.shape[0]
        if cn >= all_cn / 2.:
            new_array.append(1)
        else:
            new_array.append(0)

    return new_array