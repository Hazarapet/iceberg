import cv2
import numpy as np
import pandas as pd


def iterate_minibatches(inputs, batchsize=10):

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield np.array(inputs)[excerpt]

    if len(inputs) % batchsize != 0:
        yield np.array(inputs)[- (len(inputs) % batchsize):]


def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'], c_row['band_2']], -1).reshape((2, 75, 75))], 1)
    out_images = np.stack(out_images).squeeze()
    return out_images


def aug(x_array, angle_array, y_array, x, a, y):
    # input's shape (cn, w, h)
    rn1 = np.random.randint(0, 16)
    rn2 = np.random.randint(x.shape[1] - 16, x.shape[1])  # this is much better

    # rotate 90
    # rt90 = np.rot90(x, 1, axes=(1, 2))
    # x_array = np.concatenate((x_array, [rt90]))
    # y_array = np.concatenate((y_array, [y]))
    # angle_array = np.concatenate((angle_array, [a]))

    # flip h
    flip_h = np.flip(x, 2)
    x_array = np.concatenate((x_array, [flip_h]))
    y_array = np.concatenate((y_array, [y]))
    angle_array = np.concatenate((angle_array, [a]))

    # shift H
    shift_h = x.transpose((1, 2, 0))
    M = np.float32([[1, 0, rn1], [0, 1, 0]])
    shift_h = cv2.warpAffine(shift_h, M, (shift_h.shape[0], shift_h.shape[1]))
    shift_h = shift_h.transpose((2, 0, 1))
    x_array = np.concatenate((x_array, [shift_h]))
    y_array = np.concatenate((y_array, [y]))
    angle_array = np.concatenate((angle_array, [a]))

    # shift V
    shift_v = x.transpose((1, 2, 0))
    M = np.float32([[1, 0, 0], [0, 1, rn2]])
    shift_v = cv2.warpAffine(shift_v, M, (shift_v.shape[0], shift_v.shape[1]))
    shift_v = shift_v.transpose((2, 0, 1))
    x_array = np.concatenate((x_array, [shift_v]))
    y_array = np.concatenate((y_array, [y]))
    angle_array = np.concatenate((angle_array, [a]))

    # flip v
    # flip_v = np.flip(x, 1)
    # x_array = np.concatenate((x_array, [flip_v]))
    # y_array = np.concatenate((y_array, [y]))
    # angle_array = np.concatenate((angle_array, [a]))

    # random crop with 32px shift
    # TODO Kind of overfit preventing technique
    crop = x.transpose((1, 2, 0))
    crop = cv2.resize(crop[rn1:rn2, rn1:rn2], (crop.shape[0], crop.shape[1]))
    crop = crop.transpose((2, 0, 1))
    x_array = np.concatenate((x_array, [crop]))
    y_array = np.concatenate((y_array, [y]))
    angle_array = np.concatenate((angle_array, [a]))

    return x_array, angle_array, y_array


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