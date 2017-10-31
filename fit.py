import sys
import time
import json
import numpy as np
import pandas as pd
from utils import common
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras import callbacks as keras_cb
from models.model.simple import model as simple
from models.resnet50.cresnet50 import model as cres_model

st_time = time.time()
BATCH_SIZE = 50
WIDTH = 75
HEIGHT = 75
N_EPOCH = 400
AUGMENT = True
REG = 1e-4

# Read in our input data
df_train = pd.read_json('resource/train_split.json')
df_val = pd.read_json('resource/val_split.json')

# This prints out (rows, columns)
print 'df_train shape:', df_train.shape
print 'df_val shape:', df_val.shape

y_train = np.array(df_train['is_iceberg'].values)
y_val = np.array(df_val['is_iceberg'].values)

train_angle = df_train['inc_angle'].values
train_angle[train_angle == 'na'] = -1

val_angle = df_val['inc_angle'].values
val_angle[val_angle == 'na'] = -1

train_angle = np.array(train_angle)
val_angle = np.array(val_angle)

################################################################
x_train = df_train.drop(['is_iceberg', 'id'], axis=1)
x_val = df_val.drop(['is_iceberg', 'id'], axis=1)

x_train = x_train.apply(lambda c_row: [np.stack([c_row['band_1'], c_row['band_2']]).reshape((2, 75, 75))], 1)
x_val = x_val.apply(lambda c_row: [np.stack([c_row['band_1'], c_row['band_2']]).reshape((2, 75, 75))], 1)

x_train = np.stack(x_train).squeeze().astype(np.float32)
x_val = np.stack(x_val).squeeze().astype(np.float32)

# adding new channels as new features
x_train = np.concatenate([x_train, np.abs(x_train[:, 0] - x_train[:, 1])[:, np.newaxis, :, :]], axis=1)
x_val = np.concatenate([x_val, np.abs(x_val[:, 0] - x_val[:, 1])[:, np.newaxis, :, :]], axis=1)

if AUGMENT:
    count = 0
    for x, a, y in zip(x_train, train_angle, y_train):
        x_train, train_angle, y_train = common.aug(x_train, train_angle, y_train, x, a, y)
        count += 1
        print 'count: {}'.format(count)


################################################################

print '\ntrain_angle shape:', train_angle.shape
print 'x_train shape:', x_train.shape
print 'val_angle shape:', val_angle.shape
print 'x_val shape:', x_val.shape

print 'model loading...'
[model, structure] = simple()

model.summary()
plot_model(model, to_file='simple.png', show_shapes=True)

adam = Adam(lr=1e-3, decay=1e-5)
sgd = SGD(lr=1e-1, momentum=.9, decay=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

################################################################


def schedule(epoch):
    lr = K.get_value(model.optimizer.lr)  # this is the current learning rate
    return lr * (0.5 ** (1 * (int(epoch % 40) == 0)))

rm_cb = keras_cb.RemoteMonitor()
ers_cb = keras_cb.EarlyStopping(patience=60)
lr_cb = keras_cb.LearningRateScheduler(schedule)

################################################################

model.fit([x_train, train_angle], y_train,
          validation_data=([x_val, val_angle], y_val),
          epochs=N_EPOCH,
          batch_size=BATCH_SIZE,
          callbacks=[rm_cb, ers_cb, lr_cb],
          shuffle=True)

print('================= Validation =================')
[v_loss, v_acc] = model.evaluate([x_val, val_angle], y_val, batch_size=BATCH_SIZE, verbose=1)
print('\nVal Loss: {:.5f}, Val Acc: {:.5f}'.format(v_loss, v_acc))


# create file name to save the state with useful information
timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
model_filename = structure + \
                 'val_l:' + str(round(v_loss, 4)) + \
                 '-val_acc:' + str(round(v_acc, 4)) + \
                 '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))

# saving the weights
model.save_weights(model_filename + '.h5')


print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

