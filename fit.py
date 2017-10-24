import sys
import time
import json
import numpy as np
import pandas as pd
from utils.common import load_and_format
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras import callbacks as keras_cb
from models.resnet50.cresnet50 import model

st_time = time.time()
BATCH_SIZE = 100
N_EPOCH = 100
REG = 1e-4
DP = 0.5

# Read in our input data
df_train = load_and_format('resource/train_split.json')
df_val = load_and_format('resource/val_split.json')

# This prints out (rows, columns)
print 'df_train shape:', df_train.shape
print 'df_val shape:', df_val.shape

y_train = np.array(df_train['is_iceberg'].values)
y_val = np.array(df_val['is_iceberg'].values)

x_train = np.array(df_train.drop(['is_iceberg', 'inc_angle', 'id'], axis=1).values)
x_val = np.array(df_val.drop(['is_iceberg', 'inc_angle', 'id'], axis=1).values)


print '\nx_train shape:', x_train.shape
print 'x_val shape:', x_val.shape

sys.exit()
print 'model loading...'
[model, structure] = model()

model.summary()
plot_model(model, to_file='cresnet50.png', show_shapes=True)

adam = Adam(lr=1e-4, decay=1e-5)
sgd = SGD(lr=1e-3, momentum=.9, decay=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

rm_cb = keras_cb.RemoteMonitor()
ers_cb = keras_cb.EarlyStopping(patience=20)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=N_EPOCH, batch_size=BATCH_SIZE, callbacks=[rm_cb, ers_cb], shuffle=True)

print('================= Validation =================')
[v_loss, v_acc] = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE, verbose=1)
print('\nVal Loss: {:.5f}, Val Acc: {:.5f}'.format(v_loss, v_acc))


# create file name to save the state with useful information
timestamp = str(time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()))
model_filename = structure + \
                 '-val_l:' + str(round(v_loss, 4)) + \
                 '-val_acc:' + str(round(v_acc, 4)) + \
                 '-time:' + timestamp + '-dur:' + str(round((time.time() - st_time) / 60, 3))

# saving the weights
model.save_weights(model_filename + '.h5')

with open(model_filename + '.json', 'w') as outfile:
    json_string = model.to_json()
    json.dump(json_string, outfile)


print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

