import sys
import time
import json
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.utils import plot_model
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.initializers import RandomUniform, RandomNormal
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import callbacks

st_time = time.time()
BATCH_SIZE = 1000
N_EPOCH = 200
REG = 1e-4
DP = 0.5

# Read in our input data
df_train = pd.read_csv('resource/train_split.csv')
df_val = pd.read_csv('resource/val_split.csv')

df_test = pd.read_csv('resource/test.csv')

with open('mean.json', 'r') as outfile:
    mean = np.array(json.load(outfile))

# This prints out (rows, columns)
print 'df_train shape:', df_train.shape
print 'df_val shape:', df_val.shape
print 'df_test shape:', df_test.shape
print 'mean shape:', mean.shape

y_train = np.array(df_train['target'].values)
y_val = np.array(df_val['target'].values)

# y_train = to_categorical(y_train, num_classes=10)
# y_val = to_categorical(y_val, num_classes=10)

x_train = np.array(df_train.drop(['target', 'id'], axis=1) - mean)
x_val = np.array(df_val.drop(['target', 'id'], axis=1) - mean)

id_test = df_test['id'].values
x_test = np.array(df_test.drop(['id'], axis=1) - mean)

print '\ny_train shape:', y_train.shape
print 'x_train shape:', x_train.shape
print 'y_val shape:', y_val.shape
print 'x_val shape:', x_val.shape

model = Sequential()
model.add(Dense(32, input_shape=(57,)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Dropout(DP))

model.add(Dense(32, kernel_regularizer=l2(REG)))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Dropout(DP))

model.add(Dense(1, activation='sigmoid'))

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

adam = Adam(lr=1e-4, decay=1e-5)
sgd = SGD(lr=1e-3, momentum=.9, decay=1e-5)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

rm_cb = callbacks.RemoteMonitor()
ers_cb = callbacks.EarlyStopping(patience=20)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=N_EPOCH, batch_size=BATCH_SIZE, callbacks=[rm_cb, ers_cb], shuffle=True)

print('================= Validation =================')
[v_loss, v_acc] = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE, verbose=1)
print('\nVal Loss: {:.5f}, Val Acc: {:.5f}'.format(v_loss, v_acc))

p_test = model.predict(x_test)

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = p_test
sub.to_csv('fast_result.csv', index=False)

print(sub.head())


print('\n{:.2f}m Runtime'.format((time.time() - st_time) / 60))
print '====== End ======'

