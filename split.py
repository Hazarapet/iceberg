import sys
import numpy as np
import pandas as pd

df_train = pd.read_json('resource/train.json')
values = df_train.values

# we should shuffle all examples
np.random.shuffle(values)

# splitting to train and validation set
index = int(len(values) * 0.85)
train, val = values[:index], values[index:]

df_tr = pd.DataFrame(train)
df_tr.columns = df_train.keys()

df_tr.to_json('resource/train_split.json')

df_val = pd.DataFrame(val)
df_val.columns = df_train.keys()

df_val.to_json('resource/val_split.json')
