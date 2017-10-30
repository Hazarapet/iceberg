import sys
import json
import numpy as np
import pandas as pd

df_train = pd.read_json('resource/train_split.json')

train_angle = df_train['inc_angle'].values

print np.min(train_angle)
print np.max(train_angle)
print train_angle[:100]