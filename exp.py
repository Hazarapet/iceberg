import sys
import json
import numpy as np
import pandas as pd

with open('resource/val_split.json') as data_file:
    train = json.load(data_file)
    print np.array(train['id']).shape
