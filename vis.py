import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BAND = 1
df_val = pd.read_json('resource/val_split.json')

x_val = df_val.drop(['is_iceberg', 'id'], axis=1)
x_val = x_val.apply(lambda c_row: [np.stack([c_row['band_1'], c_row['band_2']]).reshape((2, 75, 75))], 1)

x_val = np.stack(x_val).squeeze().astype(np.float32)

print 'shape: ', x_val.shape

gs = gridspec.GridSpec(int(math.sqrt(x_val[:100].shape[0])), int(math.sqrt(x_val[:100].shape[0])),
                       top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
i = 0
for g in gs:
    tmp = np.abs((x_val[i][0] - x_val[i][1]))
    ax = plt.subplot(g)
    ax.imshow(tmp)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    i += 1

plt.savefig('visual/band_abs_{}.jpg'.format(BAND), dpi=100, bbox_inches='tight')
print 'Visualizations are done!'
