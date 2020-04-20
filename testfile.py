# filename = 'star_wars_holiday.txt'
# file = open(filename, 'rt')
# text = file.read()
# file.close()
# # split into words by white space
# words = text.split()
# # convert to lower case
# words = [word.lower() for word in words]
import numpy as np
import pandas as pd


data_df_embed = pd.DataFrame(data=None)
data_df_embed["process_name"] = [0, 1]
data_df_embed["parent_process_name"] = [222, 777]

from sklearn.cluster import DBSCAN

data_pca = np.array([[1,2,3,4,5],
                     [2,3,4,5,6],
                     [3,4,5,6,7],
                     [234243,22,3,4,5],
                     [0,0,0,0,0]
                     ])

db = DBSCAN(eps=5, min_samples=2).fit(data_pca)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=db.labels_, s=5)
ax.view_init(azim=200)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()