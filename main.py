import numpy as np
import pandas as pd
import sys
import warnings
from functions import *
warnings.filterwarnings("ignore")

# for reproducibility
np.random.seed(123)

# print full arrays/dataframes for debugging
np.set_printoptions(threshold=sys.maxsize)

# names of columns extracted.
# note we are not using filenames right now, will need to look at that
# for unique identifier because of inode reuse
feat_cols = ["inode", "pid", "ppid", "uid", "euid", "gid", "egid"]

# import (a single file at the moment) into numpy array
datanp = np.genfromtxt('./ima/20200116.dat',
                       dtype=None,
                       delimiter=' ',
                       usecols=(5, 7, 8, 9, 10, 11, 12))

# convert data to pandas dataframe
datapd = pd.DataFrame(data=datanp,
                      columns=feat_cols)

# reduce dimensions with high correlation filter
# drawmap=1 to see heatmap for justification
datapd, red_cols = high_corr_filter(df=datapd,
                                    drawmap=0)
print("datapd.shape", datapd.shape)

# reduce dimensions using tSNE (to 2 dimensions)
datatsne = func_tsne(df=datapd,
                     drawplot=0)
print("datatsne.shape", datatsne.shape)  # looks like pandas is doing something spooky, maybe do EVERYTHING in numpy and convert to pandas in-function?
print("datapd.shape", datapd.shape)

# convert dataframe back to numpy and perform 2D DBSCAN
datanp = datatsne.to_numpy()
func_dbscan(data=datanp[:, [5, 6]],
            eps=0.4,
            min_samples=7,
            drawplot=0)

print("datapd.shape", datapd.shape)
# reduce dimensions using PCA
data_pca = func_pca(df=datapd,
                    drawplot=0)
print("data_pca.shape", data_pca.shape)  # WHY 10 COLUMNS??????????????????

# convert dataframe back to numpy and perform 2D DBSCAN  (((PCA)))
data_pca_for_dbscan = data_pca.to_numpy()
func_dbscan(data=data_pca_for_dbscan[:, [5, 6]],
            eps=0.4,
            min_samples=7,
            drawplot=0)
