import numpy as np
import pandas as pd
import sys
import warnings
from functions import *
warnings.filterwarnings("ignore")

# I will work with only numpy in main.py for both simplicity's sake, and because of pandas
# handling memory strangely

# for reproducibility
np.random.seed(123)

# print full arrays/dataframes for debugging
np.set_printoptions(threshold=sys.maxsize)

# names of columns extracted.
# note we are not using filenames right now, will need to look at that
# for unique identifier because of inode reuse
feat_cols = ["inode", "pid", "ppid", "uid", "euid", "gid", "egid"]

# import all .dat files from ./ima/ into numpy array
data = None
datanp = import_ima(data)

# reduce dimensions with high correlation filter
# drawmap=1 to see heatmap for justification
datanp, red_cols = high_corr_filter(datanp=datanp, feat_cols=feat_cols,
                                    drawmap=0)
print("datanp.shape", datanp.shape)

# reduce dimensions using UMAP (to 2 dimensions), return numpy
dataumap = func_umap(datanp=datanp, feat_cols=red_cols,
                     drawplot=1)
print("dataumap.shape", dataumap.shape)
print("datanp.shape", datanp.shape)

# reduce dimensions using tSNE (to 2 dimensions), return numpy
datatsne = func_tsne(datanp=datanp, feat_cols=red_cols,
                     drawplot=0)
print("datatsne.shape", datatsne.shape)
print("datanp.shape", datanp.shape)

# perform 2D DBSCAN (TSNE)
func_dbscan(data=datatsne[:, [5, 6]],
            eps=0.4,
            min_samples=7,
            drawplot=0)

# reduce dimensions using PCA, return numpy
data_pca = func_pca(datanp=datanp, feat_cols=red_cols,
                    drawplot=0)
print("data_pca.shape", data_pca.shape)

# perform 2D DBSCAN  (PCA)
func_dbscan(data=data_pca[:, [5, 6]],
            eps=0.4,
            min_samples=7,
            drawplot=0)
