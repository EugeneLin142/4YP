from functions import *
from embed_functions import *
from eval_functions import *
import pandas as pd

# Import filepaths
datacode, filepaths_raw = import_ima()
print("Number of filepaths before filtering:{}".format(len(filepaths_raw)))

# Remove all filepaths involved in TP quotes
filepaths = process_filepaths(filepaths_raw)
print("Number of filepaths after filtering:{}".format(len(filepaths)))

dim_size = 50
ft_model, encoded_fps = do_fasttext(filepaths, dim_size=dim_size, epochs=1)
data_pca = func_pca(datanp=encoded_fps, feat_cols=range(1, dim_size+1), drawplot=0, n_components=3)
data_pca_clustered = func_dbscan(data=data_pca[:, [-3,-2,-1]],
                        eps=0.05,
                        min_samples=2,
                        drawplot=0)

new_filepaths = generate_poisoned_filepaths(filepaths, data_pca_clustered)

pd.set_option('display.max_columns', 30)
print(new_filepaths)

# for i in range(0, len(filepaths)-1):
#     filepaths[i].append(data_pca_clustered[i][-1])




# df = pd.DataFrame()
# df["filepaths"] = filepaths
# model, encoded_fps = do_doc2vec(filepaths, epochs=1)

# process_list = import_process_temp()
# lev_list = process2lev(process_list)
# print(lev_list)
# datacode, filepaths_raw = import_ima()
# print(len(filepaths_raw))
# filepaths = process_filepaths(filepaths_raw)

# df = pd.DataFrame()
# df["filepaths"] = filepaths

# print(len(filepaths))


#
# # elbow_kmeans(data, maxrange=30)
# datatsne = func_tsne(datanp=encoded_fps, feat_cols=range(1,151), drawplot=1, n_components=3)
# dataumap = func_umap(datanp=encoded_fps, feat_cols=range(1,151), drawplot=1, n_components=3)
#
#
# dataumap = func_umap(datanp=encoded_fps, feat_cols=range(1,51), drawplot=0, n_components=3)
# dataumap_clustered = func_dbscan(data=dataumap[:, [50, 51, 52]],
#                                  eps=0.2,
#                                  min_samples=20,
#                                  drawplot=0)
#
# feat_cols = []
# feat_cols.append("filepaths")
# explore_cluster(datanp=filepaths,
#                 dataclust=dataumap_clustered,
#                 cols=feat_cols,
#                 filepaths=filepaths
#                 )

