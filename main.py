from functions import *
from embed_functions import *
from hyperopt_eval_functions import *
import pandas as pd
import numpy as np

# Import filepaths
datacode, filepaths_raw = import_ima()
print("Number of quotes before filtering:{}".format(len(filepaths_raw)))

# Remove all filepaths involved in TP quotes
data, filepaths = process_filepaths(datacode, filepaths_raw, data_flag=1)
print("Number of quotes after removing ima processes:{}".format(len(filepaths)))

# convert known ppids/pids to names
data = pid_to_pname(data)

# remove unknown associated filepaths
data_df_named = trim_data(data, filepaths)
print("Number of quotes after removing unknown processes:{}".format(len(data)))

data_df_embed = pd.DataFrame(data=None)
data_df_embed["process_name"] = ascii_sum_embed(data_df_named["process_name"].tolist())
data_df_embed["parent_process_name"] = ascii_sum_embed(data_df_named["parent_process_name"].tolist())
data_df_embed["filepath_tokens"] = data_df_named["filepath_tokens"]

dim_size = 150
ft_model, encoded_fps = do_fasttext(data_df_embed["filepath_tokens"], dim_size=dim_size, epochs=1)
data_pca = func_pca(datanp=encoded_fps, feat_cols=range(1, dim_size+1), drawplot=0, n_components=3)

for column in ["process_name", "parent_process_name"]:
    data_pca = np.concatenate((data_pca, data_df_embed[column].to_numpy().reshape((-1, 1))), axis=1)

# final_pca = func_pca(datanp=data_pca[:, [-5, -4, -3, -2, -1]], feat_cols=range(1, 6), drawplot=1, n_components=3)
data_pca_clustered = func_dbscan(data=data_pca[:, [-5,-4,-3,-2,-1]],
                        eps=0.4,
                        min_samples=2,
                        drawplot=1)


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

