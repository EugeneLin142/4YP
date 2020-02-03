# ref material
# https://towardsdfscience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
# https://towardsdfscience.com/visualising-high-dimensional-dfsets-using-pca-and-t-sne-in-python-8ef87e7915b
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
from mpl_toolkits.mplot3d import Axes3D as ax

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def plot_it(data, n_components, pd_or_np, dimres_method):
    if pd_or_np == "pd":
        x = dimres_method + "-one"
        y = dimres_method + "-two"
        z = dimres_method + "-three"
        if n_components == 2:
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                            x=x, y=y,
                            # hue="uid",
                            # palette=sns.color_palette(),
                            data=data,
                            legend="full",
                            alpha=0.3
                            )

        if n_components == 3:
            plt.figure(figsize=(16, 10))
            sanD_plot = plt.figure().gca(projection='3d')
            sanD_plot.scatter(data[x], data[y], data[z])
            sanD_plot.set_xlabel(x)
            sanD_plot.set_ylabel(y)
            sanD_plot.set_zlabel(z)
            plt.show()

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=x, y=y,
                # hue="uid",
                # palette=sns.color_palette(),
                data=data,
                legend="full",
                alpha=0.3
            )
            plt.show()

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=x, y=z,
                # hue="uid",
                # palette=sns.color_palette(),
                data=data,
                legend="full",
                alpha=0.3
            )
            plt.show()

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=y, y=z,
                # hue="uid",
                # palette=sns.color_palette(),
                data=data,
                legend="full",
                alpha=0.3
            )
            plt.show()

    print("showing ", dimres_method, " plot...")

    if pd_or_np == "np":
        print("uncompleted code lol")

def import_ima(data):
    for filename in os.listdir('./ima/'):
        if filename.endswith(".dat"): # Make sure we're iterating over .dat IMA quotes
            filename = './ima/' + filename
            # print(os.path.join(directory, filename))
            print(filename)
            if data is None:
                data = np.genfromtxt(filename,
                       dtype=None,
                       delimiter=' ',
                       usecols=(5, 7, 8, 9, 10, 11, 12))
                filepaths = np.genfromtxt(filename,
                                          dtype=None,
                                          delimiter=' ',
                                          usecols=(4))
            else:
                datanew = np.genfromtxt(filename,
                                        dtype=None,
                                        delimiter=' ',
                                        usecols=(5, 7, 8, 9, 10, 11, 12))
                filepathsnew = np.genfromtxt(filename,
                                          dtype=None,
                                          delimiter=' ',
                                          usecols=(4))
                try:
                    data = np.concatenate((data, datanew), axis=0)
                    filepaths = np.concatenate((filepaths, filepathsnew), axis=0)
                except:
                    print(filename, " is likely not encoded properly, skipping this quote...")
                    pass
                        # for some reason some quotes are read in as 0-dimensional
                        # I believe because the data in the quotes aren't encoded properly
                        # and numpy is taking them as bytes literal - i will ignore for now...
                    # for array in datanew: # convert each row to 8-dimensions & concatenate 1 by 1
                    #     # print("array0", array[0].decode('utf-8'))
                    #     rowlabel = np.array([[array[0].decode('utf-8')]])
                    #     row = np.array([[array[1]], [array[2]],
                    #                     [array[3]], [array[4]], [array[5]],
                    #                     [array[6]], [array[7]] ])
                    #     print(rowlabel.shape, row.astype(int).shape)
                    #     row = np.concatenate((rowlabel, row.astype(int)), axis=0)
                    #     print("before,", row)
                    #     row = row.reshape(row.shape[1], row.shape[0])
                    #     print("after,", row)
                    #     print("check if match...", data[0])
                    #     input("check now")
                        # print(data.shape)
                        # print(row.shape)
                        ###
                        # above 'print' comments to confirm reshape works properly
                        # data = np.concatenate((data, row), axis=0)

                        # !!!
                        # NOTE, WE ASSUME THAT THE first DAY IS NOT 0-DIMENSIONAL !!
                        # !!!
        else:
            continue
        print("importing next...")

    if data is None:
        print("No IMA measurements found, please put .dat or .csv files into ./ima/")
    else:
        print("All IMA measurements in ./ima/ imported successfully.")

    return data, filepaths


def high_corr_filter(datanp, feat_cols, drawmap):

    df = pd.DataFrame(data=datanp, columns=feat_cols)

    # initialize label encoder
    label_encoder = LabelEncoder()
    # df.iloc[:,0] = label_encoder.fit_transform(df.iloc[:,0]).astype('float64')

    # determine correlation between columns
    corr = df.corr()

    # draw heatmap if required to show justification
    plt.figure(figsize=(16, 10))
    if drawmap == 1:
        sns.heatmap(corr, annot=True)
        plt.show()

    # choose enough columns to represent data
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]

    # only present representative columns & return them
    df = df[selected_columns]
    datanp = df.to_numpy()

    return datanp, selected_columns


from sklearn.decomposition import PCA


def func_pca(datanp, feat_cols, drawplot):
    np.random.seed(123)
    # initialize
    df = pd.DataFrame(data=datanp, columns=feat_cols)
    rndperm = np.random.permutation(df.shape[0])
    pca = PCA(n_components=3)
    # can adjust how many components, but will have to edit
    # below code as well.
    pca_result = pca.fit_transform(df.values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    # display the variation per PC.
    # unfortunately it seems it's not very representative of the variation,
    # except the first PC. PCA is probably not great.
    print('Explained variation per principal component:{}'.format(
        pca.explained_variance_ratio_
    ))

    if drawplot == 1:
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            # hue="uid",
            # palette=sns.color_palette("hls", 8),
            data=df.loc[rndperm, :],
            legend="full",
            alpha=0.3
        )
        print("showing plot...")
        plt.show()

    datanp = df.to_numpy()
    return datanp


from sklearn.manifold import TSNE


def func_tsne(datanp, feat_cols, drawplot, n_components):
    np.random.seed(123)
    # initialize
    df = pd.DataFrame(data=datanp, columns=feat_cols)
    rndperm = np.random.permutation(df.shape[0])
    time_start = time.time()
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    try:
        df['tsne-three'] = tsne_results[:, 2]
        print("tsne done in 3 dimensions")
    except:
        print("tsne done in 2 dimensions")

    if drawplot == 1:
        plot_it(data=df,
                n_components=n_components,
                pd_or_np="pd",
                dimres_method="tsne")

    datanp = df.to_numpy()
    return datanp


import umap


def func_umap(datanp, feat_cols, drawplot, n_components):
    np.random.seed(123)
    # initialize
    df = pd.DataFrame(data=datanp, columns=feat_cols)
    rndperm = np.random.permutation(df.shape[0])
    time_start = time.time()
    reducer = umap.UMAP(n_components=n_components)
    embedding = reducer.fit_transform(df)
    print('UMAP done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df['umap-one'] = embedding[:, 0]
    df['umap-two'] = embedding[:, 1]

    try:
        df['umap-three'] = embedding[:, 2]
        print("umap done in 3 dimensions")
    except:
        print("umap done in 2 dimensions")

    if drawplot == 1:
        plot_it(data=df,
                n_components=n_components,
                pd_or_np="pd",
                dimres_method="umap")

    datanp = df.to_numpy()
    return datanp


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def func_dbscan(data, eps, min_samples, drawplot):
    data = data

    # initialize
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))

    if drawplot == 1:
        plt.figure(figsize=(16, 10))  # only here for jupyter notebook, otherwise creates empty plots
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    # print(data.shape)
    # print(labels.shape)
    df = pd.DataFrame(data=data, columns=["red-1", "red-2"])
    df["clusters"] = labels
    data = df.to_numpy()

    return data #3rd column is labels


from IPython.core.display import display, HTML


def explore_cluster(datanp, dataclust, cols, filepaths):
    df = pd.DataFrame(data=datanp, columns=cols)
    # df["comp-1"] = dataclust[:, 0]
    # df["comp-2"] = dataclust[:, 1]
    df_display = pd.DataFrame()
    df_display["inode"] = df["inode"]
    df_display["pid"] = df["pid"]
    df_display["filepaths"] = filepaths
    df_display["cluster"] = dataclust[:, 2]

    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', -1)

    df_display = df_display.sort_values(by=['cluster', "inode", "pid"])
    display(HTML(df_display.to_html()))
    # print(datanp[datanp[:, 7].argsort()])
