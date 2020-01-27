# ref material
# https://towardsdfscience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
# https://towardsdfscience.com/visualising-high-dimensional-dfsets-using-pca-and-t-sne-in-python-8ef87e7915b
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def high_corr_filter(df, drawmap):

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
    return df, selected_columns


from sklearn.decomposition import PCA


def func_pca(df, drawplot):
    # initialize
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

    return df


from sklearn.manifold import TSNE


def func_tsne(df, drawplot):
    # initialize
    rndperm = np.random.permutation(df.shape[0])
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    if drawplot == 1:
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue="uid",
            # palette=sns.color_palette(),
            data=df,
            legend="full",
            alpha=0.3
        )
        print("showing tSNE plot...")
        plt.show()
    return df


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
