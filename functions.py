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
import plotly.express as px
from orderset import OrderedSet
from mpl_toolkits.mplot3d import Axes3D as ax

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def plot_it(n_components, pd_or_np, last_process, data = None,
            db_labels = None, db_core_samples_mask = None,
            db_n_clusters = None):
    if pd_or_np == "pd":
        x = last_process + "-one"
        y = last_process + "-two"
        z = last_process + "-three"

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
            plt.show()

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

    print("showing ", last_process, " plot...")

    if pd_or_np == "np":
        if last_process == "DBSCAN":
            if n_components == 2:
                plt.figure(figsize=(16, 10))  # only here for jupyter notebook, otherwise creates empty plots
                unique_labels = set(db_labels)
                colors = [plt.cm.Spectral(each)
                          for each in np.linspace(0, 1, len(unique_labels))]
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        # Black used for noise.
                        col = [0, 0, 0, 1]

                    class_member_mask = (db_labels == k)

                    xy = data[class_member_mask & db_core_samples_mask]
                    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                             markeredgecolor='k', markersize=14)

                    xy = data[class_member_mask & ~db_core_samples_mask]
                    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                             markeredgecolor='k', markersize=6)

                plt.title('Estimated number of clusters: %d' % db_n_clusters)
                plt.show()
        else:
            print("Plot failed, What did you pass to the plotting function?")


def import_ima():
    data = None
    for filename in os.listdir('./ima/'):
        if filename.endswith(".dat"): # Make sure we're iterating over .dat IMA quotes
            filename = './ima/' + filename
            # print(os.path.join(directory, filename))
            print(filename)
            if data is None:
                data = np.genfromtxt(filename,
                       dtype='unicode',
                       delimiter=' ',
                       usecols=(5, 7, 8, 9, 10, 11, 12))
                filepaths = np.genfromtxt(filename,
                                          dtype='unicode',
                                          delimiter=' ',
                                          usecols=(4))
            else:
                try:
                    datanew = np.genfromtxt(filename,
                                            dtype='unicode',
                                            delimiter=' ',
                                            usecols=(5, 7, 8, 9, 10, 11, 12))
                    filepathsnew = np.genfromtxt(filename,
                                                 dtype='unicode',
                                                 delimiter=' ',
                                                 usecols=(4))
                except:
                    print(filename, " is likely not encoded properly, (search code WALNUT) skipping this quote...")
                    pass

                try:
                    data = np.concatenate((data, datanew), axis=0)
                    filepaths = np.concatenate((filepaths, filepathsnew), axis=0)
                except:
                    print(filename, " is likely not encoded properly, (search code BERRY) skipping this quote...")
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


def process_filepaths(data, filepaths):
    # make filepaths into an appropriate list, and remove /etc/intel/cloudsecurity/data/nonce	and aikquote
    filepaths = [fp.split('/') for fp in
                 [fp.strip('/') for fp in filepaths]]

    fp_filtered = []
    data_filtered = None

    for line in filepaths:
        line_index = filepaths.index(line)
        if line[0] == "etc" and line[1] == "intel" and line[2] == "cloudsecurity" and line[3] == "data":
            if line[4] == "nonce" or line[4] == "aikquote":
                pass
        elif line[0] == "root" and line[1] == "ima_archive" and line[2] == "data":
            pass
        else:
            for token in line:
                token = token.lower()

            if fp_filtered == None:  # Actually I think this does nothing but don't want to change what's working
                fp_filtered = line
                data_filtered = data[:, line_index]
            else:
                fp_filtered.append(line)
                if data_filtered is None:
                    data_filtered = np.array(data[line_index, :])
                else:
                    try:
                        data_filtered = np.concatenate(([data_filtered], np.array([data[line_index, :]])), axis=0)
                    except:
                        data_filtered = np.concatenate((data_filtered, np.array([data[line_index, :]])), axis=0)
                        # above is jank as fuck but then again so is np >:( angory
    # if fp_filtered[-1] == None:
    #     del fp_filtered[-1]

    return data_filtered, fp_filtered


def pid_to_pname(data):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    # first import list of known pid/process pairs
    pairlist = 'nextcloud_ps.csv'
    pairlist = pd.read_csv(pairlist)

    # make dataframe for process names
    new_data = pd.DataFrame(data=None)
    new_data["process_name"] = data[:, 1]           # first read in pids
    new_data["parent_process_name"] = data[:, 2]    # and ppids

    # alter COMMAND field to remove [ and ] to obtain 'process name'
    for command in pairlist["COMMAND"]:
        command_index = pairlist["COMMAND"].tolist().index(command)
        if command[0] == "[":
            pairlist["COMMAND"][command_index] = command[1:-1]

    # Convert some lists to ordered sets for lookup speed
    pidlist = np.array(pairlist["PID"])
    procnamelist = np.array(pairlist["COMMAND"])
    parent_pname = np.array(new_data["parent_process_name"]).astype(np.int)
    pname = np.array(new_data["process_name"]).astype(np.int)

    # look for matches of known pairs in parent processes
    for current_search in parent_pname:
        if current_search == 1:
            pass
        try:
            searchterm = np.where(pidlist == current_search)[0][0]
        except:
            searchterm = None
        if searchterm is not None:
            data_pid_index = np.where(parent_pname == current_search)[0][0]
            parent_pname[data_pid_index] = 9999999
            new_data["parent_process_name"][data_pid_index] = procnamelist[searchterm]

    # repeat for processes
    for current_search in pname:
        if current_search == 1:
            pass
        try:
            searchterm = np.where(pidlist == current_search)[0][0]
        except:
            searchterm = None
        if searchterm is not None:
            data_pid_index = np.where(pname == current_search)[0][0]
            pname[data_pid_index] = 9999999
            new_data["process_name"][data_pid_index] = procnamelist[searchterm]

    # Should work but way too slow (alternative for above block)
    # for pid in pairlist["PID"]:
    #     # look for matches of known pairs
    #     for current_search in new_data["parent_process_name"]:
    #         data_pid_index = new_data["parent_process_name"].tolist().index(current_search)
    #         if current_search == pid:
    #             new_data["parent_process_name"][data_pid_index] = pairlist.loc[pairlist['PID'] == pid, 'COMMAND']
    #
    #     for current_search in new_data["process_name"]:
    #         data_pid_index = new_data["process_name"].tolist().index(current_search)
    #         if current_search == pid:
    #             new_data["process_name"][data_pid_index] = pairlist.loc[pairlist['PID'] == pid, 'COMMAND']

    return new_data


def trim_data(data, filepaths):
    olddata = pd.DataFrame(data=None)
    olddata["process_name"] = data["process_name"]
    olddata["parent_process_name"] = data["parent_process_name"]
    olddata["filepath_tokens"] = filepaths
    new_data = []
    for index, row in olddata.iterrows():
        if row["process_name"].isdigit() is True:
            if row["parent_process_name"].isdigit() is True:
                pass
            else:
                new_data.append([row["process_name"], row["parent_process_name"], row["filepath_tokens"]])
        else:
            new_data.append([row["process_name"], row["parent_process_name"], row["filepath_tokens"]])

    new_data = pd.DataFrame.from_records(new_data)
    new_data.columns = ["process_name", "parent_process_name", "filepath_tokens"]
    return new_data


def save_listoflists_tofile(listoflists, name):
    name = name + ".txt"
    with open(name, 'w') as f:
        for _list in listoflists:
            for _string in _list:
                # f.seek(0)
                f.write(str(_string) + ' ')
            f.write('\n')
    print("File saved to: {}".format(name))



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


def func_pca(datanp, feat_cols, drawplot, n_components):
    np.random.seed(123)
    # initialize
    df = pd.DataFrame(data=datanp, columns=feat_cols)
    rndperm = np.random.permutation(df.shape[0])
    pca = PCA(n_components=n_components)
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
                last_process="tsne")

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
                last_process="umap")

    datanp = df.to_numpy()
    return datanp


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def func_dbscan(data, eps, min_samples, drawplot):
    data = data
    if data.shape[1] == 2:
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

        df = pd.DataFrame(data=data, columns=["db-1", "db-2"])
        df["clusters"] = labels
        data = df.to_numpy()

        if drawplot == 1:
            plot_it(n_components=data.shape[1], pd_or_np="np",
                    last_process="DBSCAN", db_core_samples_mask=core_samples_mask,
                    db_labels=labels, db_n_clusters=n_clusters_)
    elif data.shape[1] == 3:

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=300)
        # ax.view_init(azim=200)
        # plt.show()

        # db = DBSCAN(eps=eps, min_samples=min_samples)
        # db.fit_predict(data)
        # pred = db.fit_predict(data)
        # if drawplot == 1:
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=db.labels_, s=300)
        #     ax.view_init(azim=200)
        #     plt.show()

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        ########################
        if drawplot == 1:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=db.labels_, s=5)
            ax.view_init(azim=200)
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.show()
            # plt.figure(figsize=(16, 10))  # only here for jupyter notebook, otherwise creates empty plots
            unique_labels = set(labels)
            # colors = [plt.cm.Spectral(each)
            #             for each in np.linspace(0, 1, len(unique_labels))]
            # for k, col in zip(unique_labels, colors):
            #     if k == -1:
            #         # Black used for noise.
            #         col = [0, 0, 0, 1]
            #
            #     class_member_mask = (labels == k)
            #
            #     xyz = data[class_member_mask & core_samples_mask]
            #     plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=14)
            #
            #     xyz = data[class_member_mask & ~core_samples_mask]
            #     plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=6)

        ########################

        print("number of cluster found: {}".format(len(set(db.labels_))))
        print('cluster for each point: ', db.labels_)

        labels = db.labels_
        df = pd.DataFrame(data=data, columns=["db-1", "db-2", "db-3"])
        df["clusters"] = labels
        data = df.to_numpy()

    elif data.shape[1] == 5:
        # For naive clustering - using no relation between process and parent process names
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)

        if drawplot == 1:
            df = pd.DataFrame(data=None)
            df["pca-1"] = data[:, 0]
            df["pca-2"] = data[:, 1]
            df["pca-3"] = data[:, 2]
            df["cluster"] = db.labels_
            fig = px.scatter_3d(df, x='pca-1', y='pca-2', z='pca-3',
                                color='cluster')
            fig.show()

            # matplotlib not working...
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter(data[:, 0], data[:, 1], data[:, 2]) #, c=db.labels_, s=5
            # ax.view_init(azim=200)
            # plt.title('Estimated number of clusters: %d' % n_clusters_)
            # plt.show()
    # print(data.shape)
    # print(labels.shape)

    # last column is labels
    return data


from IPython.core.display import display, HTML


def explore_cluster(datanp, dataclust, cols, filepaths):
    try:
        df = pd.DataFrame(data=datanp, columns=cols)
    # df["comp-1"] = dataclust[:, 0]
    # df["comp-2"] = dataclust[:, 1]
    except:
        pass

    df_display = pd.DataFrame()
    try:
        df_display["inode"] = df["inode"]
    except:
        pass
    try:
        df_display["pid"] = df["pid"]
    except:
        pass
    try:
        df_display["filepaths"] = filepaths
    except:
        pass
    df_display["cluster"] = dataclust[:, -1]


    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', -1)

    df_display = df_display.sort_values(by=['cluster']) #, "inode", "pid"
    display(HTML(df_display.to_html()))
    # print(datanp[datanp[:, 7].argsort()])


def import_process_temp():
    with open("processlist.txt") as list:
        process_names = []
        for item in list:
            process_names.append(item[0:-5])
    return process_names