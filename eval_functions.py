import numpy as np
import pandas as pd
from random import seed
from random import randint
from random import shuffle
import time
from functions import *
from embed_functions import *
# write a function that takes list of lists of strings as its input
# it outputs a user-defined number of lists which are artificial variations on the originals
# eg. replace the last string in a list with another string at the end of another list

# use dfs to include cluster so we can eval

def generate_poisoned_filepaths(filepaths, clustered_data):
    # pass cluster labels to algorithm so that new filepaths retain cluster labels
    fp_df = pd.DataFrame(data=None)
    fp_df["filepath"] = filepaths
    fp_df["cluster_label"] = clustered_data[:, -1]

    number_of_samples = int(len(filepaths)/100)
    seed(123)

    # read in simple list of pypi packages
    with open("simple.txt", 'r') as a:
        piplist = a.read().split(' ')

    # generate random python package names
    piplist_samples = []
    randint_list = []

    # generate non-repeating list of integers
    for rint in range(0, number_of_samples):
        randint_list.append(rint)
    shuffle(randint_list)

    # pull samples from indexes on list
    for rint in range(0, number_of_samples):
        # rint = randint(0, len(piplist)-1)
        piplist_samples.append(piplist[randint_list.pop()])

    # # check for python2.7 filepaths
    # new_fp_df = pd.DataFrame(data=None)
    # for filepath in filepaths:
    #     if filepath[0] == "opt" and filepath[1] == "eff.org" and filepath[2] == "certbot" and filepath[3] == "venv" and filepath[4] == "lib" and filepath[5] == "python2.7" and filepath[6] == "site-packages":
    #         if len(filepath) is not 9:
    #
    #             temp_filepaths.append(filepath)
    # print("\nold:")
    # print(temp_filepaths)

    # check for python2.7 filepaths
    i = 0  # counter for features in new_fp_df
    new_cluster_labels = []
    new_filepaths = []
    for row in range(0, len(filepaths)-1):
        if fp_df["filepath"][row][0] == "opt" and \
                fp_df["filepath"][row][1] == "eff.org" and \
                fp_df["filepath"][row][2] == "certbot" and \
                fp_df["filepath"][row][3] == "venv" and \
                fp_df["filepath"][row][4] == "lib" and \
                fp_df["filepath"][row][5] == "python2.7" and \
                fp_df["filepath"][row][6] == "site-packages" and \
                len(fp_df["filepath"][row]) is not 8:
            new_filepaths.append(fp_df["filepath"][row])
            new_cluster_labels.append(fp_df["cluster_label"][row])
    filtered_fp_df = pd.DataFrame(data=None)
    filtered_fp_df["filepath"] = new_filepaths
    filtered_fp_df["cluster_label"] = new_cluster_labels

    # print("\nold:")
    # print(new_fp_df)

    # replace package name with random ones
    new_fp = []
    og_fp = []
    clus_label = []
    old_package_list = []
    new_pacakge_list = []
    new_fp_df = pd.DataFrame(data=None)
    for rint in range(0, number_of_samples):

        # take random sample from py2.7 paths
        rint = randint(0, len(filtered_fp_df["filepath"]) - 1)
        og_fp_var = filtered_fp_df["filepath"][rint]
        og_fp.append(og_fp_var)

        # save cluster label
        clus_label.append(filtered_fp_df["cluster_label"][rint])

        # replace package name with random package name
        new_fp_var = []
        for item in og_fp_var:
            new_fp_var.append(item)
        rintdeux = randint(0, len(piplist_samples)-1)
        old_package_list.append(new_fp_var[7])
        new_fp_var[7] = piplist_samples[rintdeux]
        new_pacakge_list.append(new_fp_var[7])
        new_fp.append(new_fp_var)
    new_fp_df["original filepath"] = og_fp
    new_fp_df["new filepath"] = new_fp
    new_fp_df["old py package"] = old_package_list
    new_fp_df["new py package"] = new_pacakge_list
    new_fp_df["cluster labels"] = clus_label
    # print("\nnew:")
    # print(new_filepaths)

    return new_fp_df


def run_model(model_name="FastText", pretrained_model=None, model_epochs=200, filepaths=None, new_filepaths=None,
              encoded_fps=None, dimres_method=None, model_dim_size=300, db_params=None, drawplots=0):
    """
    db_params = [eps, min_samples]
    model_name = "FastText" or "Doc2Vec"
    """
    if filepaths is not None:
        pass
    else:
        # Import filepaths
        datadigits, filepaths_raw = import_ima()
        print("Number of filepaths before filtering:{}".format(len(filepaths_raw)))

        # Remove all filepaths involved in TP quotes
        filepaths = process_filepaths(filepaths_raw)
        print("Number of filepaths after filtering:{}".format(len(filepaths)))

    model_name = model_name.lower()
    if pretrained_model == None:
        # Build the model and encode the filepaths
        if new_filepaths is not None:
            filepaths.append(new_filepaths)
        if model_name == "fasttext":
            model, encoded_fps = do_fasttext(filepaths, dim_size=model_dim_size,
                                             epochs=model_epochs)
        elif model_name == "doc2vec":
            model, encoded_fps = do_doc2vec(filepaths, dim_size=model_dim_size,
                                            epochs=model_epochs)
        elif encoded_fps == None:
            print("Please provide a model name ('FastText' or 'Doc2Vec') for evaluation, or pre-encoded filepaths.")
            exit()
    elif new_filepaths is not None:
        model = pretrained_model
        for filepath in new_filepaths:
            filepaths.append(filepath)
            if model_name == "fasttext":
                # encode the filepath with the pretrained model
                # filepath = pretrained_model.wv[filepath]

                # make temp variable for calculations
                line_temp = [0] * model_dim_size
                fp_length = len(filepath)

                # for every folder in filepath
                for item in filepath:

                    j = 0
                    item_vec = pretrained_model.wv[item]

                    # put dimensions into new variable
                    for dim in item_vec:
                        # add dimension into right space
                        line_temp[j] = line_temp[j] + dim

                        # iterate index for every dimension
                        j = j + 1

                k = 0
                for dim in line_temp:  # average the dimensions to obtain sentence vector
                    line_temp[k] = line_temp[k] / fp_length
                    k = k + 1

                encoded_fps.append(line_temp)
            if model_name == "doc2vec":
                encoded_fps.append(model.infer_vector(filepath))

    else:
        print("Pretrained model supplied but no new filepaths supplied.")
        exit()

    if dimres_method is not None:
        dimres_method.lower()
        if dimres_method == "pca":
            data_pca = func_pca(datanp=encoded_fps, feat_cols=range(1, model_dim_size + 1),
                                drawplot=drawplots, n_components=3)
            data_clustered = func_dbscan(data=data_pca[:, [-3, -2, -1]],
                                             eps=db_params[0],
                                             min_samples=db_params[1],
                                             drawplot=drawplots)
            feat_cols = []
            feat_cols.append("filepaths")
            # explore_cluster(datanp=filepaths,
            #                 dataclust=data_clustered,
            #                 cols=feat_cols,
            #                 filepaths=filepaths
            #                 )
            pass

        elif dimres_method == "t-sne":
            data_tsne = func_tsne(datanp=encoded_fps, feat_cols=range(1, model_dim_size + 1), drawplot=drawplots,
                                n_components=3)
            data_clustered = func_dbscan(data=data_tsne[:, [-3, -2, -1]],
                                             eps=db_params[0],
                                             min_samples=db_params[1],
                                             drawplot=drawplots)
            feat_cols = []
            feat_cols.append("filepaths")
            # explore_cluster(datanp=filepaths,
            #                 dataclust=data_clustered,
            #                 cols=feat_cols,
            #                 filepaths=filepaths
            #                 )
            pass

        elif dimres_method == "umap":
            data_umap = func_umap(datanp=encoded_fps, feat_cols=range(1, model_dim_size + 1), drawplot=drawplots,
                                n_components=3)
            data_clustered = func_dbscan(data=data_umap[:, [-3, -2, -1]],
                                             eps=db_params[0],
                                             min_samples=db_params[1],
                                             drawplot=drawplots)
            feat_cols = []
            feat_cols.append("filepaths")
            # explore_cluster(datanp=filepaths,
            #                 dataclust=data_clustered,
            #                 cols=feat_cols,
            #                 filepaths=filepaths
            #                 )
            pass

        else:
            print("Dimentionality reduction method {} unrecognized, please provide"
                  "'PCA', 't-SNE', or 'UMAP'.".format(dimres_method))
            exit()

    return data_clustered, model, filepaths, encoded_fps

def calc_loss(fp_df, new_clustered_data, new_filepaths):
    sample_size = len(fp_df["new filepath"])
    new_fp_df = pd.DataFrame(data=None)
    new_fp_df["filepath"] = new_filepaths
    new_fp_df["cluster label"] = new_clustered_data[:, -1]
    success_count = 0
    for filepath in fp_df["new filepath"]:
        poison_cluster_label = new_fp_df["cluster label"][new_fp_df["filepath"].tolist().index(filepath)]

        original_filepath = fp_df["original filepath"][fp_df['new filepath'].tolist().index(filepath)]
        original_cluster_label = new_fp_df["cluster label"][new_fp_df['filepath'].tolist().index(original_filepath)]

        if poison_cluster_label == original_cluster_label:
            success_count = success_count + 1

    loss_rate = (1 - success_count/sample_size) * 100

    return loss_rate


def eval_model(model_name, dimres_method, model_epochs, model_dim_size, db_params, drawplots=1, model=None):
    np.random.seed(123)

    time_start = time.time()
    print("\nFinding original clusters...\n")
    if model is None:
        model_flag = 0
    else:
        pass
    clustered_data, model, original_filepaths, original_encoded_filepaths = run_model(model_name=model_name, model_epochs=model_epochs, dimres_method=dimres_method,
                                                          model_dim_size=model_dim_size, db_params=db_params, drawplots=drawplots, pretrained_model=model)
    if model_flag == 0:
        print("\nTime taken to build new model and find clusters: {} seconds".format(time.time()-time_start))
    else:
        print("\nTime taken to and find clusters from existing model: {} seconds".format(time.time()-time_start))

    print("\nPoisoning data...")
    new_fp_df = generate_poisoned_filepaths(filepaths=original_filepaths, clustered_data=clustered_data)

    # for filepath in new_fp_df["new filepath"]:
    #     original_filepaths.append(filepath)
    #
    # new_filepaths = original_filepaths

    time_start_2 = time.time()
    print("\nFinding new clusters...\n")
    new_clustered_data, model, combined_filepaths, new_encoded_filepaths = run_model(model_name=model_name, model_epochs=model_epochs, pretrained_model=model,
                                                         model_dim_size=model_dim_size, filepaths=original_filepaths, encoded_fps=original_encoded_filepaths,
                                                         new_filepaths=new_fp_df["new filepath"].tolist(), dimres_method=dimres_method,
                                                         db_params=db_params, drawplots=drawplots)
    print("\nTime taken to find new filepath clusters using existing model: {} seconds".format(time.time() - time_start_2))

    print("\nCalculating loss rate...")
    loss_rate = calc_loss(new_fp_df, new_clustered_data, combined_filepaths)

    print(loss_rate, "%")

    print("\nTotal Time Elapsed with this evaluation: {} minutes".format((time.time() - time_start)/60))

    return loss_rate, new_clustered_data, new_fp_df, model
