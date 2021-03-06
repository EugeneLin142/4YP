import numpy as np
import pandas as pd
from random import seed
from random import randint
from random import shuffle
from hyperopt_eval_functions import *
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import time
import csv
import pickle
import nltk
import time
from os import listdir
from os.path import isfile, join
import random
# Loss is defined as the number of randomised fake 'suspicious' quotes we fail to detect
# we need a function to create a bunch of fake quotes
# another function to add the quotes to the real dataset, adding a new 'label' column to make them distinct
# another function to determine which quotes are fake
# and finally one to generate loss value

def generate_fake_quotes(orig_df):
    filepaths = orig_df["filepath_tokens"].copy().to_list()

    # create number of samples equal to 10% of original sample size
    number_of_samples = int(len(filepaths)/10)
    seed(123)

    # read in star wars holiday script for dictionary
    wordlist = generate_dictionary()

    # create enough randomly positioned words from the wordlist to replace tokens with
    word_int_list = []
    for word_int in range(0, len(filepaths)):
        word_int_list.append(randint(0, len(wordlist)))
    shuffle(word_int_list)

    # choose samples to alter to create fake data
    randint_list = []
    for rint in range(0, len(filepaths)):
        randint_list.append(rint)
    shuffle(randint_list)

    pre_samples = []
    pre_sample_index_list = []
    for sample_chosen in range(0, number_of_samples):
        pop = randint_list.pop()
        pre_samples.append(filepaths[pop].copy())
        pre_sample_index_list.append(pop)


    # we want latter tokens in each sample to be more likely to be altered
    # also, sometimes more than one.

    for sample_index in range(0, len(pre_samples)):
        sample = pre_samples[sample_index]
        # randomly determine how many tokens to alter by log2 so 1 is most likely, 2 half as likely etc.
        sample_length = len(sample)
        position = int(np.log2(randint(1, 2 ** sample_length-1)))    # for alter position
        alter_no = sample_length - position                        # for alter number
        poslist = []                                          # to track alter position history

        for change in range(0, alter_no):
            pop = word_int_list.pop()
            pre_samples[sample_index][position] = wordlist[pop]
            poslist.append(position)
            word_int_list.append(pop)
            # choose next, unique position
            next_pos = position
            if len(poslist) == sample_length:
                pass
            else:
                while next_pos in poslist:
                    next_pos = int(np.log2(randint(1, 2 ** sample_length-1)))
            position = next_pos


    # create dataframe encompassing pre_sample filepaths and parent/process names

    poisonous_df = pd.DataFrame(data=None)

    poison_pnames = []
    poison_ppnames = []
    fakeflags = []
    realflags = []
    for index in pre_sample_index_list:
        poison_pnames.append(orig_df["process_name"][index])
        poison_ppnames.append(orig_df["parent_process_name"][index])
        fakeflags.append(1)
    poisonous_df["process_name"] = poison_pnames
    poisonous_df["parent_process_name"] = poison_ppnames
    poisonous_df["filepath_tokens"] = pre_samples
    poisonous_df["fake"] = fakeflags

    # and now something to append those to the real samples and add flags for real/fake
    # let's use 0 for real and 1 for fake
    for index in range(0, len(orig_df["process_name"])):
        realflags.append(0)
    orig_df["fake"] = realflags
    all_df = orig_df.append(poisonous_df, ignore_index=True)

    return all_df


def generate_dictionary():
    seed(123)
    filename = './data/star_wars_holiday.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    # split into words by white space
    words = text.split()
    # convert to lower case
    words = [word.lower() for word in words]

    punctuation = ["?", "!", "/", ":", "*", "|", "<", ">"]  # list of illegal filename characters

    wordlist = []
    for word in words:
        wordnew = ""
        for char in word:
            if char in punctuation:
                pass
            else:
                wordnew = wordnew + char
        wordlist.append(wordnew)


    # filter stopwords out of document
    stop_words = nltk.corpus.stopwords.words('english')
    wordlist = [token for token in wordlist if token not in stop_words]

    return wordlist


def run_val_model(quotes, model_name="FastText", pretrained_model=None, model_epochs=200, new_filepaths=None,
                  encoded_fps=None, dimres_method=None, model_dim_size=300, db_params=None, drawplots=0):
    """
    db_params = [eps, min_samples]
    model_name = "FastText" or "Doc2Vec"
    """
    filepaths = quotes["filepath_tokens"]
    model_name = model_name.lower()
    if pretrained_model == None:
        # Build the model and encode the filepaths
        if new_filepaths is not None:    # Ignore. Leftover from Hyperopt Eval equivalent func.
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
            data_pca, var_rat = func_pca(datanp=encoded_fps, feat_cols=range(1, model_dim_size + 1),
                                drawplot=drawplots, n_components=3)
            data_clustered = func_hdbscan(data=data_pca[:, [-3, -2, -1]],
                                         min_cluster_size=db_params[1], min_samples=db_params[0],
                                         #eps=(var_rat[0] * db_params[0]), min_samples=db_params[1],
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
            print("Dimensionality reduction method {} unrecognized, please provide"
                  "'PCA', 't-SNE', or 'UMAP'.".format(dimres_method))
            exit()

    return data_clustered #, model, filepaths, encoded_fps

def calc_val_loss(quotes, clustered_data):
    quotes["cluster"] = clustered_data[:, -1]
    loss_count = 0
    for index, row in quotes.iterrows():
        if row["fake"] == 1:
            if row["cluster"] == -1:
                pass
            else:
                loss_count += 1

    loss_percentage = loss_count/len(quotes["cluster"]) * 100

    return loss_percentage

def import_ima_window(windowsize=3):
    data = None
    filenamelist = []
    filelist = [f for f in listdir('./ima/archive/') if isfile(join('./ima/archive/', f))]

    filename = random.choice(filelist[0:-3])

    for i in range(0, windowsize):
        filenamelist.append(filename)
        filename = filelist[filelist.index(filename)+1]

    for filename in filenamelist:
        filename = './ima/archive/' + filename

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


    if data is None:
        print("No IMA measurements found, please put .dat or .csv files into ./ima/")
    else:
        print("{} day window from ./ima/ imported successfully.".format(len(filenamelist)))

    return data, filepaths, filenamelist


def validation(params):
    time_start = time.time()
    # Import filepaths
    datacode, filepaths_raw, filenamelist = import_ima_window(windowsize=1)
    print("Number of quotes before filtering:{}".format(len(filepaths_raw)))

    # Remove all filepaths involved in TP quotes
    data, filepaths = process_filepaths(datacode, filepaths_raw, data_flag=1)
    print("Number of quotes after removing ima processes:{}".format(len(filepaths)))

    # convert known ppids/pids to names
    data = pid_to_pname(data)

    # remove unknown associated filepaths
    data_df_named = trim_data(data, filepaths)

    quotes = generate_fake_quotes(data_df_named)

    # Choose next hyperparameters from hyperopt eval results

    clustered_data = run_val_model(model_name="Doc2Vec", model_epochs=int(params['model_epochs']), dimres_method="pca", quotes=quotes,
                                   model_dim_size=int(params['model_dim_size']), db_params=[int(params['min_cluster_size']), int(params['min_samples'])], drawplots=0)

    loss = calc_val_loss(quotes, clustered_data)
    print("Loss Rate:{} %".format(loss))
    run_time = time.time() - time_start

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, run_time])

    global valid_trials
    pickle.dump(valid_trials, open("valid_trials.p", "wb"))

    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'files_chosen': filenamelist, 'status': STATUS_OK}


space = {
    'model_epochs': 100,
    'model_dim_size': 80, #hp.quniform('model_dim_size', 60, 100, 5),
    'min_cluster_size': 3,
    'min_samples': 3
}

valid_trials = Trials()
# use this line instead if you're continuing
# tpe_trials = pickle.load(open("tpe_trials.p", "rb"))

ITERATION = 0

out_file = "valid_trials.csv"
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()


# Run evals with the tpe algorithm
tpe_best = fmin(fn=validation, space=space,
                algo=tpe.suggest, trials=valid_trials,
                max_evals=100, rstate=np.random.RandomState(50))

print(tpe_best)

