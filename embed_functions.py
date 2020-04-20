import pandas as pd
import numpy as np
import re
import nltk
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from multiprocessing import cpu_count
import gensim
import gensim.downloader as api

from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import os

def ascii_sum_embed(list_of_words):
    # Converts process/parent process name pairs into numbers by adding the ascii
    # decimal values of each character in each name together.
    list_of_nums = []
    for word in list_of_words:
        word_total = int(0)
        if word.isdigit() is True:
            # There is a number of pid/ppid pairs where one was identified with a known
            # name and the other not, for these we convert the known one to a number,
            # and simply carry over the other
            word_total = word
        else:
            for char in word:
                word_total = word_total + ord(char)
        list_of_nums.append(word_total)

    return list_of_nums

def normalize_document(doc):
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

def vectorizer(path, m):
    vec = []
    numw = 0
    for dir in path:
        try:
            if numw ==0:
                vec = m[dir]
            else:
                vec = np.add(vec, m[dir])
            numw+=1
        except:
            pass

    return np.asarray(vec) / numw

def elbow_kmeans(data, maxrange):
    time_start = time.time()
    print("Trying elbow method with up to ", maxrange, " clusters...")
    wcss = []
    for i in range(1,maxrange):
        kmeans = KMeans(n_clusters=i,
                        init='k-means++',
                        random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    print("Done! Time elapsed: {} seconds".format(time.time() - time_start))
    plt.plot(range(1,maxrange), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def KMeans_cluster(data, filepaths):
    n_clusters = 5
    clf = KMeans(n_clusters=n_clusters,
                 max_iter=100,
                 init='k-means++',
                 n_init=1)
    labels = clf.fit_predict(data)
    for index, path in enumerate(filepaths):
        print(str(labels[index]) + ':' + str(path))

def do_word2vec(data, epochs):
    print("encoding using Word2Vec...")
    model = Word2Vec(size=100, min_count=1, sg=0, workers=cpu_count(), window=2)
    # sg = 0 for CBOW, 1 for skipgram

    model.build_vocab(data, progress_per=10000)

    learning_rate = 0.025
    step_size = (learning_rate - 0.001) / epochs

    for i in range(epochs):
        end_lr = learning_rate - step_size
        trained_word_count, raw_word_count = model.train(data, compute_loss=True,
                                                         start_alpha=learning_rate,
                                                         end_alpha=learning_rate,
                                                         total_examples=model.corpus_count,
                                                         epochs=1)
        loss = model.get_latest_training_loss()
        print("iter={0}, loss={1}, learning_rate={2}".format(i, loss, learning_rate))
        learning_rate *= 0.6

    model.init_sims(replace=True) # if no longer training model, make it memory-efficient

    print("Loss: {}".format(model.get_latest_training_loss()))

    return model

def do_doc2vec(data, epochs, dim_size):
    # Create the tagged document needed for Doc2Vec
    def create_tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    train_data = list(create_tagged_document(data))

    # Init the Doc2Vec model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=dim_size, min_count=1,
                                          workers=cpu_count(), window=2,
                                          epochs=epochs
                                          )

    # Build the Vocabulary
    model.build_vocab(train_data)

    # Train the Doc2Vec model
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

    # learning_rate = 0.025
    # step_size = (learning_rate - 0.001) / epochs
    #
    # for i in range(0, epochs):  ### apparently this approach is highly outdated..
    #     end_lr = learning_rate - step_size
    #     model.train(train_data,
    #                 total_examples=model.corpus_count,
    #                 epochs=1)
    #                 # start_alpha=learning_rate,
    #                 # end_alpha=learning_rate,
    #     loss = model.get_training_loss()
    #     print("iter={0}, loss={1}, learning_rate={2}".format(i, loss, learning_rate))
    #     learning_rate *= 0.6

    # Compile paragraph vectors learned from the training data
    a = []
    for i in range(0, len(data)):
        a.append(model.docvecs[i])

    a = np.asarray(a, dtype=np.float32)

    return model, a


import Levenshtein as lev


def process2lev(data): #likely useless really
    baseline = "/"
    lev_distance = []
    for process in data:
        process_dist = lev.distance(baseline, process)
        lev_distance.append(process_dist)
    return lev_distance


from gensim.models import FastText

def do_fasttext(data, dim_size=300, epochs=200):
    # Init Model
    print("building {}".format(dim_size), "dimensional FastText model for {} epochs...".format(epochs))
    model = FastText(size=dim_size, window=3, min_count=1)

    # Build Vocab
    model.build_vocab(sentences=data)

    # Train Model
    model.train(sentences=data, total_examples=len(data), epochs=epochs)
    print("FastText model trained!")

    # Compile paragraph vectors learned from the training data
    print("calculating sentence embedding via BOW...")
    sentence_vec = []
    for i in range(0, len(data)):       # for every filepath
        line = [0] * dim_size           # make temp variable for calculations
        fp_length = len(data[i])

        for item in data[i]:            # for every folder in filepath
            j = 0
            item_vec = model.wv[item]

            for dim in item_vec:        # put dimensions into new variable
                line[j] = line[j] + dim # add dimension into right space
                j = j + 1               # iterate index for every dimension

        k = 0
        for dim in line:        # average the dimensions to obtain sentence vector
            line[k] = line[k]/fp_length
            k = k + 1

        sentence_vec.append(line)
    print("sentence vectors embedded.")
    return model, sentence_vec

def infer_new_fasttext_vector(datum, FTmodel):
    dim_size = len(FTmodel.wv["usr"])
    # datum should, as always, be a list of strings, representing the filepath
    line = [0] * dim_size               # make temp variable for calculations
    fp_length = len(datum)
    for item in datum:                  # for every folder in filepath
        j = 0
        item_vec = FTmodel.wv[item]

        for dim in item_vec:            # put dimensions into new variable
            line[j] = line[j] + dim     # add dimension into right space
            j = j + 1                   # iterate index for every dimension

        k = 0
        for dim in line:                # average the dimensions to obtain sentence vector
            line[k] = line[k] / fp_length
            k = k + 1

    return(line)