import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

pd.options.display.max_colwidth = 200
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
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

normalize_corpus = np.vectorize(normalize_document)

from nltk.corpus import gutenberg
from string import punctuation

bible = gutenberg.sents('bible-kjv.txt')
remove_terms = punctuation + '0123456789'

norm_bible = [[word.lower() for word in sent if word not in remove_terms] for sent in bible]
norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
norm_bible = filter(None, normalize_corpus(norm_bible))
norm_bible = [tok_sent for tok_sent in norm_bible if len(tok_sent.split()) > 2]

print('Total lines:', len(bible))
print('\nSample line:', bible[10])
print('\nProcessed line:', norm_bible[10])


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)
word2id = tokenizer.word_index

# build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in norm_bible]

vocab_size = len(word2id)
embed_size = 100
window_size = 2 # context window size

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[822:832])

from embed_functions import *
print(word2id.items())
# model = do_word2vec()


def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1

            context_words.append([words[i]
                                  for i in range(start, end)
                                  if 0 <= i < sentence_length
                                  and i != index])
            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)


# Test this out for some samples
# i = 0
# for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
#     if 0 not in x[0]:
#         print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
#
#         if i == 10:
#             break
#         i += 1


# import keras.backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, Lambda
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
# # tf.test.is_gpu_available() # True/False
# # Or only check for gpu's with cuda support
# # tf.test.is_gpu_available(cuda_only=True)
#
# # build CBOW architecture
# cbow = Sequential()
# cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
# cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
# cbow.add(Dense(vocab_size, activation='softmax'))
# cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#
# # view model summary
# print(cbow.summary())
#
# # visualize model structure
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False,
#                  rankdir='TB').create(prog='dot', format='svg'))
#
# print("Starting training...")
# for epoch in range(1, 2):
#     loss = 0.
#     i = 0
#     for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
#         i += 1
#         loss += cbow.train_on_batch(x, y)
#         if i % 100000 == 0:
#             print('Processed {} (context, word) pairs'.format(i))
#
#     print('Epoch:', epoch, '\tLoss:', loss)
#     print()
#
# # serialize model to JSON
# cbow_json = cbow.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(cbow_json)
# # serialize weights to hdf5
# cbow.save_weights("cbow_weights.h5")
# print("Saved model weights to disk")