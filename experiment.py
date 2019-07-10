import pdb
import string
import collections
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_vocab(path):
    with open(path, encoding="utf8") as file:
        vocab = file.readlines()

    word2id = collections.OrderedDict()

    for i, word in enumerate(vocab):
        word = word.strip() # remove \n
        word2id[word] = i
    # -------------- Special Tokens -------------- #
    # <unk> is defined in the vocab list
    # -------------------------------------------- #
    return word2id

def read_train_data(path):
    with open(path) as file:
        lines = file.readlines()
        header = lines[0]
        body = lines[1:]

    phrases = []
    sentiments = []

    for line in body:
        line = line.strip().split('\t')
        phrase = line[2]
        sentiment = line[3]
        phrases.append(phrase)
        sentiments.append(sentiment)

    return phrases, sentiments

def process_phrases(inputs, word2id, max_sentence_length=20):
    # inputs = a list of sentences
    # outputs = a list of lists, each containing a sequence numbers (mapped words)
    outputs = []
    omitted_tokens = []

    for sentence in inputs:
        output = [0] * max_sentence_length
        sentence = sentence.lower()
        for k, word in enumerate(sentence.split()):

            if k == max_sentence_length:
                break

            if word in string.punctuation:
                pass
            elif word in word2id:
                output[k] = word2id[word] + 1
            else:
                output[k] = word2id['<unk>'] + 1

        outputs.append(output)

    return outputs

def process_sentiments(inputs):
    outputs = []
    for sentiment in inputs:
        outputs.append(float(sentiment) - 2.0)
    return outputs

def build_model(config):
    num_words = config['num_words']
    embedding_size = config['embedding_size']
    lstm_size = config['lstm_size']
    max_sentence_length = config['max_sentence_length']

    inputs = keras.Input(shape=(None,), name='input') # variable-length sequence of integers (word ids)

    # embed each word in the input into 200-dimensional vector
    x = layers.Embedding(num_words, embedding_size)(inputs)

    # LSTM
    x = layers.LSTM(lstm_size)(x)

    # Regression - value in between -1 to +1
    x = layers.Dense(1, activation='tanh')(x)
    predictions = 2.0 * x

    model = keras.Model(inputs=inputs, outputs=predictions)

    return model

def main():
    # paths
    path_vocab = 'google-10000-english.txt'
    path_train = '../sentiment-analysis-on-movie-reviews/train.tsv'

    # Pre-processing & Prepare data
    max_sentence_length = 25
    word2id = load_vocab(path_vocab)
    phrases, sentiments = read_train_data(path_train)
    phrases = process_phrases(phrases, word2id, max_sentence_length)
    sentiments = process_sentiments(sentiments)


    # Build our model
    config = {}
    config['num_words'] = len(word2id) + 1
    config['embedding_size'] = 200
    config['lstm_size'] = 128
    config['max_sentence_length'] = max_sentence_length

    model = build_model(config)
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    # Train the model
    phrases = np.array(phrases, dtype=np.int32)
    sentiments = np.array(sentiments, dtype=np.float32)
    model.fit(phrases, sentiments,
              batch_size=64,
              epochs=3)



    print("experiment done!")

main()
