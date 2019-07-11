import pdb
import string
import collections
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers ---> not compatible with tf 1.5

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
        outputs.append(int(sentiment))
    return outputs

def build_model(config):
    num_words = config['num_words']
    embedding_size = config['embedding_size']
    lstm_size = config['lstm_size']
    max_sentence_length = config['max_sentence_length']

    inputs = keras.Input(shape=(None,), name='input') # variable-length sequence of integers (word ids)

    # embed each word in the input into 200-dimensional vector
    x = keras.layers.Embedding(num_words, embedding_size)(inputs)

    # LSTM
    x = keras.layers.LSTM(lstm_size)(x)

    # Regression - value in between -1 to +1
    x = keras.layers.Dense(64, activation='sigmoid')(x)
    outputs = keras.layers.Dense(5, activation='softmax')(x)


    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def main():
    # For using GPU
    if 'X_SGE_CUDA_DEVICE' in os.environ:
        print('running on the stack...')
        cuda_device = os.environ['X_SGE_CUDA_DEVICE']
        print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    else: # development only e.g. air202
        print('running locally...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '3' # choose the device (GPU) here


    # paths
    path_vocab = '/home/alta/BLTSpeaking/ged-pm574/summer2019/sentiment/google-10000-english.txt'
    path_train = '/home/alta/BLTSpeaking/ged-pm574/summer2019/sentiment-analysis-on-movie-reviews/train.tsv'

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

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    phrases = np.array(phrases, dtype=np.int32)
    sentiments = np.array(sentiments, dtype=np.int32)
    sentiments = tf.keras.utils.to_categorical(sentiments, num_classes=5)


    callbacks = [
        keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_acc',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=2e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2)
    ]

    model.fit(phrases, sentiments,
              batch_size=128,
              epochs=50,
              shuffle=True,
              validation_split=0.2,
              callbacks=callbacks)

    # Save the model
    model.save('/home/alta/BLTSpeaking/ged-pm574/summer2019/sentiment/models/lstm1-50-tf1.h5')



    print("experiment done!")

main()
