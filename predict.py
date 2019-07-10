import pdb
import string
import collections
import os
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

def read_test_data(path):
    with open(path) as file:
        lines = file.readlines()
        header = lines[0]
        body = lines[1:]

    phrases = []
    ids = []

    for line in body:
        line = line.strip().split('\t')
        phrase = line[-1]
        ids.append(int(line[0]))
        phrases.append(phrase)

    return phrases, ids

def process_phrases(inputs, word2id, max_sentence_length=25):
    outputs = []

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


def make_predictions(model, inputs):
    scores = []
    predicts = model.predict(inputs, verbose=True)
    for x in predicts:
        score = x[0]*0 + x[1]*1 + x[2]*2 + x[3]*3 + x[4]*4
        scores.append(score)
    return scores

def main():
    # Recreate the exact same model purely from the file
    trained_model = keras.models.load_model('models/lstm1-10.h5')

    word2id = load_vocab('google-10000-english.txt')
    phrases, ids = read_test_data('../sentiment-analysis-on-movie-reviews/test.tsv')
    phrases = process_phrases(phrases, word2id, 25)
    phrases = np.array(phrases, dtype=np.int32)


    scores = make_predictions(trained_model, phrases)

    pdb.set_trace()
    print(len(scores))
    print(len(ids))

    with open("predictions.csv", 'w') as file:
        for id, score in zip(ids, scores):
            grade = int(round(score))
            file.write("{},{}\n".format(id, grade))

    print("prediction done!")

main()
