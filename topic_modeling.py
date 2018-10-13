import sys
import numpy as np
import pandas as pd
import nltk
import gensim
import pickle



""" This function should train the LDA model on the whole training dataset of the essays """
def train_lda(path):
    essays = np.load(path)
    bagged_essays = convert_to_bow(essays)

    #Remove zero because it is just used for padding and does not contribute anything
    for i in range(len(bagged_essays)):
        bagged_essays[i].remove(bagged_essays[i][0])

    lda = gensim.models.LdaModel(bagged_essays, num_topics=5, minimum_probability=0, dtype=np.float32)

    with open('trained_lda_gmat.obj', 'wb') as l:
        pickle.dump(lda, l)


def get_topics_vector(essays):
    """ Should return an array of size (batch_size, number_of_topics) """

    with open('trained_lda_gmat.obj', 'rb') as l:
        lda = pickle.load(l)

    topics = []
    bagged_essays = convert_to_bow(essays)
    for i in range(len(bagged_essays)):
        topics.append(lda[bagged_essays[i]])

    return topics

""" Converts each essay to a bag of words and returns it as a list """
def convert_to_bow(phrase):
    s_phrase = []
    for i in range(phrase.shape[0]):
        s_phrase.append(set(phrase[i]))

    bow = []
    semi_bow = []
    for i in range(len(phrase)):
        for num in s_phrase[i]:
            no = (phrase[i].tolist()).count(num)
            tup = (num, no)
            semi_bow.append(tup)
        bow.append(semi_bow)
        semi_bow = []

    return bow
