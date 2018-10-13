import numpy as np
import pickle
import sys
from topic_modeling import convert_to_bow

with open('trained_lda_gmat.obj', 'rb') as handle:
	lda = pickle.load(handle)


train_essays = np.load('/vol/bitbucket/gv2117/train_essays_45.npy')
eval_essays = np.load('/vol/bitbucket/gv2117/eval_essays_45.npy')

train_bow = convert_to_bow(train_essays)
eval_bow = convert_to_bow(eval_essays)

trained_topics = []
eval_topics = []

for i in range(len(train_bow)):
	trained_topics.append(lda[train_bow[i]])
trained_topics = np.asarray(trained_topics)

for i in range(len(eval_bow)):
        eval_topics.append(lda[eval_bow[i]])
eval_topics = np.asarray(eval_topics)

np.save('trained_topics_gmat45.npy', trained_topics)
np.save('eval_topics_gmat45.npy', eval_topics)
