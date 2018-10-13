import sys
import numpy as np
import pickle
from topic_modeling import convert_to_bow

random_essays = np.random.randint(0, 7200, size=(15,750))
np.save('random_essays.npy', random_essays)
random_topics = convert_to_bow(random_essays)

with open('trained_lda_gmat.obj', 'rb') as handle:
	lda = pickle.load(handle)

topics = []
for i in range(len(random_topics)):
	topics.append(lda[random_topics[i]])
topics = np.asarray(topics)
np.save('random_topics.npy', topics)
