import numpy as np
from topic_modeling import get_topics_vector
from collections import Counter
import sys

def get_second_max(topic):
    max = 0
    max_i = 0
    for i in range(len(topic)):
        if topic[i,1] > max:
            max = topic[i,1]
            max_i = i
    return max_i

def get_count_dict(topics):
    l = []
    for i in range(len(topics)):
        l.append(get_second_max(topics[i]))

    d = {}
    s = set(l)
    for val in s:
        d[val] = 0
    for val in l:
        d[val] += 1
    return d

def test_random_essays(essays):
    top_v = get_topics_vector(essays)
    print (top_v)

def third(topics, scores):
    c=[]
    b=0
    for i in range(len(topics)):
        if topics[i, 2, 1] < 0.01:
            b+=1
    print (b)


print ("getting the dict..")
a = np.load('trained_topics_1.npy')
print (get_count_dict(a))
print ('testing randomly generated essays..')
e1 = np.random.randint(low=1, high=3500, size=(4,1204))
test_random_essays(e1)
scores = np.load('train_data_y.npy')
third(a, scores)
