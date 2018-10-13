import numpy as np
import sys

def get_essay_id_associations():
    size = data.shape[0]

    associations = {}
    unique_ids = set(map(lambda x: int(x) ,np.unique(ids)))

    for id in unique_ids:
        associations[id] = 0

    for i in range(size):
        associations[int(ids[i])] +=1 

    for key,value in associations.items():
        print ("id: "+str(key)+' - '+'number_of_essays: '+str(value)+'\n')

def select_specific(data, ids, scores):
    essays_31 = []
    scores_31 = []
    size = data.shape[0]
    for i in range(size):
        if int(ids[i]) == 31:
            essays_31.append(data[i])
            scores_31.append(int(scores[i]))

    essays_31 = np.asarray(essays_31)
    scores_31 = np.asarray(scores_31)            

    np.save('essays_31.npy', essays_31)
    np.save('scores_31.npy', scores_31)    

    return essays_31, scores_31

data = np.load('essays_final.npy')
ids = np.load('essay_ids_final.npy')
scores = np.load('scores_final.npy')
e, s = select_specific(data, ids, scores)

assoc = {}
for i in range(1,7):
    assoc[i] = 0

for i in range(e.shape[0]):
    assoc[int(s[i])] += 1  

for key,value in assoc.items():
    print(str(key)+' - ' +str(value) + '\n')      
