import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import os
import pickle

from termcolor import colored
from model import gcnn_fn
np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

e_i = np.load('/vol/bitbucket/gv2117/essays_viz.npy')
sh_i = np.load('/vol/bitbucket/gv2117/viz_scores.npy')
sl_i = np.load('/vol/bitbucket/gv2117/low_viz_scores.npy')
t_i = np.load('viz_topics.npy')
for i in range(e_i.shape[0]):
	e = np.reshape(e_i[i], (1,750))
	sh = np.reshape(sh_i[i], (1))
	sl = np.reshape(sl_i[i], (1))
	t = np.reshape(t_i[i], (1,5))
	results  = np.asarray(gcnn_fn(e, sh, sl, t))
	mask = 0
	for i in range(len(e[0])):
		if e[0][i] != 0:
			mask += 1
	grads = results[0][0][:] 
	grads = grads[:mask]
	essay_int = results[0][1]
	essay_int = essay_int[:mask]

	grads_norm = np.linalg.norm(grads, axis=1)
	grads_values = sorted(grads_norm, reverse=True)
	grads_norm = grads_norm.argsort()[:][::-1]

	with open('map_to_words31.pickle', 'rb') as handle:
		vocab_dic = pickle.load(handle)

	essay = e
	essay_set = set(essay[0])
	essay_appended = []
	gap_v = (grads_values[0] - grads_values[-1] )/ 4
	gap = len(essay_int)//4
	for i in essay[0]:
		if i!=0:
			if i in essay_int[grads_norm[:gap]]:
				essay_appended.append(colored(vocab_dic[i], 'red'))
			elif i in essay_int[grads_norm[gap:2*gap]]:
				essay_appended.append(colored(vocab_dic[i], 'cyan'))
			elif i in essay_int[grads_norm[2*gap:3*gap]]:
                                essay_appended.append(colored(vocab_dic[i], 'yellow'))
			else:
				essay_appended.append(colored(vocab_dic[i], 'green'))
	from functools import reduce
	essay_appended = reduce(lambda x1,x2:x1+' '+x2, essay_appended)
	print (essay_appended)
	print ('\n')
	j=0
