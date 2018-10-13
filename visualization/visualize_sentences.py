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
	s_h = np.reshape(sl_i[i], (1))
	s_l = np.reshape(sh_i[i], (1))
	t = np.reshape(t_i[i], (1,5))
	results  = np.asarray(gcnn_fn(e, s_h, s_l, t))	
	mask = 0
	for i in range(len(e[0])):
		if e[0][i] != 0:
			mask += 1

	grads = results[0][0][:] #(750,200)
	grads = grads[:mask]

	essay_int = results[0][1]
	essay_int = essay_int[:mask]
	sentences = []
	grads_norm = np.linalg.norm(grads, axis=1)

	sentence_grads = []
	s_grad=0
	length = 0
	sentence=[]
	lengths = []
	for i in range(len(essay_int)):
		if essay_int[i] == 2:
			sentence_grads.append(s_grad)
			lengths.append(length)
			sentences.append(sentence)
			s_grad=0
			length = 0
			sentence=[]
		else:
			s_grad += grads_norm[i]
			sentence.append(essay_int[i])
			length += 1

	sentence_grads = [sentence_grads[i]/lengths[i] for i in range(len(sentence_grads))]
	sentence_grads = np.asarray(sentence_grads)

	sorted_sent = sorted(sentence_grads, reverse=True)
	sorted_indexes = sentence_grads.argsort()[:][::-1]
	with open('map_to_words31.pickle', 'rb') as handle:
		vocab_dic = pickle.load(handle)
	s=[]
	gap = len(sentences)//4
	print (gap)
	for j in range(len(sentences)):
		
		for k in range(gap):
			if (sentences[j] == sentences[sorted_indexes[k]]):
				for i in sentences[j]:
					if i != 0:
						s.append(colored(vocab_dic[i], 'red'))
		for k in range(gap, 2*gap):
			if sentences[j] == sentences[sorted_indexes[k]]:
				for i in sentences[j]:
					if i !=0:
						s.append(colored(vocab_dic[i], 'cyan'))
		for k in range(2*gap, 3*gap):
			if sentences[j] == sentences[sorted_indexes[k]]:
				for i in sentences[j]:
					if i !=0:
						s.append(colored(vocab_dic[i], 'yellow'))
		for k in range(3*gap, 4*gap):
                        if sentences[j] == sentences[sorted_indexes[k]]:
                                for i in sentences[j]:
                                        if i !=0:
                                                s.append(colored(vocab_dic[i], 'green'))

		s.append('.')
	from functools import reduce
	ss = reduce(lambda x1,x2:x1+' '+x2, s)
	print  (ss)

	def get_predictions(preds):
		with tf.variable_scope('probabilities'):
			probs = preds
			prob_sum = tf.Variable(tf.zeros(shape=(preds.shape[0], preds.shape[1]), dtype=tf.float32), trainable=False, name='probs', collections=[tf.GraphKeys.LOCAL_VARIABLES])
			
			update_op = tf.assign_add(prob_sum, probs)

			return tf.convert_to_tensor(prob_sum), update_op
