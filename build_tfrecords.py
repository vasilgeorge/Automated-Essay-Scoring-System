import pandas as pd
import nltk
import tensorflow as tf
import numpy as np
import sys

def _int64_feature(value):
  return tf.train.Feature()


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_to_tfrecords():
    essays = np.load('/vol/bitbucket/gv2117/train_essays_96.npy')
    scores = np.load('/vol/bitbucket/gv2117/train_scores_96.npy')
    topics = np.load('trained_topics_gmat96.npy')
    num_examples = len(essays)


    filename = 'train_gmat_96.tfrecords'

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            essay = essays[index].tostring()
            score = scores[index].tostring()
            topic = topics[index][:,1].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                feature={
                'essay':_bytes_feature(essay),
                'score':_bytes_feature(score),
                'topic':_bytes_feature(topic)
                }))
            writer.write(example.SerializeToString())

def convert_eval_to_tfrecords():
    essays = np.load('/vol/bitbucket/gv2117/eval_essays_96.npy')
    scores = np.load('/vol/bitbucket/gv2117/eval_scores_96.npy')
    topics = np.load('eval_topics_gmat96.npy')

    num_examples = len(essays)


    filename = 'eval_gmat_96.tfrecords'

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            essay = essays[index].tostring()
            score = scores[index].tostring()
            topic = topics[index][:,1].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                feature={
                'essay':_bytes_feature(essay),
                'score':_bytes_feature(score),
                'topic':_bytes_feature(topic)
                }))
            writer.write(example.SerializeToString())

convert_to_tfrecords()
convert_eval_to_tfrecords()
