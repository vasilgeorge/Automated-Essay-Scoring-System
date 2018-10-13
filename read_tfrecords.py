import tensorflow as tf
import numpy as np
import pandas
import nltk
import sys
from hyperparameters import Parameters

def dataset_train_input_fn():
  filenames = ['/homes/gv2117/MSc_Project/Automatic-Essay-Scoring-System/GCNN_server/train_gmat.tfrecords']
  dataset = tf.data.TFRecordDataset(filenames)
  params = Parameters()
  params.get_params()
  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "essay": tf.FixedLenFeature((), tf.string),
        "score": tf.FixedLenFeature((), tf.string),
        "topic": tf.FixedLenFeature((), tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    essay = tf.decode_raw(parsed["essay"], tf.int64)
    score = tf.decode_raw(parsed["score"], tf.int64)
    topic = tf.decode_raw(parsed["topic"], tf.float64)
    return {'essay':essay, 'topic':topic} ,score

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(params.batch_size)
  dataset = dataset.repeat(10000)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels


def dataset_eval_input_fn():
  filenames = ['/homes/gv2117/MSc_Project/Automatic-Essay-Scoring-System/GCNN_server/eval_gmat.tfrecords']
  dataset = tf.data.TFRecordDataset(filenames)
  params = Parameters()
  params.get_params()
  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "essay": tf.FixedLenFeature((), tf.string),
        "score": tf.FixedLenFeature((), tf.string),
        "topic": tf.FixedLenFeature((), tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    essay = tf.decode_raw(parsed["essay"], tf.int64)
    score = tf.decode_raw(parsed["score"], tf.int64)
    topic = tf.decode_raw(parsed["topic"], tf.float64)
    return {'essay':essay, 'topic':topic },score

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(params.batch_size)
  dataset = dataset.repeat(1)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels


def dataset_test_input_fn():
  filenames = ['/homes/gv2117/MSc_Project/Automatic-Essay-Scoring-System/GCNN_server/random.tfrecords']
  dataset = tf.data.TFRecordDataset(filenames)
  params = Parameters()
  params.get_params()
  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "essay": tf.FixedLenFeature((), tf.string),
        "topic": tf.FixedLenFeature((), tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    essay = tf.decode_raw(parsed["essay"], tf.int64)
    topic = tf.decode_raw(parsed["topic"], tf.float64)
    return {'essay':essay, 'topic':topic }

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.batch(1)
  dataset = dataset.repeat(1)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features = iterator.get_next()
  return features
