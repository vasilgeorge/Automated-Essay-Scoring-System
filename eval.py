import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import sys
import os

from model import gcnn_fn
from read_tfrecords import dataset_train_input_fn, dataset_eval_input_fn

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("model_dir", "", "Directory where checkpoint is stored")


def main(_):
    assert FLAGS.model_dir, "--model_dir is required"
    
    tf.logging.set_verbosity(tf.logging.INFO)

    essay_classifier = tf.estimator.Estimator(
                                             model_fn = gcnn_fn,
                                             model_dir = "FLAGS.model_dir"
                                             )

    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


    print ('Starting evaluation..')
    eval_res=essay_classifier.evaluate(
                             input_fn = lambda:dataset_eval_input_fn()
    )
    print (eval_res)

if __name__=="__main__":
    tf.app.run()
