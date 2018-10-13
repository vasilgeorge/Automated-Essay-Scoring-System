import sys
import tensorflow as tf
import numpy as  np

from hyperparameters import Parameters

def gcnn_fn(features, labels, mode):

    print("Fetching parameters file...")
    params = Parameters()
    params.get_params()

    essays = features['essay']
    if mode != tf.estimator.ModeKeys.PREDICT:
    	essays = tf.reshape(essays, (params.batch_size, params.average_length))
    	scores = tf.reshape(labels, [-1])
    	topics_latent_vector = tf.cast(features['topic'], tf.float32)
    	topics_latent_vector = tf.reshape(topics_latent_vector, (params.batch_size, params.num_topics), name='topics_reshape')
    else:
        essays = tf.reshape(essays, (1, params.average_length))
        topics_latent_vector = tf.cast(features['topic'], tf.float32)
        topics_latent_vector = tf.reshape(topics_latent_vector, (1, params.num_topics), name='topics_reshape')
        

    with tf.variable_scope('word_embeddings') as embeddings_scope:
        print("Getting the word embeddings map...")
        #Create the word embeddings map, that maps words(ints) to fixed size float vectors. Shared across all the layers.
        embeddings_map = tf.get_variable(
                                        name = 'embeddings_map',
                                        shape = [params.vocab_size, params.embeddings_dimensions],
                                        dtype = tf.float32,
                                        #initializer = pretrained_weights,
                                        trainable = True
                                        )

        print("Making the words-word embedding vectors correspondences")
        word_embeddings = tf.nn.embedding_lookup(
                                                params = embeddings_map,
                                                ids = essays,
                                                name = 'word_embeddings'
                                                )
        print("Reshaping word embeddings...")
        #At this point we have to add the convolutional layers.
        #Convolutional layers need a 4-dimensional input and that's why we have to reshape our input which is 3-D. The output shape of the following
        # function is [batch_size, max_length, embeddings_dim, 1]
        word_embeddings = tf.reshape(word_embeddings, shape = (word_embeddings.shape[0], word_embeddings.shape[1], word_embeddings.shape[2], 1))

        input_ = word_embeddings
        res_input = word_embeddings

    print("Going into the convolutional-gating layers...")
    for i in range(params.num_conv_layers):
            with tf.variable_scope('layer_%d'%i) as conv_scope:

                W = tf.get_variable(name="linear_W", shape=[params.filter_height, params.filter_width, input_.shape[-1], params.filter_size],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1), trainable=True)
                b = tf.get_variable(name="linear_b", shape=[params.filter_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=True)
                V = tf.get_variable(name="gated_V", shape=[params.filter_height, params.filter_width, input_.shape[-1], params.filter_size],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1), trainable=True)
                c = tf.get_variable(name="gated_c", shape=[params.filter_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=True)

                linear_layer = tf.add(tf.nn.conv2d(input_, W, strides=[1,1,1,1], padding='SAME', name='linear'), b)
                gated_layer = tf.add(tf.nn.conv2d(input_, V, strides=[1,1,1,1], padding='SAME', name='gated'), c)

                h = tf.multiply(linear_layer, tf.sigmoid(gated_layer))
                input_ = h
                if i % params.block_size == 0:
                    h += res_input
                    res_input = h
                    input_ = h
                    
    input_ = tf.nn.max_pool(h, ksize=[1,params.pooling_size, params.pooling_size, 1], strides=[1, params.pooling_size, params.pooling_size, 1], padding='SAME')
    input_ = tf.nn.dropout(input_, keep_prob=0.95)
    

    lda_pred = tf.contrib.layers.fully_connected(
    							     inputs=topics_latent_vector,
    							     num_outputs=params.num_of_classes,
    							     activation_fn=None,
    							     reuse=False,
							     trainable=True)
 
    # Shape of h : (batch_size, max_length, embeddings_dim, filter_size_of_the_last_block)
    conv_latent = tf.reshape(input_, (input_.shape[0], -1) )# Shape of this var is: batch_size x max_length x embeddings_dim x filter_size_of_the_last_layer
    print ("Getting the logits...")
    #I am going to use a fully connected layer to get the logits for the 8 classes of scores. Output of this layer should be [batch_size, 8]
    with tf.variable_scope('logits') as logits_scope:

        conv_pred = tf.contrib.layers.fully_connected(
                                                  inputs=conv_latent,
                                                  num_outputs=params.num_of_classes,
                                                  activation_fn=None,
                                                  scope=logits_scope,
                                                  reuse=False,
						  trainable=True
                                                  )
    #Convert logits to probabilities.
    print ("Selecting mode...")
    
    pred = params.a * conv_pred + (1-params.a) * lda_pred
    soft_pred = tf.nn.softmax(pred)
    mask = []
    for i in range(pred.shape[0]):
        top_2 = tf.nn.top_k(soft_pred[i], 2)
        a = top_2[0][0]
        b = top_2[0][1]
        sub = a-b
        sub_flag = tf.less(sub, params.threshold)
        mask.append(tf.cond(sub_flag, lambda:False, lambda:True)) 

    predictions = {
                  "classes": tf.argmax(input=tf.nn.softmax(pred), axis=1),
                  "probabilities": tf.nn.softmax(pred, name="softmax_tensor")
                  }
    if mode != tf.estimator.ModeKeys.PREDICT:
        print ("Calculating loss...")
        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         labels = scores,
                                                         logits = conv_pred,
                                                         name = 'conv_loss'
                                                         )
        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         labels = scores,
                                                         logits = lda_pred,
                                                         name = 'lda_loss'
                                                         )
        loss = params.a * loss1 + (1-params.a) * loss2

        batch_loss = tf.reduce_mean(loss)
        print ("Optimizing...")
        optimizer = tf.train.MomentumOptimizer(params.learning_rate, params.momentum)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimizer.minimize(
                                         loss = batch_loss,
                                         global_step = tf.train.get_global_step()
                                         )
            return tf.estimator.EstimatorSpec(mode=mode, loss=batch_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            scores = tf.boolean_mask(scores, mask)
            predictions_short = tf.boolean_mask(predictions['classes'], mask)
            eval_metric_ops = {
                              "accuracy": tf.metrics.accuracy(labels=scores, predictions=predictions_short),
                              "conf_matrix": eval_confusion_matrix(scores, predictions_short),
                              "cohen_kappa": tf.contrib.metrics.cohen_kappa(scores, predictions_short, params.num_of_classes)
                              }
            return tf.estimator.EstimatorSpec(mode=mode, loss=batch_loss, eval_metric_ops=eval_metric_ops)
    elif mode == tf.estimator.ModeKeys.PREDICT:
       total_parameters = 0
       for variable in tf.trainable_variables():
       # shape is an array of tf.Dimension
             print (variable.name)
             shape = variable.get_shape()
             print(shape)
             print(len(shape))
             variable_parameters = 1
             for dim in shape:
                 print(dim)
                 variable_parameters *= dim.value
             print(variable_parameters)
             total_parameters += variable_parameters
       print(total_parameters)
       return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=7)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(7,7), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op
