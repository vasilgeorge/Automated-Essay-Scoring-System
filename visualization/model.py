import tensorflow as tf
import numpy as  np

from hyperparameters import Parameters

def gcnn_fn(essay, high_score, low_score, topic):
    tf.reset_default_graph()
    print("Fetching parameters file...")
    params = Parameters()
    params.get_params()

    essays = tf.placeholder(dtype=tf.int32, shape=(1,params.average_length))
    h_score = tf.placeholder(dtype=tf.int32, shape=(1))
    l_score = tf.placeholder(dtype=tf.int32, shape=(1))
    topics_latent_vector = tf.placeholder(dtype=tf.int32, shape=(1,5))
    topics_latent_vector = tf.cast(topics_latent_vector, tf.float32)      
    with tf.variable_scope('word_embeddings') as embeddings_scope:
        print("Getting the word embeddings map...")
        #Create the word embeddings map, that maps words(ints) to fixed size float vectors. Shared across all the layers.
        embeddings_map = tf.get_variable(
                                        name = 'embeddings_map',
                                        shape = [params.vocab_size, params.embeddings_dimensions],
                                        dtype = tf.float32
                                        #initializer = pretrained_weights,
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
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
                b = tf.get_variable(name="linear_b", shape=[params.filter_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0))

                V = tf.get_variable(name="gated_V", shape=[params.filter_height, params.filter_width, input_.shape[-1], params.filter_size],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.1))
                c = tf.get_variable(name="gated_c", shape=[params.filter_size], dtype=tf.float32, initializer=tf.constant_initializer(1.0))


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
    							     activation_fn=None
    							     )
 
    # Shape of h : (batch_size, max_length, embeddings_dim, filter_size_of_the_last_block)
    conv_latent = tf.reshape(input_, (input_.shape[0], -1) )# Shape of this var is: batch_size x max_length x embeddings_dim x filter_size_of_the_last_layer
    #I am going to use a fully connected layer to get the logits for the 8 classes of . Output of this layer should be [batch_size, 8]
    with tf.variable_scope('logits') as logits_scope:

        conv_pred = tf.contrib.layers.fully_connected(
                                                  inputs=conv_latent,
                                                  num_outputs=params.num_of_classes,
                                                  activation_fn=None,
                                                  scope=logits_scope
                                                  
						  
                                                  )
    #Convert logits to probabilities.
    print ("Selecting mode...")
    
    pred = params.a * conv_pred + (1-params.a) * lda_pred
    predictions = {
                  "classes": tf.argmax(input=tf.nn.softmax(pred), axis=1),
                  "probabilities": tf.nn.softmax(pred, name="softmax_tensor")
                  }
    #print('LABELS SHAPE: ', .shape)
    print ("Calculating loss...")
    loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         labels = h_score,
                                                         logits = conv_pred,
                                                         name = 'conv_loss'
                                                         )
    loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         labels = h_score,
                                                         logits = lda_pred,
                                                         name = 'lda_loss'
                                                         )
    loss_a = params.a * loss1 + (1-params.a) * loss2



    loss11 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         labels = l_score,
                                                         logits = conv_pred,
                                                         name = 'conv_loss'
                                                         )
    loss22 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                         labels = l_score,
                                                         logits = lda_pred,
                                                         name = 'lda_loss'
                                                         )
    loss_b = params.a * loss11 + (1-params.a) * loss22

    loss = loss_a - loss_b
    
    grads = tf.gradients(loss, [embeddings_map])
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
       sess.run(init_op)
       ckpt_path = tf.train.latest_checkpoint('/vol/bitbucket/gv2117/checkpoints/76_1loss/')
       saver.restore(sess, ckpt_path)
       grads = sess.run(grads, feed_dict = {essays:essay, h_score:high_score, l_score:low_score, topics_latent_vector:topic})
       return grads

def _compute_gradients(tensor, var_list):
	grads = tf.gradients(tensor, var_list)
	return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]
