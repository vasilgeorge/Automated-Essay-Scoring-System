class Parameters:
        """Inside this class we set all the hyperparameters that are needed to train our model"""

        def __init__(self):
            self.batch_size = None
            self.embeddings_dimensions = None
            self.hidden_size = None
            self.dense_shape = None
            self.loss = None
            self.epochs = None
            self.learning_rate = None
            self.decay = None
            self.window_size = None
            self.filter_height = None
            self.filter_width = None
            self.filter_size = None
            self.block_size = None
            self.num_of_classes = None
            self.momentum = None
            self.num_conv_layers = None
            self.max_length = None
            self.num_topics = None
            self.pooling_size = None
            self.a = None
            self.vocab_size = None
            self.average_length = None
            self.threshold = None
            self.n_hidden = None

        def get_params(self):
            self.batch_size = 16
            self.embeddings_dimensions = 200
            self.hidden_size = 300
            self.dense_shape = 1
            self.loss = 'mean_squared_error'
            self.epochs = 50
            self.learning_rate= 0.00001
            self.decay = 0.9
            self.window_size = 5
            self.filter_height = 3
            self.filter_width = 3
            self.filter_size = 16
            self.block_size = 2
            self.num_of_classes = 7
            self.num_conv_layers = 5
            self.grad_clip = 0.01
            self.momentum = 0.9
            self.num_topics = 5
            self.pooling_size = 3
            self.a = 0.7
            self.vocab_size = 7300
            self.average_length = 750
            self.threshold = 0.1
            self.n_hidden = 64
