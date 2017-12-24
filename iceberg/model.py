import tensorflow as tf

class CNN(object):
    """A plain convolution neural network.
    Args:
        reader: a reader providing training data
        is_training: is training
        keep_prob: keep probability
    """
    def __init__(self,  reader, mode, keep_prob=0.5, learning_rate=0.001):
        self.reader = reader
        self.keep_prob = keep_prob
        self.graph = tf.Graph()
        self.learning_rate = learning_rate
        with self.graph.as_default():
            self.x, self.labels = self.reader.read()
            if mode == "train":
                self.build_train_graph()
            elif mode == "evaluate":
                self.build_evalulate_graph()
            elif mode == "predict":
                self.build_predict_graph()

    def build_train_graph(self):
        self.build_architecture(keep_prob=self.keep_prob)
        self.calc_loss()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self._summary()

    def build_evalulate_graph(self):
        self.build_architecture(keep_prob=self.keep_prob)
        self.calc_accuracy()

    def build_predict_graph(self):
        self.build_architecture(keep_prob=self.keep_prob)

    def calc_accuracy(self):
        preditions = tf.argmax(self.logits, 1)
        labels = tf.argmax(self.labels, 1)

        self.acc, self.acc_op = tf.metrics.accuracy(labels=labels, predictions=preditions)

    def build_architecture(self, keep_prob=0.5):
        with tf.variable_scope('Conv_layer'):
            # 75 x 75 x 2
            images = tf.reshape(self.x, [-1, 75, 75, 2])

            feature_layer = conv_layer(images, shape=[3, 3, 2, 64])

            # 75 x 75 x 32
            conv1 = conv_layer(feature_layer, shape=[5, 5, 64, 64])
            conv2 = conv_layer(feature_layer, shape=[5, 5, 64, 64])

            # 37 x 37 x 32
            conv2_pool = max_pool_2x2(conv2)

            # 37 x 37 x 64
            conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 64])
            conv4 = conv_layer(conv3, shape=[5, 5, 64, 64])

            # 19 x 19 x 64
            conv4_pool = max_pool_2x2(conv4)

            # full layer 20736
            conv3_flat = tf.reshape(conv4_pool, [-1, 19*19*64])

            # dropout 20736
            conv3_drop = tf.nn.dropout(conv3_flat, keep_prob = keep_prob)

            # 1024
            full1 = tf.nn.relu(full_layer(conv3_drop, 1024))

            # dropout 1024
            full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

            # 2
            self.logits = full_layer(full1_drop, 2)
            self.softmax = tf.nn.softmax(self.logits)
            self.predict = tf.argmax(self.logits, 1)

    def calc_loss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.loss = tf.reduce_mean(loss)

    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/loss', self.loss))
        self.train_summary = tf.summary.merge(train_summary)


# util functions here
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")


def conv_layer(input, shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, w)+b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    w = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, w) + b
