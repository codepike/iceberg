import tensorflow as tf
from resnet import softmax_layer, conv_layer_res, residual_block
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, InputLayer
from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


class Model(object):
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
        self.mode = mode
        self.is_training = mode == 'train'
        with self.graph.as_default():
            self.x, self.labels = self.reader.read()
            if mode == "train":
                self.build_train_graph()
            elif mode == "evaluate":
                self.build_evalulate_graph()
            elif mode == "predict" or mode == "batch_predict":
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
        pass

    def calc_loss(self):
        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        loss = tf.losses.log_loss(self.labels, self.softmax)
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


class CNN(Model):
    def __init__(self,  reader, mode, keep_prob=0.5, learning_rate=0.001):
        super(CNN, self).__init__(reader, mode, keep_prob, learning_rate)

    def build_architecture(self, keep_prob=0.5):
        with tf.variable_scope('Conv_layer'):
            # 75 x 75 x 2
            self.images = tf.reshape(self.x, [-1, 75, 75, 2])

            K.set_learning_phase(1)
            model = Sequential()

            model.add(InputLayer(input_tensor=self.images))

            # CNN 1
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 2)))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            model.add(Dropout(0.2))

            # CNN 2
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.2))

            # CNN 3
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.3))

            # CNN 4
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.3))

            # You must flatten the data for the dense layers
            model.add(Flatten())

            # Dense 1
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))

            # Dense 2
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))

            # Output
            model.add(Dense(2, activation="sigmoid"))

            # optimizer = Adam(lr=0.001, decay=0.0)
            # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # model.summary()

            self.model = model
            self.output = model.output
            self.logits = self.output
            self.softmax = tf.nn.softmax(self.logits)
            self.predict = tf.argmax(self.logits, 1)


def resnet(inpt, n, is_training=False):
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = int((n - 20) / 12 + 1)
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer_res(inpt, [3, 3, 3, 16], 1, is_training)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.variable_scope('conv2_%d' % (i + 1)):
            conv2_x = residual_block(layers[-1], 16, False, is_training=is_training)
            conv2 = residual_block(conv2_x, 16, False, is_training=is_training)
            layers.append(conv2_x)
            layers.append(conv2)
            print("conv2 shape: {}".format(conv2.shape))

        assert conv2.get_shape().as_list()[1:] == [75, 75, 16]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 32, down_sample, is_training=is_training)
            conv3 = residual_block(conv3_x, 32, False, is_training=is_training)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [38, 38, 32]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 64, down_sample, is_training=is_training)
            conv4 = residual_block(conv4_x, 64, False, is_training=is_training)
            layers.append(conv4_x)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [19, 19, 64]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]

        out = softmax_layer(global_pool, [64, 2])
        layers.append(out)

    return layers[-1]



class Resnet(Model):
    def __init__(self, reader, mode, keep_prob=0.5, learning_rate=0.001):
        super(Resnet, self).__init__(reader, mode, keep_prob, learning_rate)

    def build_architecture(self, keep_prob=0.5):
        with tf.variable_scope('Conv_layer'):
            # 75 x 75 x 2
            print(self.x.shape)
            images = tf.reshape(self.x, [-1, 75, 75, 3])

            self.logits = resnet(images, 32, self.is_training)
            self.softmax = tf.nn.softmax(self.logits)
            self.predict = tf.argmax(self.logits, 1)