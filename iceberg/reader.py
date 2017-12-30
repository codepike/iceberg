import tensorflow as tf
import os

class TFReader:
    """A reader reads TFRecord files
    Args:

    """
    def __init__(self, data_pah, epoch, batch_size, x_shape, label_shape, shuffle=True):
        self.data_path = data_pah
        self.epoch = epoch
        self.batch_size = batch_size
        self.x_shape = x_shape
        self.label_shape = label_shape
        self.shuffle = shuffle

    def read(self):
        if self.data_path.endswith('tfrecord'):
            filenames = [self.data_path]
        else:
            filenames = os.listdir(self.data_path)
            filenames = [name for name in filenames if name.endswith('.tfrecord')]
            filenames = [os.path.join(self.data_path, name) for name in filenames]

        # filename queues
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=self.epoch)

        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'x': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'inc_angle': tf.FixedLenFeature([], tf.float32),
            }
        )

        x = tf.decode_raw(features['x'], tf.float32)
        x.set_shape(self.x_shape)

        label = features['label']
        label = tf.one_hot(label, 2)

        capacity = self.batch_size * 10

        if self.shuffle:
            x_batch, label_batch = tf.train.shuffle_batch(
                [x, label],
                batch_size=self.batch_size,
                capacity=capacity,
                min_after_dequeue=0)
        else:
            x_batch, label_batch = tf.train.batch(
                [x, label],
                batch_size=self.batch_size,
                capacity=capacity,
                allow_smaller_final_batch=True)

        return x_batch, label_batch


class DefaultReader:
    """A reader read from placehodler
    Args:

    """
    def __init__(self, x_shape):
        self.x_shape = x_shape

    def read(self):
        x_batch = tf.placeholder(tf.float32, shape=(None, 75*75*3))
        label_batch = None
        return x_batch, label_batch