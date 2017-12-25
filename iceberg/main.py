import tensorflow as tf

from model import CNN, Resnet
from reader import TFReader, DefaultReader
from task import train, evaluate, predict

flags = tf.app.flags

flags.DEFINE_string("data_path", "./train.tfrecord", "The input file path")
flags.DEFINE_string("logdir", "./logdir", "The output path")
flags.DEFINE_string("output", "./result.csv ", "The result of prediction")

flags.DEFINE_string("mode", "evaluate", "The task to perform [train, evaluate, predict]")
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")

config = flags.FLAGS


def main(_):
    if config.mode == 'train':
        reader = TFReader(config.data_path, config.epoch, config.batch_size, [75*75*2], [1])
        # cnn_model = CNN(reader, config.mode, keep_prob=0.5, learning_rate=config.learning_rate)
        resnet = Resnet(reader, config.mode, keep_prob=0.5, learning_rate=config.learning_rate)
        train(resnet, config)
    elif config.mode == 'evaluate':
        reader = TFReader(config.data_path, config.epoch, config.batch_size, [75 * 75 * 2], [1])
        resnet = Resnet(reader, config.mode, keep_prob=1.0)
        evaluate(resnet, config)
    elif config.mode == 'predict':
        reader = DefaultReader(None)
        resnet = Resnet(reader, config.mode, keep_prob=1.0)
        predict(resnet, config)


if __name__ == '__main__':
    tf.app.run()