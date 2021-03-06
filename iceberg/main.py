import tensorflow as tf

from model import CNN, Resnet
from reader import TFReader, DefaultReader
from task import train, evaluate, predict, batch_predict

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
    keep_prob = 0.5 if config.mode == 'train' else 1.0
    if config.mode == 'train':
        reader = TFReader(config.data_path, config.epoch, config.batch_size, [75*75*2], [1])
        cnn_model = CNN(reader, config.mode, keep_prob=keep_prob, learning_rate=config.learning_rate)
        # resnet = Resnet(reader, config.mode, keep_prob=0.5, learning_rate=config.learning_rate)
        train(cnn_model, config)
    elif config.mode == 'evaluate':
        reader = TFReader(config.data_path, config.epoch, config.batch_size, [75*75*2], [1])
        cnn_model = CNN(reader, config.mode, keep_prob=keep_prob, learning_rate=config.learning_rate)
        # resnet = Resnet(reader, config.mode, keep_prob=keep_prob)
        evaluate(cnn_model, config)
    elif config.mode == 'predict':
        reader = DefaultReader(None)
        cnn_model = Resnet(reader, config.mode, keep_prob=keep_prob)
        predict(cnn_model, config)
    elif config.mode == 'batch_predict':
        reader = TFReader(config.data_path, 1, config.batch_size, [75*75*2], [1], shuffle=False)
        # resnet = Resnet(reader, config.mode, keep_prob=keep_prob)
        cnn_model = CNN(reader, config.mode, keep_prob=keep_prob)
        batch_predict(cnn_model, config)


if __name__ == '__main__':
    tf.app.run()