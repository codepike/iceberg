import tensorflow as tf
import json
import os
import numpy as np

flags = tf.app.flags

flags.DEFINE_string("data_path", "./data/train.json", "The path of input file")
flags.DEFINE_string("output", "./data", "The directory of the output")
flags.DEFINE_float("start", 0.0, "The start percentage of data to be used")
flags.DEFINE_float("end", 100.0, "The end percentage of data to be used")

config = flags.FLAGS

SAME = 0
ROTATE = 1
FLIP_LEFT_RIGHT = 2
FLIP_UP_DOWN = 3


def augment(tensor, action=SAME):
    if action == ROTATE:
        return tf.contrib.keras.preprocessing.image.random_rotation(
                tensor, 30, row_axis=0, col_axis=1, channel_axis=2)
    elif action == FLIP_LEFT_RIGHT:
        return np.flip(tensor, 0)
    elif action == FLIP_UP_DOWN:
        return np.flip(tensor, 1)
    else:
        return tensor


def read_data(data, start, end, action):
    if start >= end:
        print("skipped, [{} - {}]".format(start, end))
        return []
    else:
        print("created, [{} - {}]".format(start, end))
        processed = []
        for i in range(start, end):
            label = data[i].get("is_iceberg", 0)   # 0 is ship and 1 is iceberg
            inc_angle = data[i]["inc_angle"] if data[i]["inc_angle"] != "na" else 0.0

            band_1 = np.array(data[i]["band_1"]).reshape((75,75))
            band_2 = np.array(data[i]["band_2"]).reshape((75,75))
            band_3 = band_1+band_2  # composite

            band_1 = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
            band_2 = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
            band_3 = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

            x = np.dstack((band_1, band_2, band_3))    # shape(75,75,3)
            x = augment(x, action).astype(np.float32)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'x': tf.train.Feature(bytes_list = tf.train.BytesList(value=[x.tostring()])),
                        'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                        'inc_angle': tf.train.Feature(float_list=tf.train.FloatList(value=[inc_angle])),
                    },
                )
            )

            processed.append(example)

        return processed


def create(filename, data, start, end, augment_data=False):
    writer = tf.python_io.TFRecordWriter(filename)

    examples = read_data(data, start, end, SAME)

    for example in examples:
        writer.write(example.SerializeToString())

    augment_actions = [FLIP_UP_DOWN, FLIP_LEFT_RIGHT]
    for aug in augment_actions:
        examples = read_data(data, start, end, aug)
        for example in examples:
            writer.write(example.SerializeToString())

    writer.close()


def main(_):
    raw_data = open(config.data_path).read()
    data = json.loads(raw_data)

    n = len(data)
    start = int(config.start * n / 100.0)
    end = int(config.end * n / 100.0)
    create(config.output, data, start, end)


if __name__ == '__main__':
    tf.app.run()
