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


def create(filename, data, start, end):
    if start >= end:
        print("skipped {}, [{} - {}]".format(filename, start, end))
    else:
        print("created {}, [{} - {}]".format(filename, start, end))

    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(start, end):
        label = data[i].get("is_iceberg", 0)   # 0 is ship and 1 is iceberg
        band_1 = data[i]["band_1"]
        band_2 = data[i]["band_2"]
        x = np.array([band_1, band_2])
        inc_angle = data[i]["inc_angle"]
        if inc_angle == "na":
            inc_angle = 0.0
            label = 1

        x = x.astype(np.float32)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'x': tf.train.Feature(bytes_list = tf.train.BytesList(value=[x.tostring()])),
                    'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                    'inc_angle': tf.train.Feature(float_list=tf.train.FloatList(value=[inc_angle])),
                },
            )
        )

        writer.write(example.SerializeToString())

    writer.close()


def main(_):
    rawdata = open(config.data_path).read()
    data = json.loads(rawdata)

    n = len(data)
    start = int(config.start * n / 100.0)
    end = int(config.end * n / 100.0)
    create(config.output, data, start, end)

if __name__ == '__main__':
    tf.app.run()