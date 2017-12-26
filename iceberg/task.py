import tensorflow as tf
import numpy as np
import json


def train(model, config):
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=config.logdir, save_model_secs=60, global_step=model.global_step)

    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            _, summary_str, step = sess.run([model.train_op, model.train_summary, model.global_step])
            supervisor.summary_writer.add_summary(summary_str, step)


def evaluate(model, config):
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=config.logdir, save_model_secs=0)

    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            accuracy = sess.run([model.acc, model.acc_op])
            print("accuracy: ", accuracy)


def predict(model, config):
    rawdata = open(config.data_path).read()
    data = json.loads(rawdata)

    result = open(config.output, 'w')
    result.write("id,is_iceberg\n")
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=config.logdir, save_model_secs=0)
    with supervisor.managed_session() as sess:
        for item in data:
            id = item['id']
            band_1 = item["band_1"]
            band_2 = item["band_2"]

            band_1 = np.array(band_1)
            band_2 = np.array(band_2)

            band_1 = band_1.reshape((75,75))
            band_2 = band_2.reshape((75,75))

            x = np.dstack((band_1, band_2))
            x = x.reshape((-1, 11250))
            x = x.astype(np.float32)

            probability = sess.run([model.softmax], feed_dict={model.x: x})
            result.write("{},{}\n".format(id, probability[0][0][0]))

    result.close()


def batch_predict(model, config):
    supervisor = tf.train.Supervisor(graph=model.graph, logdir=config.logdir, save_model_secs=0)

    with supervisor.managed_session() as sess:
        while not supervisor.should_stop():
            predictions = sess.run(model.softmax)
            n = predictions.shape[0]
            for i in range(n):
                print(predictions[i][1])
