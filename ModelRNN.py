#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import time
import platform
from LoadData import DataLoader
import os
from datetime import datetime


class ModelRNN:

    def __init__(self,
                 log_path,
                 model_path,
                 lstm_size=64,
                 num_layers=2,
                 learning_rate=0.001,
                 grad_clip=5):
        self.model_path = ''
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.inputs = None
        self.outputs = None
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        self.batch_size = None
        self.seq_lens = None
        self.feat_len = None
        self.symbols_len = None
        self.targets = None
        self.keep_prob = None
        self.log_path = log_path
        self.model_path = model_path
        self.seq_len_to_dynamic_rnn = None
        self.with_dependency = False

    def prefix_name(self):
        return "layers{}_seqLen{}_batchSize{}".format(self.lstm_size,
                                                      self.num_layers,
                                                      self.seq_lens,
                                                      self.batch_size)

    def build_lstm_model_lstm(self, batch_size, seq_lens, feat_len, symbols_length, test_mode: bool = False):
        if test_mode is True:
            seq_lens = None
            batch_size = 1

        self.batch_size = batch_size
        self.seq_lens = seq_lens

        self.feat_len = feat_len
        self.symbols_len = symbols_length

        self.inputs = tf.placeholder(shape=[batch_size, seq_lens, feat_len], dtype=tf.float32, name="inputs")

        self.targets = tf.placeholder(shape=[batch_size, seq_lens], dtype=tf.int32, name="targets")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        if test_mode is True:
            self.seq_len_to_dynamic_rnn = tf.placeholder(tf.int32, [None], name="time_seq")
        else:
            self.seq_len_to_dynamic_rnn = None

        def get_a_cell():
            lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
            return drop

        self.cells = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(self.num_layers)])

        self.initial_states = self.cells.zero_state(batch_size, dtype=tf.float32)

        self.state_in = tf.identity(self.initial_states, name='state_in')

        state_per_layer_list = tf.unstack(self.state_in, axis=0)
        state_in_tuple = tuple(
            # TODO make this not hard-coded to LSTM
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(self.num_layers)]
        )

        outputs, self.final_state = tf.nn.dynamic_rnn(self.cells, self.inputs, initial_state=state_in_tuple,
                                                      sequence_length=self.seq_len_to_dynamic_rnn)

        self.state_out = tf.identity(self.final_state, name='state_out')

        # when state_out finish reset state_in for next batch
        # unexpected the behavior of the control dependency
        # It is more precision by coping states from cpu to gpu
        # with tf.control_dependencies([self.state_out]):
        #     self.state_in = tf.identity(self.state_out)
        #     self.with_dependency = True

        seq_outputs = tf.concat(outputs, 1)

        x = tf.reshape(seq_outputs, [-1, self.lstm_size])

        softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, symbols_length], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(symbols_length))

        self.logits = tf.matmul(x, softmax_w) + softmax_b

        self.prediction = tf.nn.softmax(self.logits, name="prediction")

        y_ont_hot = tf.one_hot(self.targets, symbols_length)
        y_reshaped = tf.reshape(y_ont_hot, [-1, symbols_length])  # self.logits.get_shape())

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped))

        tf.summary.scalar('cross_entropy', self.loss)
        self.merged = tf.summary.merge_all()

        # optimizer
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, samples_loader: DataLoader = None):

        print("trainer")

        np.random.seed(int(time.time()))

        epochs = 1000

        save_every_n = 200

        TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())

        train_log_dir = os.path.join(self.log_path, TIMESTAMP)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(train_log_dir, sess.graph)

            state_in = sess.run(self.state_in)

            local_prediction = tf.argmax(self.prediction, 1)

            sess.graph.finalize()

            counter = 0
            for e in range(epochs):
                # Train network

                self.train_size = samples_loader.get_train_count()
                self.BATCH_SIZE = 2

                n_batchs_in_epoch = max(1, int(self.train_size / self.BATCH_SIZE))
                # print(n_batchs_in_epoch)

                for i in range(n_batchs_in_epoch):

                    counter += 1

                    x, y = samples_loader.next_batch_train(self.batch_size)
                    x = np.reshape(x, x.shape + (1,))
                    start = time.time()

                    feed = {self.inputs: x,
                            self.targets: y,
                            self.keep_prob: 1.0,
                            self.state_in: state_in
                            }

                    summary, batch_loss, state_in, _ = sess.run([self.merged,
                            self.loss,
                            self.state_out,
                            self.optimizer],
                            feed_dict=feed)
                    end = time.time()

                    if counter % 100 == 0:
                        print('epochs: {}/{}... '.format(e + 1, epochs),
                              'iterations: {}... '.format(counter),
                              'error: {:.4f}... '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end - start)))
                        writer.add_summary(summary, counter)

                    if counter % 100 == 0:
                        error_count = 0
                        loss_count = 0
                        amount = 0
                        for _ in range(samples_loader.get_validation_count()//self.batch_size):
                            x, y = samples_loader.next_batch_validation(self.batch_size)
                            x = np.reshape(x, x.shape + (1,))

                            feed = {self.inputs: x,
                                    self.targets: y,
                                    self.keep_prob: 1.0,
                                    self.state_in: state_in}

                            preds, loss, state_in = sess.run([local_prediction, self.loss, self.state_out],
                                                             feed_dict=feed)

                            diff = preds - np.reshape(y,[-1])

                            error_count += np.count_nonzero(diff)
                            loss_count += loss
                            amount += diff.size

                        print("validation match: {} % , avg loss {}".format((100 * (amount - error_count)) / amount,
                                                                        loss_count / n_batchs_in_epoch))

                    if (counter % save_every_n) == 0:
                        saver.save(sess, os.path.join(self.model_path,
                                                      "{}_trainingLoss{}".format(self.prefix_name(),
                                                                                 batch_loss)))

            saver.save(sess, os.path.join(self.model_path, "{}_trainingLoss{}".format(self.prefix_name(),
                                                                                      batch_loss)))

    def test(self, samples_loader: DataLoader=None, ckp_path: str=None):

        if ckp_path is None:
            last_model_path = tf.train.latest_checkpoint(self.model_path)
        else:
            last_model_path = ckp_path

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            print("last Model : {}".format(last_model_path))
            saver.restore(sess, last_model_path)

            self.train_size = samples_loader.get_test_count()
            self.BATCH_SIZE = 1
            next_sample_step = 1

            n_batchs_in_epoch = int(self.train_size / self.BATCH_SIZE) // next_sample_step

            state_in = sess.run(self.state_in)

            error_count = 0
            amount = 0
            loss_count = 0

            local_prediction = tf.argmax(self.prediction, 1)

            sess.graph.finalize()

            for i in range(n_batchs_in_epoch):

                x, y = samples_loader.next_batch_test()
                x = np.reshape(x, x.shape + (1,))
                seq_len_ = np.array([x.shape[1]])

                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: 1.0,
                        self.state_in: state_in,
                        self.seq_len_to_dynamic_rnn: seq_len_}

                preds, loss, state_in = sess.run([local_prediction, self.loss, self.state_out],
                                                 feed_dict=feed)

                diff = preds - y

                error_count += np.count_nonzero(diff)
                loss_count += loss

                amount += diff.size
                error_rate = np.count_nonzero(diff) / diff.size
                print("{} / {} errorRate: {} ; loss {}".format(i, n_batchs_in_epoch, error_rate, loss))

            print("result match: {} % , avg loss {}".format((100*(amount-error_count))/amount,
                                                            loss_count/n_batchs_in_epoch))


    def predict(self, loader: DataLoader, seq_count, ckp_path):

        if ckp_path is None:
            last_model_path = tf.train.latest_checkpoint(self.model_path)
        else:
            last_model_path = ckp_path

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            print("last Model : {}".format(last_model_path))
            saver.restore(sess, last_model_path)

            local_prediction = tf.argmax(self.prediction, 1)
            state_in = sess.run(self.state_in)
            sess.graph.finalize()

            for i in range(seq_count):
                x = np.array([[np.random.randint(1,self.symbols_len,1)[0]]])
                x = np.reshape(x, x.shape + (1,))
                result = []
                for j in range(5):


                    seq_len_ = np.array([x.shape[1]])

                    feed = {self.inputs: x,
                            self.keep_prob: 1.0,
                            self.state_in: state_in,
                            self.seq_len_to_dynamic_rnn: seq_len_}

                    preds, state_in = sess.run([local_prediction, self.state_out],
                                                     feed_dict=feed)
                    x[0] = preds[0]
                    result.append(preds[0])
                front = list(set(result))
                back = list(set(loader.statistics_back()))
                if len(front) == 5 and len(back) == 2:
                    print("{}: front {} back {}".format(i, front, back))


def train():

    print("run...")

    if os.path.exists("log") is False:
        os.mkdir("log")

    if os.path.exists("model") is False:
        os.mkdir("model")

    loader = DataLoader()

    loader.load_xls("dlt2.xls")

    rnn = ModelRNN("log", "model", lstm_size=128, num_layers=2, learning_rate=0.001)

    rnn.build_lstm_model_lstm(32, loader.get_seq_len(), 1, loader.get_classes_count(), test_mode=False)

    rnn.train(loader)


def test():

    if os.path.exists("log") is False:
        os.mkdir("log")

    if os.path.exists("model") is False:
        os.mkdir("model")

    loader_test = DataLoader()

    loader_test.load_xls("dlt2.xls")

    rnn = ModelRNN("log", "model", lstm_size=128, num_layers=2, learning_rate=0.001)

    rnn.build_lstm_model_lstm(1, loader_test.get_seq_len(), 1, loader_test.get_classes_count(), test_mode=True)

    rnn.test(loader_test)


def random():

    if os.path.exists("log") is False:
        os.mkdir("log")

    if os.path.exists("model") is False:
        os.mkdir("model")

    loader_test = DataLoader()

    loader_test.load_xls("dlt2.xls")

    rnn = ModelRNN("log", "model", lstm_size=128, num_layers=2, learning_rate=0.001)

    rnn.build_lstm_model_lstm(1, loader_test.get_seq_len(), 1, loader_test.get_classes_count(), test_mode=True)

    rnn.predict(loader_test, 16, None)


if __name__ == "__main__":
    test()