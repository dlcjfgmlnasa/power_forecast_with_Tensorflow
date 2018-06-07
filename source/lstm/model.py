# -*- encoding:utf-8 -*-
import tensorflow as tf


class BasicLSTMModel(object):
    def __init__(self,
                 time_step,
                 feature_size,
                 num_units,
                 rnn_static_size,
                 rnn_output_size=1,
                 dnn_dims=list(),
                 dnn_weight_initialization=tf.contrib.layers.xavier_initializer(),
                 dnn_activation_func=tf.nn.relu):

        self.time_step = time_step
        self.feature_size = feature_size
        self.num_units = num_units
        self.rnn_static_size = rnn_static_size
        self.rnn_output_size = rnn_output_size
        self.dnn_weight_initialization = dnn_weight_initialization
        self.dnn_dims = dnn_dims
        self.dnn_activation_func = dnn_activation_func
        if dnn_dims is []:
            self.dnn_dims = [1]
        else:
            self.dnn_dims.append(1)

        # input, label placeholder setting
        self.x = tf.placeholder(tf.float32, [None, self.time_step, self.feature_size], name='input')
        self.y = tf.placeholder(tf.float32, [None, 1], name='label')
        self.dropout = {}

        # dropout placeholder setting
        for i in range(self.rnn_static_size):
            dropout_input_name = str(i) + '_dropout_input'
            dropout_output_name = str(i) + '_dropout_output'
            self.dropout[dropout_input_name] = tf.placeholder(tf.float32, name=dropout_input_name)
            self.dropout[dropout_output_name] = tf.placeholder(tf.float32, name=dropout_output_name)

        self.logit = self.build_model()
        self.cost = tf.losses.mean_squared_error(
            labels=self.y,
            predictions=self.logit
        )

    def build_model(self):
        rnn_output = self.lstm()
        output = self.dnn(rnn_output)
        return output

    def lstm(self):
        with tf.variable_scope('lstm'):
            cells = []
            for i in range(self.rnn_static_size):
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.num_units)
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell,
                    input_keep_prob=self.dropout[str(i) + '_dropout_input'],
                    output_keep_prob=self.dropout[str(i) + '_dropout_output']
                )
                cells.append(cell)
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            output, state = tf.nn.dynamic_rnn(
                cell=stacked_rnn_cell,
                inputs=self.x,
                dtype=tf.float32
            )
            output = output[:, :self.time_step-self.rnn_output_size, :]
        return output

    def dnn(self, output):
        shape = output.shape
        flatten_size = int(shape[1] * shape[2])
        output = tf.reshape(output, [-1, flatten_size])
        self.dnn_dims.insert(0, flatten_size)

        with tf.variable_scope('dnn'):
            for i, dims in enumerate(zip(self.dnn_dims[:-1], self.dnn_dims[1:])):
                layer_name = str(i) + '_' + 'dnn_layer'
                weight = tf.get_variable(
                    name=layer_name+'_weight',
                    shape=dims,
                    dtype=tf.float32,
                    initializer=self.dnn_weight_initialization
                )
                bias = tf.get_variable(
                    name=layer_name+'_bias',
                    shape=dims[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer
                )
                output = tf.matmul(output, weight) + bias
        return output
