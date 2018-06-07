# -*- encoding:utf-8 -*-
from source.lstm.model import BasicLSTMModel
from source.lstm.data_processing import *
import tensorflow as tf
import argparse

dnn_weight_initialization_list = ['xavier', 'he', 'random']
dnn_activation_func_name_list = ['relu', 'tanh', 'sigmoid']


parser = argparse.ArgumentParser('lstm model for power forecasting...')
parser.add_argument('--time_step', default=96, type=int, help='lstm time step (default : 96)')
parser.add_argument('--num_units', default=20, type=int, help='lstm units size (default : 20)')
parser.add_argument('--rnn_static_size', default=2, type=int, help='lstm static size (default : 2)')
parser.add_argument('--rnn_output_size', default=2, type=int, help='lstm output dims count (default: 1)')
parser.add_argument('--dnn_dims', default=[], type=list, help='lstm output dnn layers (default: [])')
parser.add_argument('--dnn_weight_initialization', default='xavier', type=str, choices=dnn_weight_initialization_list,
                    help='tf weight_initialization (default : xavier)')
parser.add_argument('--dnn_activation_func', default='relu', type=str, choices=dnn_activation_func_name_list,
                    help='tf activation function')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate (default : 0.001)')
args = parser.parse_args()

# args checking... args 변수가 올바르게 입력되었는지 확인
# DNN 에서 사용할 weight_initialization function 초기화
dnn_weight_initialization = None
if args.dnn_weight_initialization not in dnn_weight_initialization_list:
    raise NameError('')
else:
    if args.dnn_weight_initialization == 'xavier':
        dnn_weight_initialization = tf.contrib.layers.xavier_initializer()
    elif args.dnn_weight_initialization == 'he':
        dnn_weight_initialization = tf.contrib.layers.variance_scaling_initializer()
    elif args.dnn_weight_initialization == 'random':
        dnn_weight_initialization = tf.random_uniform_initializer()

# DNN 에서 사용할 activation function 초기화
dnn_activation_func = None
if args.dnn_activation_func not in dnn_activation_func_name_list:
    raise NameError('')
else:
    if args.dnn_activation_func == 'relu':
        dnn_activation_func = tf.nn.relu
    elif args.dnn_activation_func == 'tanh':
        dnn_activation_func = tf.nn.tanh
    elif args.dnn_activation_func == 'sigmoid':
        dnn_activation_func = tf.nn.sigmoid

# Train Batch Generator
generator = LSTMBatchGenerator(
    ['../../data/original/paldal_ward_field.csv'],
    20, 20, 20,
    convert_time_property=['month', 'day', 'hour', 'minute', 'hour_minute'],
    normalization=False
)

# LSTM Model
model = BasicLSTMModel(
    time_step=args.time_step,
    feature_size=generator.feature_columns_size,
    num_units=args.num_units,
    rnn_static_size=args.rnn_static_size,
    rnn_output_size=args.rnn_output_size,
    dnn_dims=args.dnn_dims,
    dnn_weight_initialization=dnn_weight_initialization,
    dnn_activation_func=dnn_activation_func
)

global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
grads_and_vars = optimizer.compute_gradients(model.cost)
optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

tf.summary.scalar('loss', model.cost)
merged = tf.summary.merge_all()
print(merged)
#
# for batches in generator.get_batches():
#     print(batches)
#     break


