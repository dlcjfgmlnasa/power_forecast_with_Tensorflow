from source.data_helper import *
from source.lstm.model import *
import numpy as np
import random


class LSTMBatchIter(BatchIter):
    def __init__(self,
                 file_path_list,
                 time_step,
                 batch_size,
                 epoch_size):
        super(LSTMBatchIter, self).__init__(
            file_path_list=file_path_list,
            time_step=time_step,
            batch_size=batch_size,
            epoch_size=epoch_size
        )

    def get_mini_batches(self):
        end_point = self._total_size - self._time_step - 1
        batches = {'data': list(), 'label': list()}

        for _ in range(self._batch_size):
            point = random.randrange(0, end_point)
            start = point
            end = start + self._time_step
            batches['data'].append(self._values[start: end])
            batches['label'].append(self._values[end+1].reshape(-1, 1))

        # list to numpy array
        batches['data'] = np.array(batches['data'])
        batches['label'] = np.array(batches['label'])
        return batches

    def batch_generator(self):
        one_iterator = int(self._total_size / (self._time_step * self._batch_size))
        for _ in range(self._epoch_size):
            for _ in range(int(one_iterator)):
                sample_batch = self.get_mini_batches()
                yield sample_batch

if __name__ == '__main__':
    model = BasicLSTMModel(
        time_step=96,
        feature_size=2,
        num_units=100,
        rnn_static_size=2,
        dnn_dims=[100, 200]
    )
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(model.cost)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    tf.summary.scalar('cost', model.cost)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

