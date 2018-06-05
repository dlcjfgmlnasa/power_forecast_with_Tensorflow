# -*- encoding:utf-8 -*-
from source.data_helper import BatchIter
import tensorflow as tf
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
            batches['label'].append(self._values[end+1])

        # list to numpy array
        batches['data'] = np.array(batches['data'])
        batches['label'] = np.array(batches['label'])
        return batches

    def next_batches(self):
        one_iterator = int(self._total_size / (self._time_step * self._batch_size))
        for _ in range(self._epoch_size):
            for _ in range(int(one_iterator)):
                sample_batch = self.get_mini_batches()
                yield sample_batch


class Model(object):
    def __init__(self):
        pass

    def build_model(self):
        pass

if __name__ == '__main__':
    batch = LSTMBatchIter(
        ['../../data/original/paldal_ward_field.csv'],
        96, 20, 1000
    )

    for h in batch.next_batches():
        print(h['data'])
        break