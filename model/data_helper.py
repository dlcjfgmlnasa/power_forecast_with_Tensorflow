# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os


class BatchIter(object):
    def __init__(self,
                 file_path_list,
                 time_step,
                 batch_size,
                 epoch_size):
        self._file_path_list = file_path_list
        self._time_step = time_step
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._values = self.data_set_parser()
        self._total_size = len(self._values)

    def data_set_parser(self):
        frame = None
        for filename in self._file_path_list:
            sample_frame = pd.read_csv(filename, encoding='utf-8')
            if frame is None:
                frame = sample_frame
            else:
                frame = frame.append(sample_frame, ignore_index=True)
        power_values = frame['Power'].values
        return power_values
