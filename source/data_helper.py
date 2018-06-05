# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import copy
import os


class TimeConverter(object):
    def __init__(self, convert_time_property=None):
        self._convert_time_property = convert_time_property

    def convert(self, frame):
        if self._convert_time_property is None:
            return frame

        columns_names = frame.columns
        other_column_names = [column for column in columns_names if column != 'Time']

        time_series = frame['Time']
        series_dict = {}
        for _property in self._convert_time_property:
            series = time_series.apply(lambda x: self._convert_time(x, _property))
            series_dict[_property] = series
        series_dict.update(dict(frame[other_column_names]))

        new_columns_names = self._convert_time_property
        new_columns_names.extend(other_column_names)
        frame = pd.DataFrame(series_dict, columns=new_columns_names)
        return frame

    @staticmethod
    def _convert_time(value, day_type):
        time_inst = datetime.datetime.strptime(str(value), '%m/%d/%Y %H:%M')
        if day_type == 'month':
            return time_inst.month
        elif day_type == 'day':
            return time_inst.day
        elif day_type == 'hour':
            return time_inst.hour
        elif day_type == 'minute':
            return time_inst.minute
        elif day_type == 'hour_minute':
            return time_inst.hour + (time_inst.minute / 60)


def append_pre_power_frame(frame):
    pre_power_frame = frame['Power'].shift(1)
    pre_power_frame[0] = pre_power_frame[1]
    frame['Pre_Power'] = pre_power_frame
    return frame


class BatchGenerator(object):
    def __init__(self,
                 file_path_list,
                 time_step,
                 batch_size,
                 epoch_size):
        self._file_path_list = file_path_list
        self._time_step = time_step
        self._batch_size = batch_size
        self._epoch_size = epoch_size
        self._total_frame = self._data_set_parser()
        self.total_size = len(self._total_frame)

    def _data_set_parser(self):
        frame = None
        for filename in self._file_path_list:
            sample_frame = pd.read_csv(filename, encoding='utf-8')
            if frame is None:
                frame = sample_frame
            else:
                frame = frame.append(sample_frame, ignore_index=True)
        return frame
