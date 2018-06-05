from source.data_helper import *
import random


class DataConverter(object):
    def __init__(self,
                 append_pre_power=True,
                 convert_time_property=None,
                 trainable=True,
                 normalization=True,
                 min_value=None,
                 max_value=None):
        self.append_pre_power = append_pre_power
        self._convert_time_property = convert_time_property
        self._time_inst = TimeConverter(convert_time_property)
        self._normalization = normalization

        # Trainable 모드가 아닐때 사용
        if not trainable:
            self.min_value = min_value
            self.max_value = max_value

    def convert(self, frame):
        if self.append_pre_power:
            frame = append_pre_power_frame(frame)
        frame = self._time_inst.convert(frame)
        return frame

    def normalization(self):
        pass


class LSTMBatchGenerator(BatchGenerator):
    def __init__(self,
                 file_path_list,
                 time_step,
                 batch_size,
                 epoch_size,
                 convert_time_property=None,
                 normalization=False):
        super(LSTMBatchGenerator, self).__init__(
            file_path_list=file_path_list,
            time_step=time_step,
            batch_size=batch_size,
            epoch_size=epoch_size
        )
        self._convert_time_property = convert_time_property
        self._normalization = normalization
        self._converter = DataConverter(
            convert_time_property=self._convert_time_property,
            trainable=True,
            normalization=True
        )

    def get_batches(self):
        frame = self._converter.convert(self._total_frame)
        other_column_name = [column for column in frame.columns if column != 'Power']
        labels = frame['Power'].values
        features = frame[other_column_name].values

        last_index = self.total_size - self._time_step - 1

        for _ in range(self._epoch_size):
            for _ in range(int(self.total_size / self._time_step)):
                # make sample mini batch
                feature_batch = []
                label_batch = []
                for _ in range(self._batch_size):
                    first_index = random.randrange(0, last_index)
                    last_index = first_index + self._batch_size + 1
                    feature_batch.append(features[first_index:last_index])
                    label_batch.append(labels[last_index])

                batches = {'feature': np.array(feature_batch), 'label': np.array(label_batch).reshape(-1, 1)}
                yield batches

generator = LSTMBatchGenerator(
    ['../../data/original/paldal_ward_field.csv'],
    20, 20, 20,
    convert_time_property=['month', 'day', 'hour', 'minute', 'hour_minute'],
    normalization=True
)

test = next(generator.get_batches())
print(test['feature'])
print(test['label'])

