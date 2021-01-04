from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import random
import scipy.stats as stat


def generate_time_features(timestamps, features):
    time_features = []
    if 'abs_time' in features:
        time_features.append([ts.timestamp() for ts in timestamps])
    if 'hour_of_day' in features:
        time_features.append([ts.hour for ts in timestamps])
    if 'day_of_week' in features:
        time_features.append([ts.weekday() for ts in timestamps])
    if 'month_of_year' in features:
        time_features.append([ts.month for ts in timestamps])
    if 'is_weekday' in features:
        time_features.append([int(ts.weekday() < 5) for ts in timestamps])
    return np.array(time_features).T


def segment_time_series(data_dict,
                        num_types,
                        X_context=168,
                        y_horizon=48,
                        window_skip=10,
                        feature_set=['abs_time', 'hour_of_day', 'day_of_week', 'month_of_year']):
    # standardizing multivariate time series
    time_series = data_dict['time_series']
    event_sequence = data_dict['event_sequence']
    time_series_scaler = StandardScaler().fit(time_series)
    time_series_norm = time_series_scaler.transform(time_series)
    time_series_norm = pd.DataFrame(
        time_series_norm, columns=time_series.columns, index=time_series.index)

    # standardizing time features

    time_features = generate_time_features(time_series.index, feature_set)
    time_features_scaler = StandardScaler().fit(time_features)
    tf_pd = pd.DataFrame(time_features_scaler.transform(
        time_features), columns=feature_set, index=time_series.index)

    sampling_delta_norm = np.diff(tf_pd['abs_time']).mean()

    # processing event features
    event_time_stamps = [event[1] for event in event_sequence]
    event_tf = generate_time_features(event_time_stamps, features=feature_set)
    event_tf_norm = time_features_scaler.transform(event_tf)

    # ont-hot encoding of event types
    event_types = np.array([event[0] for event in event_sequence])
    event_type_encoder = OneHotEncoder().fit(event_types[:, None])
    event_type_code = event_type_encoder.transform(
        event_types[:, None]).toarray()

    # generate time delta features
    timestamps = [event[1] for event in event_sequence]
    timedeltas = np.diff(timestamps, prepend=timestamps[0])
    timedeltas = np.array([td.total_seconds() for td in timedeltas])
    abs_times = event_tf_norm[:, 0]
    timedelta_scaler = MinMaxScaler().fit(timedeltas[:, None])
    timedeltas = timedelta_scaler.transform(timedeltas[:, None]).ravel()
    event_pd = pd.Series(
        zip(*[event_type_code, abs_times, timedeltas]), index=event_time_stamps)
    # counting intenisty function
    intensities = []
    for i in range(num_types):
        time_sequence = [event[1] for event in event_pd if event[0][i] == 1]
        intensity = np.sum([stat.norm(time, sampling_delta_norm).pdf(
            tf_pd['abs_time']) for time in time_sequence], axis=0)
        intensities.append(intensity)
    intensities = np.stack(intensities, axis=1) / stat.norm(0, sampling_delta_norm).pdf(0)
    intensity_pd = pd.Series(
        [intensity for intensity in intensities], index=tf_pd.index)
    # generate data pairs
    data_pairs = []
    for idx in range(0, time_series.shape[0]-X_context-y_horizon, window_skip):
        pair_dict = {}
        pair_dict['X_ts'] = time_series_norm[idx:idx+X_context]
        pair_dict['y_ts'] = time_series_norm[idx +
                                             X_context:idx+X_context+y_horizon]
        pair_dict['X_tf'] = tf_pd[idx:idx+X_context]
        pair_dict['y_tf'] = tf_pd[idx+X_context:idx+X_context+y_horizon]
        X_start, X_end = pair_dict['X_ts'].index[0], pair_dict['X_ts'].index[-1]
        y_start, y_end = pair_dict['y_ts'].index[0], pair_dict['y_ts'].index[-1]
        pair_dict['X_events'] = event_pd[X_start:X_end]
        pair_dict['y_events'] = event_pd[y_start:y_end]
        pair_dict['X_intensity'] = intensity_pd[idx:idx+X_context]
        pair_dict['y_intensity'] = intensity_pd[idx + X_context:idx+X_context+y_horizon]
        data_pairs.append(pair_dict)

    dataset = {}
    dataset['data_pairs'] = data_pairs
    dataset['time_series_scaler'] = time_series_scaler
    dataset['time_feature_scaler'] = time_features_scaler
    dataset['timedelta_scaler'] = timedelta_scaler
    return dataset


class TsEventDataset(Dataset):
    def __init__(self, data_dict, num_types, X_context, y_horizon, window_skip, train_prop=0.8,
                 feature_set=['abs_time', 'hour_of_day', 'day_of_week', 'month_of_year']):
        self.dataset = segment_time_series(data_dict, num_types,
                                           X_context=X_context,
                                           y_horizon=y_horizon,
                                           window_skip=window_skip, feature_set=feature_set)

        # train_prop (p) : if p < 0 : leave last #p chunks out testing.
        #                  if 0<p<1 : first p proportion of chunks for traing.
        self.x_dim = self.dataset['data_pairs'][0]['X_ts'].shape[1]
        self.t_dim = self.dataset['data_pairs'][0]['X_tf'].shape[1]
        self.num_types = num_types
        self.train_prop = train_prop
        self.indexs_train = None
        self.indexs_test = None
        self.train_cursor = None
        self.test_cursor = None
        self.break_point = None
        self.__split_train_test()

    def __len__(self):
        return len(self.dataset['data_pairs'])

    def train_shuffle(self):
        random.shuffle(self.dataset['data_pairs'][0: self.break_point])
        self.train_cursor = 0

    def test_shuffle(self):
        random.shuffle(self.dataset['data_pairs'][self.break_point:])
        self.test_cursor = self.break_point

    def __split_train_test(self):
        if self.train_prop < 0:
            break_point = int(len(self)+self.train_prop)
        else:
            break_point = int(len(self)*self.train_prop)
        self.indexs_train = range(0, break_point)
        self.indexs_test = range(break_point, len(self))
        self.train_cursor = 0
        self.test_cursor = break_point
        self.break_point = break_point

    def __getitem__(self, idx):
        data_pair = self.dataset['data_pairs'][idx]
        X_ts, y_ts = np.array(data_pair['X_ts']).astype(
            np.float32), np.array(data_pair['y_ts']).astype(np.float32)
        X_tf, y_tf = np.array(data_pair['X_tf']).astype(
            np.float32), np.array(data_pair['y_tf']).astype(np.float32)
        X_events, _ = np.array(
            data_pair['X_events']), np.array(data_pair['y_events'])
        X_event_array = np.array([np.hstack([event_type, event_time, event_delta]).astype(
            np.float32) for event_type, event_time, event_delta in X_events])
        # processing auxiliary transition marks
        transition_time = X_tf[-1, 0]
        if len(X_events) > 0:
            transition_delta = transition_time - X_events[-1][-2]
        else:
            transition_delta = transition_time - X_tf[0, 0]

        transition_array = np.array(
            [0 for i in range(self.num_types)]+[transition_time, transition_delta])[None, :]
        if X_event_array.size == 0:
            X_event_array = transition_array
        else:
            X_event_array = np.concatenate(
                [X_event_array, transition_array], axis=0)
        # y_event_array = np.array([np.hstack([event_type, event_time, event_delta]).astype(
        #    np.float32) for event_type, event_time, event_delta in y_events])
        y_intensity = np.stack(data_pair['y_intensity']).astype(np.float32)
        X_intensity = np.stack(data_pair['X_intensity']).astype(np.float32)
        return (X_ts, X_tf, X_event_array, X_intensity), (y_ts, y_tf, y_intensity)

    def next_batch(self, batch_size=10, train=True):
        # X_ts_batch : (batch_size)
        end = False
        if train:
            if batch_size > len(self.indexs_train):
                batch_size = len(self.indexs_train)
            end_cursor = self.train_cursor + batch_size
            if (end_cursor > self.break_point):
                end_cursor = self.break_point
                end = True
            indexs = range(self.train_cursor, end_cursor)
            self.train_cursor = end_cursor
        else:
            if batch_size > len(self.indexs_test):
                batch_size = len(self.indexs_test)
            end_cursor = self.test_cursor + batch_size
            if (end_cursor > len(self.dataset['data_pairs'])):
                end_cursor = len(self.dataset['data_pairs'])
                end = True
            indexs = range(self.test_cursor, end_cursor)
            self.test_cursor = end_cursor
        (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_intensity_batch) = self._get_batch(indexs)
        return (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_intensity_batch), end

    def _get_batch(self, indexs):
        X_ts_batch, y_ts_batch = [], []
        X_tf_batch, y_tf_batch = [], []
        X_event_arrays = []
        X_intensity_batch, y_intensity_batch = [], []
        for index in indexs:
            (X_ts, X_tf, X_event_array, X_intensity), (y_ts, y_tf, y_intensity) = self.__getitem__(index)
            X_ts_batch.append(X_ts)
            X_tf_batch.append(X_tf)
            y_ts_batch.append(y_ts)
            y_tf_batch.append(y_tf)
            X_event_arrays.append(torch.Tensor(X_event_array))
            y_intensity_batch.append(torch.Tensor(y_intensity))
            X_intensity_batch.append(torch.Tensor(X_intensity))
            # add auxiliary transition mark
        X_ts_batch = torch.Tensor(np.stack(X_ts_batch, axis=0))
        X_tf_batch = torch.Tensor(np.stack(X_tf_batch, axis=0))
        y_ts_batch = torch.Tensor(np.stack(y_ts_batch, axis=0))
        y_tf_batch = torch.Tensor(np.stack(y_tf_batch, axis=0))
        max_X_length = max(map(len, X_event_arrays))
        X_event_batch = []
        for X_event_array in X_event_arrays:
            X_event_batch.append(torch.cat([torch.zeros(
                max_X_length-X_event_array.shape[0], X_event_array.shape[1]), X_event_array]))
        X_event_batch = torch.stack(X_event_batch)
        X_intensity_batch = torch.Tensor(np.stack(X_intensity_batch, axis=0))
        y_intensity_batch = torch.Tensor(np.stack(y_intensity_batch, axis=0))
        return (X_ts_batch, X_tf_batch, X_event_batch, X_event_arrays), (y_ts_batch, y_tf_batch, y_intensity_batch)
