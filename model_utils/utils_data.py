'''
This file contains functions to load and preprocess datasets for training and evaluation.
'''

import os
import numpy as np
from torch.utils.data import Dataset
from scipy.io.arff import loadarff
from model_utils.utils_distance_matrix import *
from model_utils.utils_traffic_data import get_AMS_dataset, get_INT_dataset
from sklearn.preprocessing import StandardScaler


def set_nan_to_zero(a):
    a[np.isnan(a)] = 0
    return a


def normalize_TS(TS):
    TS = set_nan_to_zero(TS)
    if TS.ndim == 2: # univariate
        TS_max = TS.max(axis = 1).reshape(-1,1)
        TS_min = TS.min(axis = 1).reshape(-1,1)
        TS = (TS - TS_min)/(TS_max - TS_min + (1e-6))
    elif TS.ndim == 3: # multivariate
        N, D, L = TS.shape
        TS_max = TS.max(axis=2).reshape(N,D,1) 
        TS_min = TS.min(axis=2).reshape(N,D,1)
        TS = (TS - TS_min) / (TS_max - TS_min + (1e-6))
    return TS


def compute_sim_mat(data, dist_metric='DTW'):
    """
    Compute similarity matrix for time series, sim_mat.shape = (n_instance, n_instance)
    """
    if data.ndim == 3: # (n_instance, n_timestamps, n_features)
        if data.shape[2] == 1:
            multivariate = False
            data = data.reshape(data.shape[0], -1)
        else:
            multivariate = True
        norm_TS = normalize_TS(data)
        sim_mat = save_sim_mat(norm_TS, multivariate, dist_metric)
    elif data.ndim == 4: 
        """
        MacroTraffic (n_instance, n_timestamps, n_nodes, n_features)
        MicroTraffic (n_instance, n_agents, n_timestamps, n_features)
        """
        if data.shape[1] == 26: # MicroTraffic, transpose to (n_instance, n_timestamps, n_agents, n_features)
            data = data.transpose(0, 2, 1, 3)
        if data.shape[3] == 1:
            data = data.reshape(data.shape[0], data.shape[1], -1)
            sim_mat = compute_sim_mat(data, dist_metric)
        else:
            sim_mat = np.zeros((data.shape[0], data.shape[0]))
            for channel_idx in range(data.shape[3]):
                sim_mat += compute_sim_mat(data[..., channel_idx], dist_metric)
            sim_mat /= data.shape[3]
    return sim_mat


def get_sim_mat(loader, data, dataset='', dist_metric='DTW', prefix='train'):
    if prefix=='train' and data.shape[0] > 7000:
        print(f"Data shape {data.shape} is too large, {dist_metric} similarity will be computed during training.")
        return None
    else:
        sim_mat_dir = os.path.join(f'datasets/{loader}', dataset)
        os.makedirs(sim_mat_dir, exist_ok=True)
        sim_mat_dir = os.path.join(sim_mat_dir,f'{prefix}_{dist_metric}.npy')

        if os.path.exists(sim_mat_dir):
            print(f"Loading {dist_metric} ...")
            sim_mat = np.load(sim_mat_dir)
        else:
            print(f"Computing & Saving {dist_metric} ...")
            sim_mat = compute_sim_mat(data, dist_metric)
            np.save(sim_mat_dir, sim_mat)

    return sim_mat


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def modify_train_data(train_data, train_labels=None, max_train_length=3000):
    # check if the dimensionality of training data is equal to 3, 
    # expected shape (n_instance, n_timestamps, n_features)
    assert train_data.ndim == 3
    # split the training data into sections if the length is too long
    if max_train_length is not None:
        sections = train_data.shape[1] // max_train_length
        if sections >= 2:
            train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
            if train_labels is not None:
                train_labels = np.concatenate([train_labels] * sections, axis=0).reshape(-1)
    # if the first or last row has missing values (varying lengths), centerize the data
    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)
    if train_labels is not None:
        return train_data, train_labels
    else:
        return train_data


def load_UEA(dataset, dataset_dir=None):
    dataset_dir = 'datasets' if dataset_dir is None else dataset_dir
    try:
        train_data = loadarff(f'{dataset_dir}/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
        test_data = loadarff(f'{dataset_dir}/UEA/{dataset}/{dataset}_TEST.arff')[0]
    except:
        import io
        with open(f'{dataset_dir}/UEA/{dataset}/{dataset}_TRAIN.arff', 'r', encoding='utf-8') as f:
            train_data = loadarff(io.StringIO(f.read()))[0]
        with open(f'{dataset_dir}/UEA/{dataset}/{dataset}_TEST.arff', 'r', encoding='utf-8') as f:
            test_data = loadarff(io.StringIO(f.read()))[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    # Modify train_data in case of too long sequences or missing values, max. length is by default 3000
    train_X, train_y = modify_train_data(train_X, train_y)

    return train_X, train_y, test_X, test_y
    

def load_MacroTraffic(years, time_interval=5, horizon=15, observation=20, dataset_dir='datasets'):
    datasets = get_AMS_dataset(years, time_interval, horizon, observation, dataset_dir)
    trainset, validationset, testset = datasets

    scaler = StandardScaler()
    scaler.fit(trainset.reshape(-1, trainset.shape[-1]))
    trainset = scaler.transform(trainset.reshape(-1, trainset.shape[-1])).reshape(trainset.shape)
    validationset = scaler.transform(validationset.reshape(-1, validationset.shape[-1])).reshape(validationset.shape)
    testset = scaler.transform(testset.reshape(-1, testset.shape[-1])).reshape(testset.shape)
    return trainset, validationset, testset


def load_MicroTraffic(filenames=['train1', 'train2', 'train3', 'train4'], dataset_dir='datasets'):
    trainset = get_INT_dataset('train', filenames, dataset_dir)
    validationset = get_INT_dataset('val', dataset_dir=dataset_dir)
    testset = get_INT_dataset('test', dataset_dir=dataset_dir)
    return trainset, validationset, testset


def assign_soft_labels(sim_mat, tau_inst):
    if tau_inst <= 0:
        soft_labels = None
    else:
        if sim_mat is None:
            soft_labels = 'compute'
        else:
            tau_inst = float(tau_inst)
            alpha = 0.5
            soft_labels = (2*alpha) / (1 + np.exp(tau_inst*abs(1 - sim_mat))) + (1-alpha)*np.eye(sim_mat.shape[0])
    return soft_labels


class custom_dataset(Dataset):
    def __init__(self, X, loader):
        self.X = X
        if 'Macro' in loader:
            self.X = self.X[:, :-15, :, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, idx

