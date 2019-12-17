import pandas as pd
import torch
import os
import numpy as np
import sys
from pathlib import Path
from random import shuffle
np.set_printoptions(threshold=sys.maxsize)


def load_data(ground_truth_dir): # gap_data_dir
    ground_truth_names = os.listdir(ground_truth_dir)
    ground_truth_names.sort()

    shuffle(ground_truth_names)

    idx = int(0.8*len(ground_truth_names))
    train_names = ground_truth_names[:idx]
    test_names = ground_truth_names[idx:]

    partition = {'train': train_names, 'test': test_names}

    training_set_size = len(ground_truth_names)

    return partition, training_set_size


def crop_data(path_gt, path_aug, new_gt_dir, new_aug_dir):

    assert isinstance(path_gt, str)
    assert isinstance(path_aug, str)
    assert isinstance(new_gt_dir, str)
    assert isinstance(new_aug_dir, str)

    directory_gt = Path(path_gt)
    directory_aug = Path(path_aug)

    files_gt = [p for p in directory_gt.iterdir() if p.is_file() and not str(p).endswith('.DS_Store')]
    files_aug = [p for p in directory_aug.iterdir() if p.is_file() and not str(p).endswith('.DS_Store')]

    files_gt.sort()
    files_aug.sort()

    file_size = 30

    dirs = [files_gt, files_aug]

    for d in dirs:
        i = 1
        for file in d:
            data = pd.read_csv(file, index_col=[0])
            if data.shape[1] == 31:
                data = data.drop(data.columns[30], axis=1)
            data = data[:(data.shape[0] - (data.shape[0] % 5))]
            for j in range(0, data.shape[0] - file_size, 5):
                new_file = data[j:j + file_size]
                if d == files_gt:
                    if i < 10:
                        np.save(new_gt_dir + 'CGT_0000' + str(i), new_file)
                        i += 1
                    elif i < 100:
                        np.save(new_gt_dir + 'CGT_000' + str(i), new_file)
                        i += 1
                    elif i < 1000:
                        np.save(new_gt_dir + 'CGT_00' + str(i), new_file)
                        i += 1
                    elif i < 10000:
                        np.save(new_gt_dir + 'CGT_0' + str(i), new_file)
                        i += 1
                    else:
                        np.save(new_gt_dir + 'CGT_' + str(i), new_file)
                        i += 1
                elif d == files_aug:
                    if i < 10:
                        np.save(new_aug_dir + 'CAUG_0000' + str(i), new_file)
                        i += 1
                    elif i < 100:
                        np.save(new_aug_dir + 'CAUG_000' + str(i), new_file)
                        i += 1
                    elif i < 1000:
                        np.save(new_aug_dir + 'CAUG_00' + str(i), new_file)
                        i += 1
                    elif i < 10000:
                        np.save(new_aug_dir + 'CAUG_0' + str(i), new_file)
                        i += 1
                    else:
                        np.save(new_aug_dir + 'CAUG_' + str(i), new_file)
                        i += 1


def remove_nan_files(path_gt, path_aug):
    assert isinstance(path_gt, str)
    assert isinstance(path_aug, str)

    directory_gt = Path(path_gt)
    directory_aug = Path(path_aug)

    files_gt = [p for p in directory_gt.iterdir() if p.is_file()]

    i = 0
    bad_files = []
    for file in files_gt:
        data = pd.read_csv(file, index_col=[0])
        if data.shape[1] == 31:
            data = data.drop(data.columns[30], axis=1)
        data = np.array(data)
        if True in np.isnan(data):
            i += 1
            bad_files.append(str(file)[-12:])

    bad_files_gt = []
    bad_files_aug = []

    for i in bad_files:
        bad_files_gt.append([str(p) for p in directory_gt.iterdir() if p.is_file() and i in str(p)])
        bad_files_aug.append([str(p) for p in directory_aug.iterdir() if p.is_file() and i in str(p)])
    bad_files_gt = [i for sublist in bad_files_gt for i in sublist]
    bad_files_aug = [i for sublist in bad_files_aug for i in sublist]

    for i in bad_files_gt:
        os.remove(i)

    for i in bad_files_aug:
        os.remove(i)


def apply_missing(time_series_tensor, max_erasures, max_gap_size):
    assert isinstance(time_series_tensor, torch.Tensor)

    if len(time_series_tensor.size()) < 3:
        time_series_tensor_copy = time_series_tensor.clone()
        time_series_tensor_copy = time_series_tensor_copy.double()
        time_series_tensor_copy = time_series_tensor_copy.unsqueeze(0)
    else:
        time_series_tensor_copy = time_series_tensor.clone()

    time_series_tensor_copy = time_series_tensor_copy
    missing_array = torch.Tensor().double()
    for k in range(time_series_tensor_copy.size(0)):
        erased_data = time_series_tensor_copy[k]

        # Pick starting index for erasure
        index_min = 0
        index_max = erased_data.size(0)

        # Pick starting column for erasure (must be x dim column)
        cols_min = 0
        cols_max = erased_data.size(1)
        start_cols = [i for i in range(cols_min, cols_max, 3)]

        previous_col = 0
        update = 0.6
        for erasure in range(np.random.randint(1, max_erasures)):
            if erasure == 0:
                probabilities = [1 / len(start_cols) for i in start_cols]
            else:
                probabilities = [(1 - update) / (len(start_cols) - 1) for i in start_cols]
                probabilities[int(previous_col / 3)] = update
            start_row = np.random.randint(index_min, index_max)
            start_col = np.random.choice(start_cols, p=probabilities, replace=True)
            erase_len = np.random.randint(1, max_gap_size)
            for i in range(erase_len):
                for j in range(3):
                    if start_row + i < index_max:
                        erased_data[start_row + i][start_col + j] = 0.0000
                    elif i == index_max:
                        continue
            previous_col = start_col
        erased_data = erased_data.unsqueeze(0)
        missing_array = torch.cat((missing_array, erased_data))

    return missing_array


def nan_to_zero(tensor):
    assert isinstance(tensor, torch.Tensor)

    tensor_copy = tensor.clone()
    tensor_copy = tensor_copy.float()
    tensor_copy[torch.isnan(tensor_copy)] = 0.0000
    return tensor_copy


def normalize_time_series(time_series_tensor, min_val, max_val):
    assert isinstance(time_series_tensor, torch.Tensor)

    time_series_tensor_copy = time_series_tensor.clone()
    normed_array = torch.Tensor().double()
    for i in range(time_series_tensor_copy.size(0)):
        normalized = time_series_tensor_copy[i]
        min = torch.min(normalized)
        normalized = (normalized - min)
        max = torch.max(normalized)
        normalized = normalized/max
        normalized = normalized * (max_val - min_val) + min_val
        normalized = normalized.unsqueeze(0)
        normed_array = torch.cat((normed_array, normalized))
    return normed_array


def split(data, index_cols=True, size=30, gap_length=5, result_dim=(0, 30, 30)):
    data = np.genfromtxt(data, delimiter=',')
    if index_cols:
        data = data[1:][:, 1:]
    data = data[:(data.shape[0] - (data.shape[0] % gap_length))]
    print('Original data shape: ' + str(data.shape))
    sub_lists = np.empty(result_dim)
    for j in range(0, data.shape[0] - size + gap_length, gap_length):
        data_slice = data[j:j + size]
        data_slice = np.expand_dims(data_slice, axis=0)
        sub_lists = np.append(sub_lists, data_slice, axis=0)
    data[np.isnan(data)] = 0.0000
    sub_lists[np.isnan(sub_lists)] = 0.0000
    return data, sub_lists


def merge(sub_lists, gap_length=5, result_dim=(0, 30)):
    super_list = np.empty(result_dim)
    for i in range(sub_lists.shape[0]-1):
        array = sub_lists[i]
        data_slice = array[:gap_length]
        super_list = np.append(super_list, data_slice, axis=0)
    super_list = np.append(super_list, sub_lists[-1], axis=0)
    print('Merged data shape: ' + str(super_list.shape))
    return super_list


def gap_fill(original, estimated):
    gap_filled = original.copy()
    idx = np.argwhere(np.isnan(gap_filled))
    for i in range(len(idx)):
        gap_filled[idx[i][0]][idx[i][1]] = estimated[idx[i][0]][idx[i][1]]
    return gap_filled

