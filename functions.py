import pandas as pd
import torch
import os
import numpy as np
from pathlib import Path
from random import shuffle


def load_data(ground_truth_dir): # gap_data_dir
    # Input (gapped data)
    # get a list of strings of the input data file names (just names not whole path + extension)
    # gap_data_names = os.listdir(gap_data_dir)
    # gap_data_names.sort()

    ground_truth_names = os.listdir(ground_truth_dir)
    ground_truth_names.sort()

    # Output (full data)
    # List of torch tensors of ground truth
    directory = Path(ground_truth_dir)
    ground_truth_files = [p for p in directory.iterdir() if p.is_file()]
    ground_truth = []
    # for file in ground_truth_files:
    #     ground_truth_data = np.load(file)
    #     ground_truth_data = torch.Tensor(ground_truth_data)
    #     ground_truth.append(ground_truth_data)

    shuffle(ground_truth_names)

    idx = int(0.8*len(ground_truth_names))
    train_names = ground_truth_names[:idx]
    test_names = ground_truth_names[idx:]

    # x_y_pairs = list(zip(gap_data_names, ground_truth))
    # shuffle(x_y_pairs)
    #
    # training_pairs = x_y_pairs[:int(0.8*len(x_y_pairs))]
    # test_pairs = x_y_pairs[int(0.8*len(x_y_pairs)):]
    #
    # train_gap_data_names = list(list(zip(*training_pairs))[0])
    # test_gap_data_names = list(list(zip(*test_pairs))[0])

    partition = {'train': train_names, 'test': test_names}
    # ground_truth = dict(x_y_pairs)

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


def normalize_time_series(time_series_tensor, min_val, max_val):
    assert isinstance(time_series_tensor, torch.Tensor)

    time_series_tensor_copy = time_series_tensor.clone()
    time_series_tensor_copy = time_series_tensor_copy.float()
    normed_array = torch.Tensor()
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


def apply_missing(time_series_tensor, max_erasures, max_gap_size):
    assert isinstance(time_series_tensor, torch.Tensor)

    time_series_tensor_copy = time_series_tensor.clone()
    time_series_tensor_copy = time_series_tensor_copy.float()
    missing_array = torch.Tensor()
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
    tensor_copy[torch.isnan(tensor_copy)] = 0.0000
    return tensor_copy


# crop_data(path_gt='/home/henryp/PycharmProjects/MoGap/ground_truth_data/',
#           path_aug='/home/henryp/PycharmProjects/MoGap/augmented_data/',
#           new_gt_dir='/home/henryp/PycharmProjects/MoGap/cropped_ground_truth_data/',
#           new_aug_dir='/home/henryp/PycharmProjects/MoGap/cropped_augmented_data/')


# missing_marker_sim(gt_data_path='/home/henryp/PycharmProjects/MoGap/cropped_augmented_data/',
#                    erased_data_dir='/home/henryp/PycharmProjects/MoGap/cropped_augmented_data/',
#                    max_erasures=5,
#                    max_gap_size=28)

# def missing_marker_sim(gt_data_path, erased_data_dir, max_gap_size=200, max_erasures=10):
#     assert isinstance(gt_data_path, str)
#     assert isinstance(erased_data_dir, str)
#
#     directory = Path(gt_data_path)
#
#     files = [p for p in directory.iterdir() if p.is_file() and not str(p).endswith('.DS_Store')]
#
#     files.sort()
#     for file in files:
#         raw_data = pd.read_csv(file, index_col=[0])
#         erased_data = raw_data.drop(raw_data.columns[30], axis=1)
#         cols = [i for i in range(erased_data.shape[1])]
#         erased_data.columns = cols
#
#         # Pick starting index for erasure
#         index_min = erased_data.index.values[0]
#         index_max = erased_data.index.values[-1]
#
#         # Pick starting column for erasure (must be x dim column)
#         cols_min = erased_data.columns[0]
#         cols_max = erased_data.columns[-1]
#         start_cols = [i for i in range(cols_min, cols_max, 3)]
#
#         previous_col = 0
#         update = 0.6
#         for erasure in range(np.random.randint(1, max_erasures)):
#             if erasure == 0:
#                 probabilities = [1 / len(start_cols) for i in start_cols]
#             else:
#                 probabilities = [(1 - update) / (len(start_cols) - 1) for i in start_cols]
#                 probabilities[int(previous_col / 3)] = update
#             start_row = np.random.randint(index_min, index_max)
#             start_col = np.random.choice(start_cols, p=probabilities, replace=True)
#             erase_len = np.random.randint(1, max_gap_size)
#             for i in range(erase_len):
#                 for j in range(3):
#                     if start_row + i <= index_max:
#                         erased_data.iat[start_row + i, start_col + j] = np.nan
#                     elif i == index_max:
#                         continue
#             previous_col = start_col
#
#         file_name = str(file)[-12:]
#
#         erased_data.to_csv(erased_data_dir + file_name + 'GAPS' + '.csv')