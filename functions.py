"""
Author: Henry Powell
Institution: Institute of Neuroscience and Psychology, Glasgow University, Scotland.

Python file of functions that are used in the preprocessing of data used to train the autoencoder.
"""

import pandas as pd
import torch
import os
import numpy as np
from pathlib import Path
from random import shuffle


def load_data(ground_truth_dir):
    """
    Loads motion capture data from a given directory and splits it into training and test datasets indexed with a
    dictionary.
    :param ground_truth_dir: Path to motion capture dataset
    :return: Returns a dictionary that maps train and test keys to lists of filenames of each element. Also return the
    size of the training set.
    """
    ground_truth_names = os.listdir(ground_truth_dir)
    ground_truth_names.sort()

    shuffle(ground_truth_names)

    idx = int(0.8*len(ground_truth_names))
    train_names = ground_truth_names[:idx]
    test_names = ground_truth_names[idx:]

    partition = {'train': train_names, 'test': test_names}

    training_set_size = len(ground_truth_names)

    return partition, training_set_size


def crop_data(path_gt, new_gt_dir, padding=5, file_size=30):
    """
    Given a path to a dataset of motion capture files, this function will split the files into a larger data set by
    taking slices of length 'file_size' through a moving window with a given padding size between each slice.
    :param path_gt: Path to ground truth data (with no missing markers)
    :param new_gt_dir: Path to where the new dataset of slices will be stored.
    :param padding: How many time steps to move the moving window after taking a slice.
    (it's better if this is a smaller number).
    :param file_size: The number of time steps to take in each slice. Rough guide for LSTM training suggests 150-250
    time steps to avoid problems during training.
    :return:
    """
    assert isinstance(path_gt, str)
    assert isinstance(new_gt_dir, str)

    directory_gt = Path(path_gt)

    files_gt = [p for p in directory_gt.iterdir() if p.is_file() and not str(p).endswith('.DS_Store')]

    files_gt.sort()

    i = 1
    for file in files_gt:
        data = pd.read_csv(file, index_col=[0])
        if data.shape[1] == 31:
            data = data.drop(data.columns[30], axis=1)
        data = data[:(data.shape[0] - (data.shape[0] % padding))]
        for j in range(0, data.shape[0] - file_size, padding):
            new_file = data[j:j + file_size]
            if i < 10:
                np.save(new_gt_dir + '/CGT_0000' + str(i), new_file)
                i += 1
            elif i < 100:
                np.save(new_gt_dir + '/CGT_000' + str(i), new_file)
                i += 1
            elif i < 1000:
                np.save(new_gt_dir + '/CGT_00' + str(i), new_file)
                i += 1
            elif i < 10000:
                np.save(new_gt_dir + '/CGT_0' + str(i), new_file)
                i += 1
            else:
                np.save(new_gt_dir + '/CGT_' + str(i), new_file)
                i += 1


def remove_nan_files(path_gt):
    """
    Given a path to a dataset of motion capture data, this function will remove any files that containing files with
    missing data
    :param path_gt: Path to dataset
    :return:n/a
    """
    assert isinstance(path_gt, str)

    directory_gt = Path(path_gt)

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

    for i in bad_files:
        bad_files_gt.append([str(p) for p in directory_gt.iterdir() if p.is_file() and i in str(p)])

    bad_files_gt = [i for sublist in bad_files_gt for i in sublist]

    for i in bad_files_gt:
        os.remove(i)


def apply_missing(time_series_tensor, max_erasures, max_gap_size, missing_val=0.0000):
    """
    Given a time series tensor will simulate missing markers by replacing values in the tensor with missing_val (this is
    suggested to be something like np.nan or 0.0000).
    :param time_series_tensor: torch.Tensor containing time series data.
    :param max_erasures: The maximum number of erasures to place into the data i.e. how many times the marker "drops
    out"
    :param max_gap_size: maximum number of samples to take out for each erasure. The length of the erasure will be
    randomized between 1 and this number.
    :return: torch.Tensor of the original array with the applied missing data.
    """
    assert isinstance(time_series_tensor, torch.Tensor)

    if len(time_series_tensor.size()) < 3:
        time_series_tensor_copy = time_series_tensor.clone()
        time_series_tensor_copy = time_series_tensor_copy.unsqueeze(0)
    else:
        time_series_tensor_copy = time_series_tensor.clone()

    time_series_tensor_copy = time_series_tensor_copy
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

        # this loop simulates a faulty marker by giving each marker an equal probability to drop out and then updating
        # the probability that the chosen marker will drop out a second time
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
                        erased_data[start_row + i][start_col + j] = missing_val
                    elif i == index_max:
                        continue
            previous_col = start_col
        erased_data = erased_data.unsqueeze(0)
        missing_array = torch.cat((missing_array, erased_data))

    return missing_array


def nan_to_zero(tensor):
    """
    Given a torch.Tensor containing nan vaules, will convert all of those nan values to 0.0000.
    :param tensor: torch.Tensor containg nan data.
    :return: a copy of the original tensor with nans replaced by 0.000
    """
    assert isinstance(tensor, torch.Tensor)

    tensor_copy = tensor.clone()
    tensor_copy = tensor_copy.float()
    tensor_copy[torch.isnan(tensor_copy)] = 0.0000
    return tensor_copy


def split(data, index_cols=True, size=30, padding=5, result_dim=(0, 30, 30)):
    """
    Given a motion capture data file this function will split the files into a number of subfiles of a given size. This
    should be used to provide unseen data with missing markers to a trained network to fill in the gaps. Split will
    split the data up into the appropriate sized chucks for the trained network, which can then be reassembled by
    the merge function.
    :param data: path to datafile
    :param index_cols: set to true if the first row and column of you data are row and column indeces.
    :param size: Size of the slices the data will be split into. This should equal the length of data that the network
    was trained on.
    :param padding: How many time steps to move the moving window after taking a slice.
    :param result_dim: the desired resulting dimension of you slice array. This should be of shape (0, num_time_steps,
    num_features). The zero in the first dimension is required to stack the arrays on top of one another.
    :return: Copy of the original data. A np.array of the slices of shape (num_slices, num_time_steps, num_features)
    where num_feautres should be the number of markers used * the number of observed dimensions for each marker
    (x,y,z etc).
    """
    data = np.genfromtxt(data, delimiter=',')
    data_copy = data.copy()
    if index_cols:
        data_copy = data_copy[1:][:, 1:]
    data_copy = data_copy[:(data_copy.shape[0] - (data_copy.shape[0] % padding))]
    print('Original data shape: ' + str(data_copy.shape))
    sub_lists = np.empty(result_dim)
    for j in range(0, data_copy.shape[0] - size + padding, padding):
        data_slice = data_copy[j:j + size]
        data_slice = np.expand_dims(data_slice, axis=0)
        sub_lists = np.append(sub_lists, data_slice, axis=0)
    data_copy[np.isnan(data_copy)] = 0.0000
    sub_lists[np.isnan(sub_lists)] = 0.0000
    return data_copy, sub_lists


def merge(sub_lists, padding=5, result_dim=(0, 30)):
    """
    Given an array of sublists of a data file (should be given by the split function) will rejoin said sublists
    into the original array. This should be used after feeding the slices of the datafile into the network to create
    an array of estimated data of the same size of the file you want to gap fill.
    :param sub_lists: np.array of slices of a given data file.
    :param padding: padding: How many time steps to move the moving window after taking a slice.
    :param result_dim: The resulting dimension of the output array. This should be of shape (0, num_features) where
    num_features is the number of markers * number of observed dimensions for each marker (x,y,z etc)
    :return: a merged np.array of the given split data
    """
    super_list = np.empty(result_dim)
    for i in range(sub_lists.shape[0]-1):
        array = sub_lists[i]
        data_slice = array[:padding]
        super_list = np.append(super_list, data_slice, axis=0)
    super_list = np.append(super_list, sub_lists[-1], axis=0)
    print('Merged data shape: ' + str(super_list.shape))
    return super_list


def gap_fill(original, estimated):
    """
    Given an original data file with missing markers and an estimated version of that file (given by a trained network)
    this function will fill in the gaps of the original with the estimates of those values given by the network.
    :param original: np.array of original data with missing markers.
    :param estimated: np.array of the same data estimated by the neural network.
    :return: np.array of the original data but with the missing marker data points filled in with the network's
    estimates.
    """
    gap_filled = original.copy()
    idx = np.argwhere(np.isnan(gap_filled))
    for i in range(len(idx)):
        gap_filled[idx[i][0]][idx[i][1]] = estimated[idx[i][0]][idx[i][1]]
    return gap_filled


def find_translated_mean_pose(num_markers=10, path='/home/henryp/PycharmProjects/MoGap/ground_truth_data',
                              central_marker_num=4):
    """
    Given a dataset of motion capture data will first translate all markers to a body-centered coordinate system
    by subtracting the central marker position from each marker at each timestep. Then finds the mean position over the
    data set by taking the mean of each marker over the whole dataset. These values can be used to noramlize the
    dataset for better reults during learning.
    :param num_markers: The number of markers used for data collection, this should be the same across all files.
    :param path: path to data set
    :param central_marker_num: the number of the desired marker used for translation. I.e. the number of the marker
    according to where that marker's data columns fall in the data set.
    :return: torch.Tensor with the mean pose across the data set.
    """
    # original data set path
    num_coords = num_markers * 3

    directory = Path(path)
    files = [p for p in directory.iterdir() if p.is_file() and not str(p).endswith('tore')]

    # keep track of markers for calculating mean pose
    markers = np.array([0.0 for i in range(num_coords)])

    for file in files:
        data = np.genfromtxt(file, delimiter=',')
        data = data[1:][:, 1:-1]

        center_x_idx = 3 * (central_marker_num - 1)
        # translate data to collar centered coordinate system
        for i in range(data.shape[0]):
            collar_coords = data[i][center_x_idx:center_x_idx + 3]
            for j in range(0, data.shape[1], 3):
                data[i][j:j + 3] -= collar_coords

        # calculate running sum of mean marker positions
        for k in range(data.shape[1]):
            markers[k] += np.mean(data[:, k])

    # divide by number of files to get mean pose
    mean_pose = markers / len(files)
    return torch.tensor(mean_pose)


def find_max_val(path='/home/henryp/PycharmProjects/MoGap/ground_truth_data'):
    """
    Given a path to directory containing a dataset of motion capture data will return the maximum value from the
    whole dataset.
    :param path: Path to dataset.
    :return: Largest value from dataset.
    """
    directory = Path(path)
    files = [p for p in directory.iterdir() if p.is_file() and not str(p).endswith('tore')]
    max_val = []
    for file in files:
        data = np.genfromtxt(file, delimiter=',')
        data = data[1:][1:-1]
        max = np.nanmax(data)
        max_val.append(max)
    return np.max(max_val)


def normalize_series(time_series_tensor, mean_pose, data_max_val):
    """
    Given a torch.Tensor of times series data will return the normalized version of the tensor in the range -1,1 given
    the mean pose of the dataset to which that tensor belongs and the largest value in the whole data set. These can be
    obtained using the find_translated_mean_pose and find_max_val function defined in this code base.
    :param time_series_tensor: torch.Tensor of time series data.
    :param mean_pose: torch.Tensor containing the mean pose
    :param data_max_val: max value from dataset
    :return:
    """
    assert isinstance(time_series_tensor, torch.Tensor)

    time_series_tensor_copy = time_series_tensor.clone()
    for i in range(time_series_tensor_copy.shape[0]):
        time_series_tensor_copy[i] -= mean_pose
    time_series_tensor_copy = time_series_tensor_copy / data_max_val
    return time_series_tensor_copy


crop_data(path_gt='/home/henryp/PycharmProjects/MoGap/ground_truth_data',
          new_gt_dir='/home/henryp/PycharmProjects/MoGap/cropped_ground_truth_data',
          padding=1, file_size=200)
