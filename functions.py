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
from scipy.signal import savgol_filter


def filter_batch(data_dir='', new_dir=''):
    """
    Filters a directory of numpy arrays converting them to torch.tensors and filtering them using a savitsky golay
    filter
    :param data_dir: path to data files
    :param new_dir: path where filtered tensors are to be saved
    """
    directory = Path(data_dir)
    files = [p for p in directory.iterdir() if p.is_file() and str(p).endswith('.npy')]
    files.sort()

    n = 1
    for file in files:
        data = np.load(file)
        data = torch.as_tensor(data, dtype=torch.float)
        filtered_tensor = filter_tensor(data, window_length=31, polyorder=5)
        torch.save(filtered_tensor, new_dir + 'sav_filt_tensor_' + str(n) + '.pt')
        n += 1


def filter_tensor(tensor, window_length=201, polyorder=5):
    """
    Returns a filtered version of a time series tensor using a Savitsky-Golay. Each column of each sub tensor
    is filtered in turn using the window_length and polyorder params.
    :param tensor: input time_series tensor
    :param window_length: size of moving window
    :param polyorder: order of polynomial used to fit the window of data
    :return:
    """
    assert isinstance(tensor, torch.Tensor)

    tensor_copy = tensor.clone()
    filtered = torch.Tensor()
    if len(tensor.size()) == 3:
        for i in range(tensor.size(0)):
            filtered_sub_tensor = torch.Tensor()
            for j in range(tensor.size(2)):
                filtered_col = savgol_filter(tensor_copy[i][:, j], window_length=window_length, polyorder=polyorder)
                filtered_col = torch.Tensor(filtered_col)
                filtered_sub_tensor = torch.cat((filtered_sub_tensor, filtered_col.unsqueeze(0)), axis=0)
            filtered_sub_tensor = torch.transpose(filtered_sub_tensor, 1, 0)
            filtered = torch.cat((filtered, filtered_sub_tensor.unsqueeze(0)), axis=0)

    elif len(tensor.size()) == 2:
        filtered = torch.Tensor()
        for j in range(tensor.size(1)):
            filtered_col = savgol_filter(tensor_copy[:, j], window_length=window_length, polyorder=polyorder)
            filtered_col = torch.Tensor(filtered_col)
            filtered = torch.cat((filtered, filtered_col.unsqueeze(0)), axis=0)
        filtered = torch.transpose(filtered, 1, 0)
    return filtered


def load_data(ground_truth_dir, type='train'):
    """
    Loads motion capture data from a given directory and splits it into training and test datasets indexed with a
    dictionary.
    :param ground_truth_dir: Path to motion capture dataset
    :param type: The type of set you are loading: 'trian', 'test', or 'val'
    :return: Returns a dictionary that maps train and test keys to lists of filenames of each element. Also return the
    size of the training set.
    """

    if type == 'train':
        names = []
        for i in range(1, 35):   # max 35
            ground_truth_names = os.listdir(ground_truth_dir + str(i))
            ground_truth_names = [str(i) + '/' + k for k in ground_truth_names]
            ground_truth_names.sort()
            shuffle(ground_truth_names)
            names.append(ground_truth_names)

        names = [i for sublist in names for i in sublist]
        partition = {'train': names}

    elif type == 'test':
        names = []
        ground_truth_names = os.listdir(ground_truth_dir)
        ground_truth_names.sort()
        shuffle(ground_truth_names)
        names.append(ground_truth_names)
        names = [i for sublist in names for i in sublist]
        partition = {'test': names}
    #
    # elif type == 'val':
    #     names = []
    #     ground_truth_names = os.listdir(ground_truth_dir)
    #     ground_truth_names.sort()
    #     shuffle(ground_truth_names)
    #     names.append(ground_truth_names)
    #     partition = {'val': names}

    set_size = len(names)

    return partition, set_size


def crop_data(path_gt, new_gt_dir, padding=5, file_size=64):
    """
    Given a path to a dataset of motion capture files in .csv format, this function will split the files into a
    larger data set by taking slices of length 'file_size' through a moving window with a given
    padding size between each slice.
    :param path_gt: Path to ground truth data (with no missing markers)
    :param new_gt_dir: Path to where the new dataset of slices will be stored.
    :param padding: How many time steps to move the moving window after taking a slice.
    (it's better if this is a smaller number).
    :param file_size: The number of time steps to take in each slice. Rough guide for LSTM training suggests 150-250
    time steps to avoid problems during training.
    """
    assert isinstance(path_gt, str)
    assert isinstance(new_gt_dir, str)

    directory_gt = Path(path_gt)

    files_gt = [p for p in directory_gt.iterdir() if p.is_file() and str(p).endswith('.csv')]

    files_gt.sort()

    num_counter = 0

    file_num = 1

    i = 1
    for file in files_gt:
        data = np.genfromtxt(file, delimiter=',')
        data = data[:(data.shape[0] - (data.shape[0] % padding))]
        for j in range(0, data.shape[0] - file_size, padding):

            new_file = data[j:j + file_size]
            if i < 10:
                np.save(new_gt_dir + '/' + 'sliced_' + str(file_num) + '/CGT_000000' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1
            elif i < 100:
                np.save(new_gt_dir + '/' +'sliced_' + str(file_num) + '/CGT_00000' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1
            elif i < 1000:
                np.save(new_gt_dir + '/' +'sliced_' + str(file_num) + '/CGT_0000' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1
            elif i < 10000:
                np.save(new_gt_dir + '/' +'sliced_' + str(file_num) + '/CGT_000' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1
            elif i < 100000:
                np.save(new_gt_dir + '/' +'sliced_' + str(file_num) + '/CGT_00' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1
            elif i < 1000000:
                np.save(new_gt_dir + '/' +'sliced_' + str(file_num) + '/CGT_0' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1
            else:
                np.save(new_gt_dir + '/' +'sliced_' + str(file_num) + '/CGT_' + str(i) + '.npy', arr=new_file)
                i += 1
                num_counter += 1

            if num_counter == 10000:
                file_num += 1
                num_counter = 0


def remove_nan_files(path_gt):
    """
    Given a path to a dataset of motion capture data, this function will remove any files that containing files with
    missing data
    :param path_gt: Path to dataset
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


def apply_missing(time_series_tensor, max_erasures, max_gap_size, missing_val=None):
    """
    Given a time series tensor (torch.tensor) will simulate missing markers by replacing values in the tensor with missing_val (this is
    suggested to be something like np.nan or 0.0000).
    :param time_series_tensor: torch.Tensor containing time series data.
    :param max_erasures: The maximum number of erasures to place into the data i.e. how many times the marker "drops
    out"
    :param max_gap_size: maximum number of samples to take out for each erasure. The length of the erasure will be
    randomized between 1 and this number.
    :param missing_val: whatvalue to use to represent missing values (recommend one of {np.nan, 0.0000})
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
                if start_cols == [0]:
                    probabilities = [1]
                else:
                    probabilities = [1 / len(start_cols) for i in start_cols]
            else:
                if start_cols == [0]:
                    probabilities = [1]
                else:
                    probabilities = [(1 - update) / (len(start_cols) - 1) for i in start_cols]
                probabilities[int(previous_col / 3)] = update
            start_row = np.random.randint(index_min, index_max)
            start_col = np.random.choice(start_cols, p=probabilities, replace=True)
            erase_len = np.random.randint(1, max_gap_size)
            for i in range(erase_len):
                if start_row + i < index_max:
                    for j in range(3):
                        erased_data[start_row + i][start_col + j] = missing_val
                elif start_row + i == index_max:
                    pass
            previous_col = start_col
        erased_data = erased_data.unsqueeze(0)
        missing_array = torch.cat((missing_array, erased_data))

    return missing_array


def apply_missing_cmu_val(time_series_tensor, erasures_perc=None, missing_val=None):
    """
    Given a time series tensor (torch.tensor) will simulate missing markers by replacing values in the tensor with missing_val (this is
    suggested to be something like np.nan or 0.0000).
    :param time_series_tensor: torch.Tensor containing time series data.
    :param max_erasures: The maximum number of erasures to place into the data i.e. how many times the marker "drops
    out"
    :param max_gap_size: maximum number of samples to take out for each erasure. The length of the erasure will be
    randomized between 1 and this number.
    :param missing_val: whatvalue to use to represent missing values (recommend one of {np.nan, 0.0000})
    :param erasures_perc: percentageof markers to experience drop out
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
        index_max = erased_data.size(0)

        # Pick starting column for erasure (must be x dim column)
        cols_min = 0
        cols_max = erased_data.size(1)
        start_cols = [i for i in range(cols_min, cols_max, 3)]

        num_markers = round((erasures_perc / 100) * len(start_cols))

        markers_to_erase = np.random.choice(start_cols, size=num_markers, replace=False)

        max_erasure_len = 0
        # this loop simulates missing markers by erasing x,y, and z data from given markers
        for erasure in markers_to_erase:
            erase_len = abs(int(np.random.normal(10, 5)))
            if erase_len > max_erasure_len:
                max_erasure_len = erase_len
            start_row = np.random.randint(0, index_max-erase_len)
            start_col = erasure
            for i in range(erase_len):
                for j in range(3):
                    erased_data[start_row + i][start_col + j] = missing_val
        erased_data = erased_data.unsqueeze(0)
        missing_array = torch.cat((missing_array, erased_data))

    return missing_array, max_erasure_len


def apply_missing_fixed(time_series_tensor, gap_duration=50, num_markers=5, missing_val=None):
    """
    Applies a determined length of missing data to a time series dataframe. Used to compare to state of the art
    metrics.
    :param time_series_tensor: a torch.tensor of time series data
    :param gap_duration: how long the gaps in the data should be
    :param num_markers: how many markers you want to have missing data
    :param missing_val: what value the missing data should take (e.g. NaN, 0.0000 etc)
    :return: torch.tensor with applied missing data.
    """
    assert isinstance(time_series_tensor, torch.Tensor)

    missing_idx = []

    if len(time_series_tensor.size()) < 3:
        time_series_tensor_copy = time_series_tensor.clone()
        time_series_tensor_copy = time_series_tensor_copy.unsqueeze(0)
    else:
        time_series_tensor_copy = time_series_tensor.clone()

    missing_array = torch.Tensor()
    for k in range(time_series_tensor_copy.size(0)):
        erased_data = time_series_tensor_copy[k]

        # Pick starting index for erasure
        index_max = erased_data.size(0)

        # Pick starting column for erasure (must be x dim column)
        cols_min = 0
        cols_max = erased_data.size(1)
        start_cols = [i for i in range(cols_min, cols_max, 3)]

        markers_to_erase = np.random.choice(start_cols, size=num_markers, replace=False)

        # this loop simulates missing markers by erasing x,y, and z data from given markers
        for erasure in markers_to_erase:
            start_row = np.random.randint(0, index_max - gap_duration)
            start_col = erasure
            for i in range(gap_duration):
                for j in range(3):
                    erased_data[start_row + i][start_col + j] = missing_val
                    missing_idx.append((start_row+i, start_col+j))
        erased_data = erased_data.unsqueeze(0)
        missing_array = torch.cat((missing_array, erased_data))

    return missing_array, missing_idx


def split(data, chunk_size=None):
    """
    Given a motion capture data file as a torch tensor this function will split the files into a number of subfiles
    of a given size. This should be used to provide unseen data with missing markers to a trained network to fill in
    the gaps. Split will split the data up into the appropriate sized chucks for the trained network, which can then
    be reassembled by the merge function.
    :param data: torch tensor of data
    :param chunk_size: Size of the slices the data will be split into. This should equal the length of data that the
    network was trained on.
    :return: Tuple of (Copy of the original data, A torch.tensor of the slices of shape (num_slices, num_time_steps,
    num_features)) where num_features should be the number of markers used * the number of observed dimensions for
    each marker
    (x,y,z etc).
    """
    data_copy = data.clone()
    data_copy = data_copy[:(data_copy.size(0) - (data_copy.size(0) % chunk_size))]
    print('Original data shape: ' + str(data_copy.size()))
    sub_lists = torch.Tensor()
    for j in range(0, data_copy.size(0) - chunk_size + 1, chunk_size):
        data_slice = data_copy[j:j + chunk_size]
        data_slice = data_slice.unsqueeze(0)
        sub_lists = torch.cat((sub_lists, data_slice))
    data_copy[torch.isnan(data_copy)] = 0.0000
    sub_lists[torch.isnan(sub_lists)] = 0.0000
    return data_copy.float(), sub_lists.float()


def merge(sub_lists):
    """
    Given an torch Tensor of sublists of a data file (should be given by the split function) will rejoin said sublists
    into the original array. This should be used after feeding the slices of the datafile into the network to create
    an array of estimated data of the same size of the file you want to gap fill.
    :param sub_lists: np.array of slices of a given data file.
    :return: a merged torch.Tensor of the given split data
    """
    sub_lists = sub_lists.clone()
    super_list = torch.Tensor()
    for i in range(sub_lists.size(0)):
        array = sub_lists[i]
        super_list = torch.cat((super_list, array))
    print('Merged data shape: ' + str(super_list.size()))
    return super_list.float()


def gap_fill(original, estimated):
    """
    Given an original data file with missing markers and an estimated version of that file (given by a trained network)
    this function will fill in the gaps of the original with the estimates of those values given by the network.
    :param original: np.array of original data with missing markers.
    :param estimated: np.array of the same data estimated by the neural network.
    :return: torch.tensor of the original data but with the missing marker data points filled in with the network's
    estimates.
    """
    if type(original) == torch.Tensor:
        original = original.detach().numpy()
    gap_filled = original.copy()
    idx = np.argwhere(np.isnan(gap_filled))
    for i in range(len(idx)):
        gap_filled[idx[i][0]][idx[i][1]] = estimated[idx[i][0]][idx[i][1]]
    return torch.as_tensor(gap_filled).float()


def find_mean_pose(data_directory='', num_markers=None):
    """
    Given a path toa  driectory of motion capture data (with same shape) will return the mean pose over
    the data
    :param data_directory: Path to data
    :param num_markers: Number of markers used for capture (this should be the same in each file)
    :return: torch.Tensor of mean pose.
    """

    directory = Path(data_directory)
    files = [p for p in directory.iterdir() if p.is_file() and str(p).endswith('.csv')]
    files.sort()

    mean_poses = np.empty((0, num_markers*3))

    for file in files:
        data = np.genfromtxt(file, delimiter=',')
        data = data[1:]

        mean = np.mean(data, axis=0)

        mean_poses = np.append(mean_poses, np.expand_dims(mean, axis=0), axis=0)

    return torch.as_tensor(np.mean(mean_poses, axis=0))


def csv_to_tensor(data_dir='', new_data_dir=''):
    """
    Converts a directrory of csv files into torch.tensors
    :param data_dir: path to data directory
    :param new_data_dir: path to where the new tensors will be stored
    """

    directory = Path(data_dir)
    files = [p for p in directory.iterdir() if p.is_file() and not str(p).endswith('tore')]

    for file in files:
        name = str(file)[-10:-4]
        if name[0] == '/':
            name = name[1:]
        data = np.genfromtxt(file, delimiter=',')
        data = data[1:]
        data = torch.as_tensor(data)
        data = filter_tensor(data, window_length=47, polyorder=5)
        torch.save(data, new_data_dir + name + '.pt')


def find_tensor_max(path=''):
    """
    Given a path to directory containing a dataset of motion capture data will return the maximum value from the
    whole dataset.
    :param path: Path to dataset.
    :param data_type: Type of data to be processed.
    :return: Largest value from dataset as a torch tensor.
    """
    directory = Path(path)
    files = [p for p in directory.iterdir() if p.is_file() and not str(p).endswith('tore')]

    max_val = []
    for file in files:
        data = torch.load(file)
        max = torch.max(data)
        max_val.append(max)
    return torch.max(torch.as_tensor(max_val))


def find_max(path=''):
    """
    Given a path to a directory of time series data of the same shape in the first dimension
    (i.e. same number of cols) will return the max value of that dataset. This can be used for
    normalizing the data.
    :param path: Path to dataset
    :return: Maximum value from dataset as dtype: float.64.
    """

    directory = Path(path)
    files = [p for p in directory.iterdir() if p.is_file() and str(p).endswith('.csv')]

    max = 0

    for file in files:
        data = np.genfromtxt(file, delimiter=',')
        data_max = np.nanmax(data)
        if data_max > max:
            max = data_max
        else:
            continue

    return max


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
    for i in range(time_series_tensor_copy.size(0)):
        time_series_tensor_copy[i] -= mean_pose
    time_series_tensor_copy = time_series_tensor_copy / data_max_val
    return time_series_tensor_copy


def denormalize_series(time_series_tensor, mean_pose, data_max_val):
    """
    Denormalizes a time series from -1, 1 range back into CMU values.
    :param time_series_tensor: torch.tensor of noramlized time series data
    :param mean_pose: mean pose from non-normalized dataset
    :param data_max_val: maximum value from non-normalized dataset
    :return:
    """
    time_series_tensor_copy = time_series_tensor.clone()
    time_series_tensor_copy = time_series_tensor_copy * data_max_val
    for i in range(time_series_tensor_copy.size(0)):
        time_series_tensor_copy[i] += mean_pose
    return time_series_tensor_copy


def translate_to_marker(data, origin_marker=21):
    """
    Given a numpy array of marker data will return that marker data with all makers
    translated to a origin maker.
    :param data: 2D NParray containing your timeseries data
    :param origin_marker: The marker that will become the origin
    :return: Data translated to new origin given by origin_marker.
    """

    center_x_idx = 3 * (origin_marker - 1)

    for row in range(data.shape[0]):
        origin = data[row][center_x_idx:center_x_idx + 3].copy()
        for m in range(0, 123, 3):
            data[row][m:m + 3] -= origin

    return data


def crop_to_missing(local_batch, local_batch_missing, outputs, nan_val=None):
    """
    Crops elements from ground truth arrays that were made to be missing and crops the same elements
    from the estimated array to create two n length vectors where n is the number of missing data points.
    These two vectors can then be used to more accurate cost calculations
    :param local_batch: batch of ground truth tensors.
    :param local_batch_missing: batch of local_batch tensors with missing markers applied
    :param outputs: batch of output tensors from the neural network.
    :param nan_val: what numerical value NaN has been assigned in the data.
    :return: y, y_hat where y is ground truth values of the missing marker elements and y_hat is
    the estimated values.
    """

    idx = torch.where(local_batch_missing == nan_val)
    idxs = list(zip(idx[0], idx[1], idx[2]))
    idxs = (list(i) for i in idxs)

    y = torch.Tensor().cuda()
    y_hat = torch.Tensor().cuda()

    for i, j, k in idxs:
        val_x = local_batch[[i, j, k]]
        val_y = outputs[[i, j, k]]

        y = torch.cat((y, val_x.unsqueeze(0)))
        y_hat = torch.cat((y_hat, val_y.unsqueeze(0)))

    return y, y_hat
