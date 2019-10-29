import pandas as pd
import torch
import os
from pathlib import Path
from random import shuffle


def load_data(gap_data_dir, ground_truth_dir):
    # Input (gapped data)
    # get a list of strings of the input data file names (just names not whole path + extension)
    gap_data_names = os.listdir(gap_data_dir)
    gap_data_names.sort()

    # Output (full data)
    # List of torch tensors of ground truth
    directory = Path(ground_truth_dir)
    ground_truth_files = [p for p in directory.iterdir() if p.is_file()]
    ground_truth = []
    for file in ground_truth_files:
        ground_truth_data = pd.read_csv(file, index_col=0)
        ground_truth_data = ground_truth_data.drop(ground_truth_data.columns[30], axis=1)
        ground_truth_data = torch.Tensor(ground_truth_data.values)
        # ground_truth_data = ground_truth_data[:700]
        ground_truth.append(ground_truth_data)

    x_y_pairs = list(zip(gap_data_names, ground_truth))
    shuffle(x_y_pairs)

    training_pairs = x_y_pairs[:int(0.8*len(x_y_pairs))]
    test_pairs = x_y_pairs[int(0.8*len(x_y_pairs)):]

    train_gap_data_names = list(list(zip(*training_pairs))[0])
    test_gap_data_names = list(list(zip(*test_pairs))[0])

    partition = {'train': train_gap_data_names, 'test': test_gap_data_names}
    ground_truth = dict(x_y_pairs)

    training_set_size = len([str(p) for p in ground_truth_files])

    return partition, ground_truth, training_set_size
