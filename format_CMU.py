from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions import filter_csv, filter_tensor, find_translated_mean_pose
import os

file_type_to_remove = '.csv'

path = '/run/media/henryp/Henry\'s HDD/DataSets/CMU/Unformatted/Train/'
subjects_dir = Path(path)
subjects = [p for p in subjects_dir.iterdir() if p.is_dir()]
subjects.sort()

for subject in subjects:
    subject_path = str(subject)
    subject_dir = Path(subject_path)
    motions = [str(p) for p in subject_dir.iterdir() if p.is_file() and str(p).endswith(file_type_to_remove)]
    motions.sort()

    for motion in motions:

        file_name = str(motion)[-10:-4]
        if file_name[0] == '/':
            file_name = file_name[1:]

        central_marker = 21
        data = np.genfromtxt(motion, delimiter=',')
        data = data[1:]
        center_x_idx = 3 * (21 - 1)

        for row in range(data.shape[0]):
            origin = data[row][center_x_idx:center_x_idx+3].copy()
            for m in range(0, 123, 3):
                data[row][m:m+3] -= origin

        np.savetxt('/run/media/henryp/Henry\'s HDD/DataSets/CMU/Formatted/Train/' + file_name + '.csv',
                   X=data,
                   delimiter=',')