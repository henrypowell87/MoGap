from pathlib import Path
import numpy as np
import torch
from functions import normalize_series

file_type_to_remove = '.csv'

path = '/run/media/henryp/Henry\'s HDD/DataSets/CMU/Formatted/Test/'
# subjects_dir = Path(path)
# subjects = [p for p in subjects_dir.iterdir() if p.is_dir()]
# subjects.sort()

# for subject in subjects:
subject_path = str(path)
subject_dir = Path(subject_path)
motions = [str(p) for p in subject_dir.iterdir() if p.is_file() and str(p).endswith(file_type_to_remove)]
motions.sort()

for motion in motions:

    file_name = str(motion)[-10:-4]
    if file_name[0] == '/':
        file_name = file_name[1:]

    data = np.genfromtxt(motion, delimiter=',')
    mean_pose = torch.as_tensor(np.load('./mean_pose.npy'))
    max_val = torch.as_tensor(np.load('./data_max.npy'))

    data = torch.as_tensor(data)
    data = normalize_series(data, mean_pose=mean_pose, data_max_val=max_val)
    data = np.array(data)

    np.savetxt('/run/media/henryp/Henry\'s HDD/DataSets/CMU/Formatted/Test/' + file_name + '.csv',
               X=data,
               delimiter=',')
