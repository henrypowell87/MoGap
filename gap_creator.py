import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 3000)

max_gap_size = 200
max_erasures = 10

path = '/home/henryp/Neurobots_data/NeuroBotsPilotAnalysis/A002'
erased_data_dir = '/home/henryp/PycharmProjects/MoGap/augmented_data/'

directory = Path(path)

files = [p for p in directory.iterdir() if p.is_file() and not str(p).endswith('.DS_Store')]

files.sort()

for file in files:
    raw_data = pd.read_csv(file, index_col=[0])
    erased_data = raw_data.drop(raw_data.columns[30], axis=1)
    cols = [i for i in range(erased_data.shape[1])]
    erased_data.columns = cols

    # Pick starting index for erasure
    index_min = erased_data.index.values[0]
    index_max = erased_data.index.values[-1]

    # Pick starting column for erasure (must be x dim column)
    cols_min = erased_data.columns[0]
    cols_max = erased_data.columns[-1]
    start_cols = [i for i in range(cols_min, cols_max, 3)]

    previous_col = 0
    update = 0.6
    for erasure in range(np.random.randint(max_erasures)):
        if erasure == 0:
            probabilities = [1/len(start_cols) for i in start_cols]
        else:
            probabilities = [(1-update)/(len(start_cols)-1) for i in start_cols]
            probabilities[int(previous_col/3)] = update
        start_row = np.random.randint(index_min, index_max)
        start_col = np.random.choice(start_cols, p=probabilities, replace=True)
        erase_len = np.random.randint(max_gap_size)
        for i in range(erase_len):
            for j in range(3):
                if start_row + i <= index_max:
                    erased_data.iat[start_row + i, start_col + j] = np.nan
                elif i == index_max:
                    continue
        previous_col = start_col

    file_name = str(file)[-12:]

    erased_data.to_csv(erased_data_dir + file_name + 'GAPS' + '.csv')


