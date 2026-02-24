import os
import pickle
from tqdm import tqdm

def read_outs(path):
    files = os.listdir(path)
    files = [i for i in files if i.endswith('.pkl')]
    file_name = []
    for file in tqdm(files):
        file_name.append(file)
        with open(path + f'{file}', 'rb') as f:
            cell_data = pickle.load(f)

            for keys, values in cell_data.items():
                if keys == 'cycle_data':
                    for k, v in values[0].items():# values[0] means only print the key and value for the first cycle
                        print(k, v)
                else:
                    print(keys, values)
        break # break for only reading out the first battery for this dataset

if __name__ == '__main__':
    read_outs(path='./path/to/your/folder')