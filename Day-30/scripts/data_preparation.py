import os
import json
import pandas as pd
import torch

def load_data(data_path):
    data = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        item = json.load(f)
                        data.append(item)
    return pd.DataFrame(data)

train_data_path = 'MATH/train'
test_data_path = 'MATH/test'

train_df = load_data(train_data_path)
test_df = load_data(test_data_path)
