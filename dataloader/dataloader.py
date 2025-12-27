import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from dataset.dataset_details import user_details_norm, ACTIVITY_LABELS, ACTIVITY_DETAILS, USER_DETAILS
import numpy as np


class TrainDataset(Dataset):
    def __init__(self,
                 inputs,
                 outputs,
                 file_names,
                 input_motion_length=196,
                 train_dataset_repeat_times=1,
                 normalization=True,
                 debug=False,
                 with_context=False):
        

        assert len(inputs) == len(outputs)
        self.filename_list = file_names
        
        self.raw_x = inputs
        self.raw_y = outputs
        self.norm = normalization

        self.train_dataset_repeat_times = train_dataset_repeat_times if not debug else 1
        self.input_motion_length = input_motion_length
        self.with_context = with_context  

    def __len__(self):
        return len(self.raw_x) * self.train_dataset_repeat_times

    def __getitem__(self, idx):
        x = torch.from_numpy(self.raw_x[idx % len(self.raw_x)]).float()
        y = torch.from_numpy(self.raw_y[idx % len(self.raw_x)]).float()
        if True:
            mean_vals = x.mean(dim=1, keepdim=True)
            std_vals = x.std(dim=1, keepdim=True)
            x = (x - mean_vals) / std_vals
        
        seqlen = x.shape[0]

        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]


        x = x[idx: idx + self.input_motion_length] 
        y = y[idx: idx + self.input_motion_length]


        if self.with_context:
            #### retrive bio information and activity label
            filename = self.filename_list[idx % len(self.raw_x)]
            filename = os.path.basename(filename).split(".")[0]
            user = filename.split("v")[0]
            user_vector = torch.from_numpy(user_details_norm()[user]).float()
            activity = torch.tensor(ACTIVITY_LABELS[ACTIVITY_DETAILS[filename.split('_')[0]]], dtype=torch.long)
            return x , y, user_vector, activity 
        else:
            return x, y 


class TestDataset(Dataset):
    def __init__(self, inputs, outputs, filename_list, normalization=True, with_context=False):
        self.filename_list = filename_list
        self.raw_x = inputs
        self.raw_y = outputs
        self.norm = normalization
        self.with_context = with_context


    def __len__(self):
        return len(self.raw_x)

    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.raw_x[idx]).float()
        y = torch.from_numpy(self.raw_y[idx]).float()
        filename_raw = self.filename_list[idx]

        if self.with_context:
            filename = self.filename_list[idx]
            filename = os.path.basename(filename).split(".")[0]
            user = filename.split("v")[0]
            user_vector = torch.from_numpy(user_details_norm()[user]).float()
            # pressure = USER_DETAILS[user][0]

            activity = torch.tensor(ACTIVITY_LABELS[ACTIVITY_DETAILS[filename.split('_')[0]]], dtype=torch.long)

            return x, y, filename_raw, user_vector, activity
        else:
            return x, y, filename_raw



def get_data(data_list, input_dim=36, output_dim=8):

    def norm_data(data, val1, val2, method='minmax', feature_range=(0, 1)):

        if method == 'minmax':
            if val1 is None or val2 is None:
                val1 = np.min(data, axis=0)
                val2 = np.max(data, axis=0)

            scale = (feature_range[1] - feature_range[0]) / (val2 - val1 + 1e-8)
            return feature_range[0] + (data - val1) * scale


    ''' min-max norm '''
    inputs = [norm_data(df.iloc[:, -input_dim:].apply(pd.to_numeric, errors='raise').to_numpy(), None, None, 'minmax') for (df, _) in data_list]
    outputs = [norm_data(df.iloc[:, 1:1+output_dim].apply(pd.to_numeric, errors='raise').to_numpy(), None, None, 'minmax') for (df, _) in data_list]
    
    return inputs, outputs



def read_file_list(protocol, split):

    if os.path.exists(f"./dataset/protocols/{protocol}_{split}.txt") is False:
        print(f"==> {protocol}_{split}.txt does not exist, see dataset/create_split.py")
        return

    with open(f"./dataset/protocols/{protocol}_{split}.txt", "r") as f:
        files = f.readlines()

    return [f.strip() for f in files if os.path.exists(f.strip())]

def get_path(split, protocol):
    assert split in ["train", "test", "val"]
    
    files = read_file_list(protocol, split)
    if len(files) == 0:
        return

    print(f"==> {split} under {protocol} protocol")
    return files
    

def load_data(dataset_path, split, protocol, **kwargs):

    assert ("input_motion_length" in kwargs), "Please specify the input_motion_length"
   

    files = get_path(split, protocol)
    input_motion_length = kwargs["input_motion_length"]
    
    data_list = [(pd.read_csv(i),i) for i in tqdm(files)]

    feet, muscle = get_data(data_list)

    if split == "test" or split == "val":
        return feet, muscle, files
    

    new_feet = []
    new_muscle = []
    for idx, x in enumerate(feet):
        if x.shape[0] < input_motion_length :  # Arbitrary choice
            continue
        new_feet.append(feet[idx])
        new_muscle.append(muscle[idx])

    return new_feet, new_muscle, files 


def get_dataloader(dataset, split, batch_size, num_workers=32):
    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader
