from torch.utils.data import Dataset
import torch
import os
import numpy as np
from image_utils import label2onehot
import nibabel as nib
import csv
import pandas as pd
from utils import age_transform
import random
import consts

class UKbiobank_40k_EDES_cycle(Dataset):

    def __init__(self, data_dir, txt_path, mode=None):
        imgs = []
        with open(txt_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                imgs.append((row['Unnamed: 0'], float(row['Age']), float(row['Sex'])))

        self.imgs = imgs

        self.data_dir = data_dir
        self.mode = mode

        #self.file_list = file_list

    def __getitem__(self, index):
        #subject_name = self.file_list[index]

        segID, age, sex = self.imgs[index]
        age_group = age_transform(age)
        ##generate a target age
        condition = True

        while condition:
            age_group_target = random.randint(0, consts.NUM_AGES - 1)
            condition = age_group == age_group_target

        data_dir = self.data_dir
        #time 1
        seg1 = {'ED': [],'ES': []}
        for phase in ['ED','ES']:
            filename = f'{data_dir}seg_edes_40k/{segID}/seg_sa_{phase}_SR.npy'
            labelmap = np.load(filename)
            seg_onehot = label2onehot(labelmap)
            seg1[phase]=seg_onehot
        seg = torch.tensor(np.concatenate((seg1['ED'],seg1['ES'])).astype(np.float32))
        return seg, age_group, age_group_target, sex, segID

    def __len__(self):
        return len(self.imgs)

