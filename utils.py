import torch

from typing import Dict, Any

import omegaconf
import pandas as pd
import consts
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from pdb import set_trace as st


def age_transform(age):
    ##
    age -= 1
    if age < 50:
        # first 4 age groups are for kids <= 20, 5 years intervals
        return 0
    else:
        # last (6?) age groups are for adults > 50, 5 years intervals
        return min(1 + (age - 50) // 5, 7 - 1)

def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def two_sided(x):
    return 2 * (x - 0.5)

def add_noise_to_label(age,sex,normalize=True):
    nb = age.shape[0]
    noise_sigma = 0.2
    label_age = torch.Tensor(nb, consts.NUM_AGES)
    label_gender = -torch.ones(nb, consts.LABEL_LEN_EXPANDED-consts.NUM_AGES)
    for i in range(nb):
        label_age[i, :] = (noise_sigma * torch.randn(1, consts.NUM_AGES))
        label_age[i, int(age[i]) * 1:(int(age[i]) + 1) * 1] += 1

        if normalize:
            gender_tensor = -torch.ones(consts.NUM_GENDERS)
            gender_tensor[int(sex[i])] *= -1
            label_gender[i] = gender_tensor.repeat(consts.NUM_AGES // consts.NUM_GENDERS)

    result = torch.cat((label_age, label_gender), 1)
    return result

def generate_agegroups_label(gender):
    noise_sigma=0.2
    gender_tensor = -torch.ones(consts.NUM_GENDERS)
    gender_tensor[int(gender)] *= -1
    gender_tensor = gender_tensor.repeat(consts.NUM_AGES,
                                         consts.NUM_AGES // consts.NUM_GENDERS)  # apply gender on all images

    age_tensor = torch.Tensor(consts.NUM_AGES, consts.NUM_AGES)
    for i in range(consts.NUM_AGES):
        age_tensor[i, :] = (noise_sigma * torch.randn(1, consts.NUM_AGES))
        age_tensor[i][i] += 1
  # apply the i'th age group on the i'th image

    l = torch.cat((age_tensor, gender_tensor), 1)
    return l

def str_to_tensor(age, sex, normalize=False):
    age_tensor = -torch.ones(consts.NUM_AGES)
    age_tensor[int(age)] *= -1
    gender_tensor = -torch.ones(consts.NUM_GENDERS)
    gender_tensor[int(sex)] *= -1
    if normalize:
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES // consts.NUM_GENDERS)
    result = torch.cat((age_tensor, gender_tensor), 0)
    return result

def int_to_tensor_one_element(age, normalize=False):
    age_tensor = -torch.ones(7)
    age_tensor[int(age)] *= -1
    return age_tensor

def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.
    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension
    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param,) * ndim
    elif isinstance(param, (tuple, list, omegaconf.listconfig.ListConfig)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param


def save_dict_to_csv(d, csv_path, model_name='modelX'):
    for k, x in d.items():
        if not isinstance(x, list):
            d[k] = [x]
    pd.DataFrame(d, index=[model_name]).to_csv(csv_path)


def worker_init_fn(worker_id):
    """ Callback function passed to DataLoader to initialise the workers """
    # Randomly seed the workers
    random_seed = random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)



def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

def merge(file1, file2):
    # merge two txt file
    f3 = open('./template.txt','a+')
    with open(file1, 'r') as f1:
        for i in f1:
            f3.write(i)
    with open (file2, 'r') as f2:
        for i in f2:
            f3.write(i)
    return f3


def list_folder_images(dir, opt):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in os.listdir(dir):
        path = os.path.join(dir, fname)
        images.append(path)

    # sort according to identity in case of FGNET test
    if 'fgnet' in opt.dataroot.lower():
        images.sort(key=str.lower)


    return images

def get_transform(opt, normalize=True):
    transform_list = []

    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, interpolation=Image.NEAREST))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]

    if normalize:
        mean = (0.5,)
        std = (0.5,)
        transform_list += [transforms.Normalize(mean,std)]

    return transforms.Compose(transform_list)
