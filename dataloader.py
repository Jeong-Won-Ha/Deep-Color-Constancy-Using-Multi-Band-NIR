import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torchvision.transforms import *
import torch
import math
from os.path import join
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import scipy
from scipy import io

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])
def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img

def resize_img():
    return torchvision.transforms.Resize([256,256])

def get_patch(img_in, nir_in, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in = nir_in[:,ix:ix+ip,iy:iy+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, nir_in

def get_patch_3(img_in, nir_in, nir_in_2, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in = nir_in[:,ix:ix+ip,iy:iy+ip]
    nir_in_2 = nir_in_2[:, ix:ix + ip, iy:iy + ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, nir_in, nir_in_2

def get_patch_2(img_in, nir_in, map_in, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size
    iter = 0
    while True:
        iter = iter + 1
        if ix == -1:
            ix = random.randrange(0, iw - ip + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip + 1)

        img_patch = img_in.crop((iy, ix, iy + ip, ix + ip))
        nir_patch = nir_in[:,ix:ix+ip,iy:iy+ip]
        map_patch = map_in[:,ix:ix+ip,iy:iy+ip]
        map_mean = torch.mean(map_patch)
        # print(map_mean)

        if map_mean > 10/255: #10
            break
        elif iter > 30:
            # print('no patch')
            # print(map_mean)
            break

    return img_patch, nir_patch, map_patch

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, gt_ilu, folder, Num_ch, isTrain, transform=None):
        super(DatasetFromFolder, self).__init__()

        self.Train = isTrain
        self.istrain = folder
        self.num = Num_ch
        self.im_dir = os.path.join(image_dir, self.istrain, 'bright_rgb/')
        self.nir_dir = os.path.join(image_dir, self.istrain, self.num)
        self.nir_dir_2 = os.path.join(image_dir, self.istrain, 'nir_1ch/')

        self.image_filenames = [join(self.im_dir, x) for x in listdir(self.im_dir) if is_image_file(x)]
        self.nir_filenames = [join(self.nir_dir, x) for x in listdir(self.nir_dir) if is_mat_file(x)]
        self.nir_filenames_2 = [join(self.nir_dir_2, x) for x in listdir(self.nir_dir_2) if is_mat_file(x)]

        self.patch_size = patch_size
        self.transform = transform
        self.gt_ilu = gt_ilu
        # self.resize = resize_img()

    def __getitem__(self, index):
        # self.train_data = self.im_dir
        input_im = load_img(self.image_filenames[index])
        nir_im = scipy.io.loadmat(self.nir_filenames[index])
        nir_im_2 = scipy.io.loadmat(self.nir_filenames_2[index])


        nir_im = nir_im['mat_data']
        nir_im = self.transform(nir_im)

        nir_im_2 = nir_im_2['mat_data']
        nir_im_2 = self.transform(nir_im_2)

        gt = self.gt_ilu[index]
        gt = torch.from_numpy(gt)

        if self.Train == True:

            input_patch, nir_patch, nir_patch_2 = get_patch_3(input_im, nir_im, nir_im_2, self.patch_size)
            if self.transform:
                input_patch = self.transform(input_patch)

            return input_patch, nir_patch, nir_patch_2,gt
        else:

            input_patch = input_im
            # input_patch = self.resize(input_im)
            nir_patch = nir_im
            nir_patch_2 = nir_im_2
            # _, nir_patch = get_patch(input_im, nir_im, self.patch_size)
            _, file = os.path.split(self.image_filenames[index])
            if self.transform:
                input_patch = self.transform(input_patch)
                # nir_patch = self.transform(nir_patch)[0,:,:].unsqueeze(0)
            return input_patch, nir_patch, nir_patch_2, gt, file

    def __len__(self):
        return len(self.image_filenames)
