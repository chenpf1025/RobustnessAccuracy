# from __future__ import print_function
from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import zipfile

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, makedir_exist_ok, verify_str_arg, check_integrity

class DATASET_CUSTOM(VisionDataset):

    def __init__(self, root, data, targets, transform=None, target_transform=None):
        super(DATASET_CUSTOM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data, self.targets = data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.data)

class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, use_noisy_val=False):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)


        if not use_noisy_val: # benchmark setting
            if mode=='train':
                flist = os.path.join(root, "annotations/noisy_train.txt")
            if mode=='val':
                flist = os.path.join(root, "annotations/clean_val.txt")
            if mode=='test':
                flist = os.path.join(root, "annotations/clean_test.txt")
        else: # using a nnoisy validation setting, saving clean labels for training
            if mode=='train':
                flist = os.path.join(root, "noisy_val_annotations/nv_noisy_train.txt")
            if mode=='val':
                flist = os.path.join(root, "noisy_val_annotations/nv_noisy_val.txt")
            if mode=='test':
                flist = os.path.join(root, "noisy_val_annotations/nv_clean_test.txt")

        self.impaths, self.targets = self.flist_reader(flist)

        # # for debug
        # if mode=='train':
        #     self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]


    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.targets[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.impaths)

    def flist_reader(self, flist):
        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impaths.append(self.root + '/' + row[0])
                targets.append(int(row[1]))
        return impaths, targets
