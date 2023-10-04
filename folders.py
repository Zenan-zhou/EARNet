import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from data_tranform import CropPatches
import torch
from tqdm import tqdm


class LIVEFolder(data.Dataset):

    def __init__(self, index, train):

        im_dir = "/project/root/dataset/LIVE/"
        im_names = []
        for line1 in open("./data/live/im_names.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)

        ref_ids = []
        for line0 in open("./data/live/ref_ids.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)

        mos = []
        for line5 in open("./data/live/mos.txt", "r"):
            line5 = float(line5.strip())
            mos.append(line5 / 100)
        mos = np.array(mos)

        self.train = train
        self.patch_size = 112
        self.stride = 80
        self.CropPatches = CropPatches(self.patch_size, self.stride)
        self.patches = ()
        self.label = []
        self.index = []

        # get index
        for i in range(len(ref_ids)):
            if ref_ids[i] in index:
                self.index.append(i)

        # get image info
        if train:
            print("processing the training data...")
        else:
            print("processing the testing data...")
        for item in tqdm(self.index):
            im = pil_loader(os.path.join(im_dir, im_names[item]))
            patches = self.CropPatches(im)

            if train:
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(mos[item])
            else:
                self.patches = self.patches + (torch.stack(patches),)
                self.label.append(mos[item])

    def __getitem__(self, item):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return self.patches[item], torch.Tensor([self.label[item]])

    def __len__(self):
        length = len(self.patches)
        return length


class CSIQFolder(data.Dataset):

    def __init__(self, index, train):

        im_dir = "/project/root/dataset/CSIQ/"
        im_names = []
        for line1 in open("./data/csiq/im_names.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)

        ref_ids = []
        for line0 in open("./data/csiq/ref_ids.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)

        mos = []
        for line5 in open("./data/csiq/mos.txt", "r"):
            line5 = float(line5.strip())
            mos.append(line5)
        mos = np.array(mos)

        self.train = train
        self.patch_size = 112
        self.stride = 80
        self.CropPatches = CropPatches(self.patch_size, self.stride)
        self.patches = ()
        self.label = []
        self.index = []

        # get index
        for i in range(len(ref_ids)):
            if ref_ids[i] in index:
                self.index.append(i)

        # get image info
        for item in tqdm(self.index):
            im = pil_loader(os.path.join(im_dir, im_names[item]))
            patches = self.CropPatches(im)

            if train:
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(mos[item])
            else:
                self.patches = self.patches + (torch.stack(patches),)
                self.label.append(mos[item])

    def __getitem__(self, item):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return self.patches[item], torch.Tensor([self.label[item]])

    def __len__(self):
        length = len(self.patches)
        return length


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
