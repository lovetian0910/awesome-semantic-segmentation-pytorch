import os
import torch
import numpy as np
from PIL import Image
from .segbase import SegmentationDataset
import cv2

class LiveSegmentation(SegmentationDataset):
    NUM_CLASS = 5
    def __init__(self, file_path = "live_more_", root='../QGameData/', split=' ', mode=None, transform=None, **kwargs):
        self.width = 256
        self.height = 256
        self.root_path = root
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        if split == 'val':
            split = 'test'
        f = open(root + file_path + split + ".txt", 'r')
        self.items = f.readlines()

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.root_path + item.split(' ')[0]
        label_path = self.root_path + item.split(' ')[-1].strip()
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 1)
        img = cv2.resize(img, (self.width, self.height))
        label_img = cv2.resize(label_img, (self.width, self.height))
        im = img
        lim = label_img
        lim = lim[:, :, 0]
        im = np.array(im).astype(np.float32)
        lim = np.array(lim).astype('int32')
        # im /= 255.0
        # im -= self.mean
        # im /= self.std
        im = np.transpose(im, [2, 0, 1])
        # torch_img = torch.from_numpy(im).float()
        mask = torch.from_numpy(lim).long()
        return im, mask, os.path.basename(image_path)

    def __len__(self):
        return len(self.items)
        # return 32
