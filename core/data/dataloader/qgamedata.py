import os
from numpy.core.fromnumeric import shape
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .segbase import SegmentationDataset
import cv2

class LiveSegmentation(SegmentationDataset):
    NUM_CLASS = 5
    def __init__(self, file_path = "live_more_", root='../QGameData/', split=' ', mode=None, transform=None, **kwargs):
        super(LiveSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        self.root_path = root
        if split == 'val':
            split = 'test'
        f = open(root + file_path + split + ".txt", 'r')
        self.items = f.readlines()

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.root_path + item.split(' ')[0]
        label_path = self.root_path + item.split(' ')[-1].strip()
        img = Image.open(image_path).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(image_path)
        mask = Image.open(label_path)
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = mask[:,:,0]
        # self.process_mask(mask)
        return img, mask, os.path.basename(image_path)

    def __len__(self):
        return len(self.items)
        # return 5

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        # target[target > 0] = 1
        return torch.from_numpy(target).long()

    def _sync_transform(self, img, mask):
        img = img.resize((self.crop_size, self.crop_size), Image.NEAREST)
        mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
        img = self._img_transform(img)
        mask = self._mask_transform(mask)
        return img, mask

    def process_mask(self, mask):
        colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
        mask_img = np.zeros((len(mask), len(mask), 3))
        for x in range(len(mask)):
            for y in range(len(mask[x])):
                num_class = mask[x][y]
                mask_img[x][y] = colors[num_class]
        cv2.imshow("mask", mask_img)
        cv2.waitKey(8000)

if __name__ == "__main__":
    qgamedata = LiveSegmentation(crop_size = 512, split = "train")
    for i in range(qgamedata.__len__()):
        img, mask, _ = qgamedata.__getitem__(i)

    
