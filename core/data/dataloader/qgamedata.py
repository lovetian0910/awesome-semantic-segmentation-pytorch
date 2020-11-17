import os
from numpy.core.fromnumeric import shape
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .segbase import SegmentationDataset

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
        # if self.transform is not None:
        #     img = self.transform(img)
        input_tensor = transforms.ToTensor()
        img = input_tensor(img)
        mask = mask[:,:,0]
        return img, mask, os.path.basename(image_path)
        # img = cv2.imread(image_path, 1)
        # label_img = cv2.imread(label_path, 1)
        # img = cv2.resize(img, (self.width, self.height))
        # label_img = cv2.resize(label_img, (self.width, self.height))
        # im = img
        # lim = label_img
        # lim = lim[:, :, 0]
        # im = np.array(im).astype(np.float32)
        # lim = np.array(lim).astype('int32')
        # # im /= 255.0
        # # im -= self.mean
        # # im /= self.std
        # im = np.transpose(im, [2, 0, 1])
        # # torch_img = torch.from_numpy(im).float()
        # mask = torch.from_numpy(lim).long()
        # print("img shape = " + str(shape(im)) + " mask shape = " + str(shape(mask)))
        # return im, mask, os.path.basename(image_path)

    def __len__(self):
        return len(self.items)
        # return 32

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
