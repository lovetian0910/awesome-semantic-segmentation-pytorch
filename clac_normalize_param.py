from torch.utils.data import DataLoader
from core.data.dataloader.qgamedata import LiveSegmentation
import numpy as np


def calc_mean_std():
    train_data = LiveSegmentation(crop_size = 512, split = "train")
    train_loader = DataLoader(dataset=train_data, batch_size=1000, shuffle=True)
    train = iter(train_loader).next()[0]
    train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
    train_std = np.std(train.numpy(), axis=(0, 2, 3))
    return train_mean, train_std

if __name__ == "__main__":
    mean, std = calc_mean_std()
    print(mean, std)