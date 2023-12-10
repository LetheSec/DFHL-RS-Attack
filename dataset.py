import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


def strong_aug(image):
    device = image.device
    image = TF.center_crop(
        image,
        [int(32.0 * random.uniform(0.95, 1.0)), int(32.0 * random.uniform(0.95, 1.0))],
    )
    image = TF.resize(image, [32, 32])
    noise = torch.randn_like(image).to(device) * 0.001
    image = torch.clamp(image + noise, 0.0, 1.0)
    if random.uniform(0, 1) > 0.5:
        image = TF.vflip(image)
    if random.uniform(0, 1) > 0.5:
        image = TF.hflip(image)
    angles = [-15, 0, 15]
    angle = random.choice(angles)
    image = TF.rotate(image, angle)
    return image


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data


class FakeDataset(torch.utils.data.Dataset):
    """Some Information about FakeDataset"""

    def __init__(self, root="", transform=None):
        super(FakeDataset, self).__init__()

        self.transform = transform

        history_images = np.load(os.path.join(root, "fake_images.npy"))
        history_labels = np.load(os.path.join(root, "fake_labels.npy"))
        self.images = torch.from_numpy(history_images)
        self.labels = history_labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return image, label

    def __len__(self):
        return len(self.labels)
