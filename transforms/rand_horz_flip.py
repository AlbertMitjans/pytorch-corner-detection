import torch
import numpy as np


class HorizontalFlip(object):
    def __call__(self, sample):
        if np.random.random() < 0.5:
            image = sample['image']
            grid = sample['grid']
            sample['image'] = torch.flip(image, [-1])
            sample['grid'] = torch.flip(grid, [-1])
        return sample