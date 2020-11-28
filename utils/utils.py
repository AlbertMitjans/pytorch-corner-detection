from __future__ import print_function, division

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max

from data.dataset import CornersDataset
from loss.loss import JointsMSELoss
from models.Stacked_Hourglass import HourglassNet, Bottleneck
from transforms.rand_crop import RandomCrop
from transforms.rand_horz_flip import HorizontalFlip


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_model_and_dataset(depth, directory, lr=5e-6, weight_decay=0, momentum=0):
    # define the model
    model = HourglassNet(Bottleneck)
    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)

    checkpoint = torch.load("checkpoints/hg_s2_b1/model_best.pth.tar")

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model = nn.Sequential(model, nn.Conv2d(16, 1, kernel_size=1).cuda())

    if depth:
        end_file = '.tif'
    if not depth:
        end_file = '.png'

    cudnn.benchmark = True

    random_crop = RandomCrop(size=0.8)
    horizontal_flip = HorizontalFlip()

    train_dataset = CornersDataset(root_dir=directory + 'train_dataset', end_file=end_file, depth=depth,
                                   transform=transforms.Compose([random_crop, horizontal_flip]))
    val_dataset = CornersDataset(root_dir=directory + 'val_dataset', end_file=end_file, depth=depth,
                                 transform=transforms.Compose([random_crop, horizontal_flip]))

    return model, train_dataset, val_dataset, criterion, optimizer


def accuracy(corners, output, target, global_recall, global_precision):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    # we send the data to CPU
    output = output.cpu().detach().numpy().clip(0)
    target = target.cpu().detach().numpy()

    for batch_unit in range(batch_size):  # for each batch element
        recall, precision, max_out = multiple_gaussians(output[batch_unit], target[batch_unit])

        global_recall.update(recall)
        for i, (a, b) in enumerate(sorted(corners[batch_unit], key=lambda x: x[0], reverse=True)):
            if a != 0 and b != 0:
                global_precision[i].update(precision[i])

    return max_out


def multiple_gaussians(output, target):
    # we calculate the positions of the max value in output and target
    max_target = peak_local_max(target[0].clip(0.99), min_distance=19, exclude_border=False,
                                indices=False)  # num_peaks=4)
    labels_target = label(max_target)[0]
    max_target = np.array(center_of_mass(max_target, labels_target, range(1, np.max(labels_target) + 1))).astype(np.int)

    true_p = np.array([0, 0, 0, 0]).astype(np.float)
    all_p = np.array([0, 0, 0, 0]).astype(np.float)

    max_out = peak_local_max(output[0], min_distance=19, threshold_rel=0.1, exclude_border=False, indices=False)
    labels_out = label(max_out)[0]
    max_out = np.array(center_of_mass(max_out, labels_out, range(1, np.max(labels_out) + 1))).astype(np.int)

    max_values = []

    for index in max_out:
        max_values.append(output[0][index[0]][index[1]])

    max_out = np.array([x for _, x in sorted(zip(max_values, max_out), reverse=True, key=lambda x: x[0])])

    for n in range(min(4, max_out.shape[0])):
        max_out2 = max_out[:n + 1]
        for i, (c, d) in enumerate(max_out2):
            if i < max_out2.shape[0] - 1:
                dist = np.absolute((max_out2[i + 1][0] - c, max_out2[i + 1][1] - d))
                if dist[0] <= 8 and dist[1] <= 8:
                    continue
            all_p[n] += 1
            count = 0
            for (a, b) in max_target:
                l = np.absolute((a - c, b - d))
                if l[0] <= 8 and l[1] <= 8:
                    true_p[n] += 1
                    count += 1
                    if count > 1:
                        all_p[n] += 1

    num_targets = max_target.shape[0]

    if num_targets == 0:
        recall = 0
        precision = np.array([0, 0, 0, 0]).astype(np.float)
    else:
        recall = true_p[min(4, max_out.shape[0]) - 1] / num_targets
        precision = true_p / all_p
        precision[np.isnan(precision)] = 0

    return recall, precision, max_out
