import torch
import torch.utils.data
from utils.utils import init_model_and_dataset
import argparse

from train import train
from test import test


def main(train_flag, evaluate_val, save_imgs, depth, ckpt, num_epochs, batch_size):

    if train_flag:
        train(ckpt, depth, num_epochs, batch_size)

    elif not train_flag:
        num_workers = 0
        directory = 'data/'

        model, train_dataset, val_dataset, _, _ = init_model_and_dataset(depth, directory)

        train_dataset.evaluate()
        val_dataset.evaluate()

        if evaluate_val:
            transformed_dataset = val_dataset
        if not evaluate_val:
            transformed_dataset = torch.utils.data.ConcatDataset((train_dataset, val_dataset))

        # load the pretrained network
        if ckpt is not None:
            checkpoint = torch.load(ckpt)

            model.load_state_dict(checkpoint['model_state_dict'])

        val_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True)

        test(val_loader, model, save_imgs=save_imgs)


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str2bool, default=True, help="if True/False, training/testing will be implemented")
    parser.add_argument("--val_data", type=str2bool, default=True, help="if True/False, all/validation data will be used "
                                                                    "for testing")
    parser.add_argument("--save_imgs", type=str2bool, default=True, help="if True, output imgs will be saved")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--depth", type=str2bool, default=True, help="if True/False, depth/RGB images will be used")
    parser.add_argument("--ckpt", type=str, default=None, help="path to ckpt file")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
    opt = parser.parse_args()
    print(opt)

    main(train_flag=opt.train, evaluate_val=opt.val_data, save_imgs=opt.save_imgs, depth=opt.depth, ckpt=opt.ckpt,
         num_epochs=opt.num_epochs, batch_size=opt.batch_size)
