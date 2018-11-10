import argparse
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

from optimisation.training import train, validate
from optimisation import loss
from utils import TransformedHuaweiDataset
import models

parser = argparse.ArgumentParser()

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument('-sd', '--save_dir', type=str, metavar='PATH', default='results/save_dir',
                    help='path to save results and checkpoints to (default: results/save_dir)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-trb', '--train_batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size for training data (default: 256)')
parser.add_argument('-teb', '--test_batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size for test data (default: 1)')

parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate (default: 0.005)')
parser.add_argument('--loss', type=str, default='MSELoss')
parser.add_argument('--model', type=str, default='BasicGenerator')
parser.add_argument('--optim', type=str, default='Adam')

parser.add_argument('--resume', metavar='PATH', help='load from a path to a saved checkpoint')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')

# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

# CNN
parser.add_argument('--cnn_in_channels', type=int, default=3)
parser.add_argument('--cnn_hidden_channels', type=int, default=32)
parser.add_argument('--cnn_num_hidden_layers', type=int, default=7)
parser.add_argument('-no', '--no_iso', action='store_true', default=False,
                    help='not to use image ISO values as extra conditioning data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.iso = not args.no_iso

# Random seeding
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

if args.cuda:
    # gpu device number
    torch.cuda.set_device(args.gpu_num)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {'num_workers': args.workers}


def main(args, kwargs):
    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    # Save config
    torch.save(args, args.save_dir + 'denoising' + '.config')
    writer = SummaryWriter(os.path.join(args.savedir, 'Summaries'))

    # construct network from args
    model = getattr(models, args.model)(args)
    optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=args.lr)
    criterion = getattr(loss, args.loss)()

    if args.cuda:
        print("Model on GPU")
        model.cuda()
        criterion.cuda()

    # TODO: Set root-dir for dataset
    noisy_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    clean_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    train_dataset = HuaweiDataset(root_dir='')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              shuffle=True, **kwargs)

    val_dataset = HuaweiDataset(root_dir='')
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size,
                            shuffle=False, **kwargs)

    best_loss = np.inf

    if args.resume:
        print('==> Loading checkpoint')
        checkpoint = torch.load(args.resume)
        print('==> Checkpoint loaded')
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        training_iters = args.start_epoch * len(train_loader)
        validate(args, val_loader, model, criterion, training_iters, writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        training_iters = (epoch + 1) * len(train_loader)

        # Train
        print("===> Training on Epoch %d" % epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        # Validate
        print("===> Validating on Epoch %d" % epoch)
        val_loss = validate(val_loader, model, criterion, training_iters)

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        # Save checkpoint
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }
        save_checkpoint(checkpoint, model_filename, is_best)


def save_checkpoint(checkpoint, filename, is_best):
    print("===> Saving checkpoint '{}'".format(filename))
    model_filename = os.path.join(args.save_dir, filename)
    best_filename = os.path.join(args.save_dir, 'model_best.pth.tar')
    torch.save(checkpoint, model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)
    print("===> Saved checkpoint '{}'".format(model_filename))


if __name__ == '__main__':
    main(args, kwargs)
