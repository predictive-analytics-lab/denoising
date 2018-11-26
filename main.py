import argparse
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from optimisation.testing import test
from optimisation.training import train, validate, evaluate_psnr_and_vgg_loss
from optimisation import loss
from utils import TransformedHuaweiDataset, transform_sample, CsvLoader
import models


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--run_on_test', nargs='+',
                        help='Evaluate a model (checkpoint) on test set. '
                        'Args: Checkpoint path, Test data path[, Save path]')

    parser.add_argument('-dd', '--data_dir', help='location of transformed data')
    parser.add_argument('-ts', '--test_split', help='Fraction of data to be used for validation',
                        default=0.2, type=float)
    parser.add_argument('-ds', '--data_subset', help='Fraction of crops per image to be used',
                        default=1.0, type=float)

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--random_seed', action="store_true",
                        help='use random seed rather than 42.')

    parser.add_argument('-sd', '--save_dir', type=str, metavar='PATH', default='',
                        help='path to save results and checkpoints to '
                             '(default: ../results/<model>/<current timestamp>)')
    parser.add_argument('--num-samples-to-log', type=int, metavar='N', default=32,
                        help='number of image samples to write to tensorboard each epoch'
                             ' (default: 32)')

    # training parameters
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-trb', '--train_batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size for training data (default: 256)')
    parser.add_argument('-teb', '--test_batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size for test data (default: 256)')

    parser.add_argument('-lr', '--learning_rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate (default: 0.005)')

    # model parameters
    parser.add_argument('--loss', type=str, default='MSELoss')
    parser.add_argument('--model', type=str, default='SimpleCNN')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--args_to_loss', action='store_true', default=False,
                        help='whether to pass the commandline arguments to the loss function')

    parser.add_argument('--resume', metavar='PATH', help='load from a path to a saved checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set (default: false)')

    # gpu/cpu
    parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU',
                        help='choose GPU to run on.')

    # CNN
    parser.add_argument('--cnn_in_channels', type=int, default=3)
    parser.add_argument('--cnn_hidden_channels', type=int, default=32)
    parser.add_argument('--cnn_hidden_layers', type=int, default=7)
    parser.add_argument('--interpolate', action='store_true', default=False,
                        help='interpolate rather than learn noise as an image residual')
    parser.add_argument('-ni', '--no_iso', action='store_true', default=False,
                        help='not to use image ISO values as extra conditioning data')
    parser.add_argument('--use_class', action='store_true', default=False,
                        help='use the class information of images')

    # VGG loss
    parser.add_argument('--vgg_feature_layer', type=int, default=11,
                        help='VGG19 layer number from which to extract features')

    args = parser.parse_args(raw_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.iso = not args.no_iso
    args.num_classes = 3 if args.use_class else 0

    if args.random_seed:
        args.seed = random.randint(1, 100000)
    else:
        args.seed = 42

    return args


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda:
        # gpu device number
        torch.cuda.set_device(args.gpu_num)

    if args.run_on_test:
        test(args, transform_sample)
        return

    # Create results path
    if args.save_dir: # If specified
        save_path = Path(args.save_dir).resolve()
    else:
        save_path = Path().resolve().parent / "results" / args.model / str(round(time.time()))
        save_path.parent.mkdir(exist_ok=True)
    save_path.mkdir()  # Will throw an exception if the path exists OR the parent path _doesn't_

    kwargs = {'pin_memory': True} if args.cuda else {}

    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.seed)

    # Save config
    torch.save(args, save_path / 'denoising.config')
    writer = SummaryWriter(save_path / 'summaries')

    # construct network from args
    model = getattr(models, args.model)(args)
    model = model.cuda() if args.cuda else model
    optimizer = getattr(torch.optim, args.optim)(model.parameters(), lr=args.learning_rate)
    criterion_constructor = getattr(loss, args.loss)
    criterion = criterion_constructor(args) if args.args_to_loss else criterion_constructor()
    criterion = criterion.cuda() if args.cuda else criterion

    # dataset = TransformedHuaweiDataset(root_dir=args.data_dir, transform=transform_sample)
    dataset = CsvLoader(args.data_dir, transform=transform_sample)
    train_dataset, val_dataset = dataset.random_split(test_ratio=args.test_split,
                                                      data_subset=args.data_subset)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=args.workers, **kwargs)

    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=args.workers, **kwargs)

    best_loss = np.inf

    if args.resume:
        print('==> Loading checkpoint')
        checkpoint = torch.load(args.resume)
        print('==> Checkpoint loaded')
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        # Evaluate model using PSNR and SSIM metrics
        evaluate_psnr_and_vgg_loss(args, model, val_loader)
        return

    for epoch in range(args.start_epoch, args.epochs):
        training_iters = (epoch + 1) * len(train_loader)

        # Train
        print("===> Training on Epoch %d" % epoch)
        train(args, train_loader, model, criterion, optimizer, epoch, writer)

        # Validate
        print("===> Validating on Epoch %d" % epoch)
        val_loss = validate(args, val_loader, model, criterion, training_iters, writer)

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
        save_checkpoint(checkpoint, model_filename, is_best, save_path)

    # Evaluate model using PSNR and SSIM metrics
    evaluate_psnr_and_vgg_loss(args, model, val_loader)


def save_checkpoint(checkpoint, filename, is_best, save_path):
    print("===> Saving checkpoint '{}'".format(filename))
    model_filename = save_path / filename
    best_filename = save_path / 'model_best.pth.tar'
    torch.save(checkpoint, model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)
    print("===> Saved checkpoint '{}'".format(model_filename))


if __name__ == '__main__':
    main(parse_arguments())
