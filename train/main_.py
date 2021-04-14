
import torch
import torch.optim as optim
import time
import shutil
import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm

import argparse
from tensorboardX import SummaryWriter

from datetime import datetime


import sys
sys.path.append('../')

from utils.metrics import MetricTracker, jaccard_index, dice_coeff
from utils.loss_ import CrossEntropyLoss, DiceLoss, BDDiceLoss, WeightedCELoss, WeightedDiceLoss, DiceWithCrossEntropyLoss, \
    NoiseRobustDiceLoss, ExpLogLoss, BD_LACE_WDiceLoss
from utils.data_gen import Data_Gen
from utils.unet import UNet
import utils.augmentation as aug


model_choices = ['unet']
loss_choices = ['CE', 'CEDice', 'Dice', 'NR-Dice', 'WCE', 'WDice', 'ELL', 'BDDice', 'BD_LACE_WDice']

parser = argparse.ArgumentParser(description='Building Extraction')

parser.add_argument('--train-img-dir', metavar='DATA_DIR', 
                        help='path to train img dir')
parser.add_argument('--train-mask-dir', metavar='DATA_DIR', 
                        help='path to train mask dir')
parser.add_argument('--val-img-dir', metavar='DATA_DIR', 
                        help='path to val img dir')
parser.add_argument('--val-mask-dir', metavar='DATA_DIR', 
                        help='path to val mask dir')
parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
parser.add_argument('--crop-sz', default=112, type=int, metavar='SIZE',
                        help='number of cropped pixels from orig image (default: 112)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='epoch to start from (used with resume flag')
parser.add_argument('--loss', default='CE', type=str, metavar='LOSS',
                        help='choose loss for training, choices are:'\
                            + '|'.join(loss_choices) + ' (default: CE)')
parser.add_argument('--model', default='unet', type=str, metavar='M',
                    choices=model_choices,
                        help='choose model for training, choices are: ' \
                        + ' | '.join(model_choices) + ' (default: unet)')
parser.add_argument('--cls-num', default=2, type=int, metavar='CNUM',
                        help='number of classes (default: 2)')

args = parser.parse_args()

sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
print('saving file name is ', sv_name)

checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
logs_dir = os.path.join('./', sv_name, 'logs')

if os.path.isdir(os.path.join('./', sv_name)):
    raise FileExistsError

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    
    if os.path.exists(filename):
        raise FileExistsError

    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))


def main():
    global args, sv_name, logs_dir, checkpoint_dir

    write_arguments_to_file(args, os.path.join('./', sv_name, sv_name+'_arguments.txt'))

    if args.model == 'unet':
        # get model
        model = UNet(num_classes=args.cls_num)
    else:
        pass

    if torch.cuda.is_available():
        model = model.cuda()

    if args.loss == 'CE':
        criterion = CrossEntropyLoss(cls_num=args.cls_num)
    elif args.loss == 'CEDice':
        criterion = DiceWithCrossEntropyLoss(cls_num=args.cls_num, ce_weight=1.0)
    elif args.loss == 'Dice':
        criterion = DiceLoss(cls_num=args.cls_num)
    elif args.loss == 'WCE':
        criterion = WeightedCELoss(cls_num=args.cls_num, pix_weights=[0.1,0.9])
    elif args.loss == 'WDice':
        criterion = WeightedDiceLoss(cls_num=args.cls_num, cls_weights=[0.1,0.9])
    elif args.loss == 'BDDice':
        criterion = BDDiceLoss(cls_num=2)
    elif args.loss == 'BD_LACE_WDice':
        criterion = BD_LACE_WDiceLoss(cls_num=2, priors=[0.9,0.1], dice_cls_weights=[0.1,0.9])
    else:
        pass

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_loss = 999

    start_epoch = args.start_epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)
            
            if checkpoint['epoch'] > args.start_epoch:
                start_epoch = checkpoint['epoch']
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    dataset_train = Data_Gen(img_dir=args.train_img_dir,
                                    mask_dir=args.train_mask_dir,
                                    val_img_dir=args.val_img_dir,
                                    val_mask_dir=args.val_mask_dir,
                                    transform=transforms.Compose([aug.RandomCropTarget(output_size=args.crop_sz),
                                                            aug.RandomFlip(),
                                                            aug.RandomRotate(),
                                                            aug.ToTensorTargetDist(cls_num=args.cls_num)]),
                                    phase='train')

    dataset_val = Data_Gen(img_dir=args.train_img_dir,
                                    mask_dir=args.train_mask_dir,
                                    val_img_dir=args.val_img_dir,
                                    val_mask_dir=args.val_mask_dir,
                                    transform=transforms.Compose([aug.RandomCropTarget(output_size=args.crop_sz),
                                                            aug.ToTensorTarget()]),
                                    phase='val')

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True, pin_memory=True)
    

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        meters = train(train_dataloader, model, criterion, optimizer, train_writer, epoch)
        val(val_dataloader, model, val_writer, epoch)

        best_loss = min(meters['train_loss'].val, best_loss)
        is_best_loss = meters['train_loss'].val < best_loss

        save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict()
            }, is_best_loss, sv_name)

        lr_scheduler.step()

def make_train_step(idx, data, model, optimizer, criterion, meters):

    inputs = data['sat_img'].cuda()
    labels = data['map_img'].cuda()

    optimizer.zero_grad()

    outputs = model(inputs)


    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    # meters["train_acc"].update(dice_coeff(outputs, labels), outputs.size(0))
    meters["train_loss"].update(loss.item(), outputs.size(0))
    # meters["train_IoU"].update(jaccard_index(outputs, labels), outputs.size(0))
    return meters



def train(train_loader, model, criterion, optimizer, train_writer, epoch):

    # train_acc = MetricTracker()
    train_loss = MetricTracker()
    # train_IoU = MetricTracker()
    # train_BCE = metrics.MetricTracker()
    # train_DICE = metrics.MetricTracker()

    meters = {"train_loss": train_loss}

    model.train()

    for idx, data in enumerate(tqdm(train_loader, desc="training", ascii=True, ncols=20)):

        meters = make_train_step(idx, data, model, optimizer, criterion, meters)


    info = {
        "Loss": meters["train_loss"].avg,
        # "Acc": meters["train_acc"].avg,
        # "IoU": meters["train_IoU"].avg
    }

    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)
    
    print('Train Loss: {:.6f}'.format(
            meters["train_loss"].avg,
            # meters["train_acc"].avg,
            # meters["train_IoU"].avg
            ))

    return meters


def val(val_dataloader, model, val_writer, epoch):

    val_Dice = MetricTracker()

    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_dataloader, desc="val", ncols=10)):

            inputs = data['sat_img'].cuda()
            labels = data['map_img'].cuda()

            outputs = model(inputs)

            outputs = torch.argmax(outputs, dim=1).float()

            val_Dice.update(dice_coeff(outputs, labels), outputs.size(0))

    info = {
        "Dice": val_Dice.avg
    }

    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)
    
    print('Val Dice: {:.6f}'.format(
            val_Dice.avg
            ))

if __name__ == "__main__":
    main()
