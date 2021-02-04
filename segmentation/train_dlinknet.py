import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter
import cv2
import os
import numpy as np

from time import time
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
import sys

argument = sys.argv
SHAPE = (512, 512)

if len(argument) > 1:
    target = argument[1]
else:
    target = "bone"

NAME = 'log01_dlink34_'+target
ROOT = '../dataset/' + target + '/crop'

weight_dir = 'weights/' + NAME+'.th'

imagelist = list(os.listdir(ROOT))
trainlist = imagelist

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)

batchsize = 1
if torch.cuda.is_available():
    BATCHSIZE_PER_CARD = 4
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT, target)

train_data, val_data = torch.utils.data.random_split(dataset, [ int(len(dataset)* 0.7) ,  len(dataset) - int(len(dataset)* 0.7) ])

data_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2)

val_data_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2)

tic = time()
no_optim = 0
total_epoch = 200
train_epoch_best_loss = 1000000.

writer = SummaryWriter(os.path.join('runs', 'experiment_' + target , ""))
iter_count =0
loss_list = []
epoch_loss_list = []

try:
    solver.load(weight_dir)
    print("load trained model")
except:
    print("no model trained")

for epoch in range(1, total_epoch + 1):
    
    data_loader_iter = iter(data_loader)
    val_loader_iter = iter(val_data_loader)
    train_epoch_loss = 0
    #print(len(data_loader_iter))
    train_epoch_loss /= len(data_loader_iter)
    
    val_loss = 0
    val_epoch_loss = 0
    n = 0
    #writer.add_scalars('epoch_loss/per_epoch', {'epoch' : train_epoch_loss}, epoch)
    for img, mask in val_loader_iter:
        
        solver.set_input(img, mask)
        val_loss = solver.calc_loss()
        val_epoch_loss += val_loss.item()
        
        n = n + 1
    writer.add_scalars('loss/per_epoch', {'val_loss': val_epoch_loss/n }, epoch)

    train_loss = 0
    train_epoch_loss = 0
    n = 0

    for img, mask in data_loader_iter:
        
        solver.set_input(img, mask)
        train_loss = solver.calc_loss()
        train_epoch_loss += val_loss.item()
        
        n = n + 1

    writer.add_scalars('loss/per_epoch', {'train_loss': train_epoch_loss/n }, epoch)

    train_loss = 0
    train_epoch_loss = 0
    data_loader_iter = iter(data_loader)

    for img, mask in data_loader_iter:
        
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
        #print(train_loss)
        #print(iter_count)
        if (iter_count % 1000) == 0:
            t_img = solver.test_one_img_to_real(img)
            #print(t_img.shape)
            #print(1) * 2 - 1
            writer.add_image('raw_image/raw_image', t_img[0]*255, iter_count)
            t_mask = solver.test_one_img_to_real(mask)
            writer.add_image('raw_mask/raw_mask', t_mask[0]*255, iter_count)
            #iter_img = solver.test_one_img(img)
            iter_img = solver.test_one_mask_to_real(img)#[0,]
            #print(iter_img.shape)
            iter_img = iter_img[0,]  
            #print(iter_img.shape)
            iter_img = iter_img.reshape(1,512,512)
            #print(iter_img.shape)

            writer.add_image('gen_mask/gen_mask', iter_img, iter_count)
            #print(train_loss)
            #loss_list.append(train_loss.item())
            print(str(iter_count ) + " iter snap_shot saved")

        writer.add_scalars('loss/per_iter', {'iter': train_loss.item()}, iter_count)
        iter_count = iter_count + 1

    print( '********')
    print( 'epoch:',epoch,'    time:',int(time()-tic))
    print( 'train_loss:',train_epoch_loss)
    print( 'SHAPE:',SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(weight_dir)
        
    if no_optim > 3:
        print( 'early stop at %d epoch' % epoch)
        break
        
print( 'Finish')

