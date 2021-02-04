import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter
import cv2
import os
import numpy as np
import random
from time import time
from PIL import Image
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import TEST_ImageFolder
import sys

def smoothing(img, iter, filter_size):
    dst =None

    for j in range(iter):
        kernel = np.ones((filter_size,filter_size),np.float32)/(filter_size * filter_size)
        dst = cv2.filter2D(img,-1,kernel)
        img = dst

    return dst

def sharpening(img, iter):
    dst =None

    for j in range(iter):
        kernel = np.array([[0, -1, 0], 
                        [-1, 5,-1], 
                        [0, -1, 0]])
        dst = cv2.filter2D(img,-1,kernel)
        img = dst

    return dst

start = time()

argument = sys.argv
SHAPE = (512, 512)

if len(argument) > 1:
    target = argument[2]
else:
    target = "bone"

NAME = 'log01_dlink34_' + target
WEIGHT_DIR = os.path.join('weights', NAME+'.th')
print(WEIGHT_DIR)
LOG_DIR = os.path.join('runs', 'experiment_' + target , "")
RESULT_DIR = os.path.join('dlink_test_result', target , "") 

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

#BATCHSIZE_PER_CARD = 4
solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
#batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

no_optim = 0
writer = SummaryWriter(LOG_DIR)
iter_count =0
loss_list = []
epoch_loss_list = []
train_epoch_loss = 0

#try:
solver.load(WEIGHT_DIR)
print("load trained model")
#except:
#    print("no model trained")

raw_imgdir = "../dataset/"+target+"/img/"
file_list = os.listdir(raw_imgdir)

for i in file_list:
    map_run_time = time()
    src = cv2.imread(raw_imgdir + i, cv2.IMREAD_COLOR)
    src = cv2.resize(src, (512, 512))
    h, w, c = src.shape    
    img = np.array(src)
        
    dst = src.copy() 
    dist = []
    temp = np.array(src)
    temp = np.array(temp, np.float32).transpose(2,0,1)/255.0
    dist.append(temp)

    dist = np.array(dist)    
    img = torch.Tensor(dist)

    solver.set_input(img)#, mask)
    _ = solver.forward()#optimize()
    
    t_img = solver.test_one_img_to_real(img)
    
    t_img= t_img[0]
    t_img = t_img.transpose(1, 2, 0)
            
    t_img_out = t_img 
    t_img_out = t_img_out.transpose(2,0,1)
    t_img_out = np.uint8(t_img_out.transpose(2,0,1))
    #print(t_img_out.shape)
    iter_img = solver.test_one_mask_to_real(img)#[0,]
    iter_img = np.uint8(iter_img)
    #Image.fromarray(t_img_out).save(RESULT_DIR + 'raw_' + str(iter_count) + '_' + str(j) + '.jpg')
    
    t_out = Image.fromarray(iter_img, mode='L')
    
    #t_out.save(RESULT_DIR + 'result_' + str(iter_count) + '_' + str(j) + '.jpg')
    iter_img = np.uint8(iter_img.reshape(1,512,512))

    final_output = Image.new("L", (w, h))

    final_output.paste(t_out, (0,0))#, 1024, 1024))    
    RESULT_OUT_DIR = os.path.join(RESULT_DIR, 'result_' + i[:-4] +'.jpg')
    final_output.save(RESULT_OUT_DIR)
    
    filter_list = [1, 2, 3, 3]
    smoothing_iter = 2
    sharpening_iter = 2

    img = cv2.imread(RESULT_OUT_DIR)
    for j in filter_list:
        dst = smoothing(img, smoothing_iter, j)
        dst[dst>=96] = 255
        dst[dst<96] = 0
        img = dst
        dst = sharpening(img, sharpening_iter)
        img = dst

    RESULT_OUT_DIR = os.path.join(RESULT_DIR, 'result_' + i[:-4] +'_post.jpg')
    cv2.imwrite(RESULT_OUT_DIR, dst)    
    result_img = dst.copy()

    dst[dst == 255] = True
    dst[dst != True] = False
    print(dst.shape)

    starting = 0
    ending = 0
    top_end = 0

    for k in range(dst.shape[0]):
        if dst[k, :].any():
            starting = k
            for kk in range(dst.shape[1]):
                if dst[k, kk].all():
                    top_end = kk
                    break
            break

    print(starting)

    for k in range(starting, dst.shape[0]):
        if not dst[k, :].any():
            ending = k
            break
    if ending == 0:
        ending = dst.shape[0] - 1
    print(ending)
    
    left_end = dst.shape[0] -1
    left_top = 0
    right_end = 0
    right_top = 0

    for k in range(starting, int(starting + (ending-starting)/4) ):
        for kk in range(dst.shape[1]):
            if dst[k, kk].all():
                if kk < left_end:
                    left_end = kk
                    left_top = k
                if kk > right_end:
                    right_end = kk
                    right_top = k


    print((starting, top_end))
    pos = []
    pos.append((top_end, starting))
    if( abs(top_end - left_end) < abs(top_end - right_end) ):
        print( (right_end, right_top) )
        pos.append(( right_end, right_top ) )
    else:
        print( (left_end, left_top) )
        pos.append(( left_end, left_top ))
    print()


    result_img = cv2.line(result_img, pos[0], pos[1], (255, 0, 0), 3)
    #RESULT_OUT_DIR = os.path.join(RESULT_DIR, 'result_' + i[:-4] +'_post_line.jpg')
    #cv2.imwrite(RESULT_OUT_DIR, result_img)    

    left_end = dst.shape[0] -1
    left_top = 0
    right_end = 0
    right_top = 0
        
    for kk in range(dst.shape[1]):
        if dst[ending-1, kk].all():
            if kk < left_end:
                left_end = kk
                left_top = ending -1
            if kk > right_end:
                right_end = kk
                right_top = ending -1

    pos_prime = []
    
    pos_prime.append(( int((pos[0][0] + pos[1][0])/2) , int((pos[0][1] + pos[1][1])/2)  ))
    pos_prime.append( ( int((left_end + right_end)/2) , ending ) )
    print(pos_prime)
    

    pos_prime_v = np.array([pos_prime[0][0] - pos_prime[1][0], pos_prime[0][1] - pos_prime[1][1]] )
    pos_v = np.array( [pos[0][0] - pos[1][0], pos[0][1] - pos[1][1]] )
    print( pos_prime_v)
    print( pos_v)
    norm_pos_v = np.linalg.norm(pos_v)
    norm_pos_prime_v = np.linalg.norm(pos_prime_v)
    inner_product =  pos_v[0] * pos_prime_v[0] + pos_v[1] * pos_prime_v[1]
    cos_v = inner_product/ (norm_pos_v * norm_pos_prime_v)
    print("angles")    
    print(np.arccos( cos_v ))

    cos_v = (np.arccos( cos_v ) / 3.14) * 180
    print(cos_v)
    result_img = cv2.line(result_img, pos_prime[0], pos_prime[1], (0, 255, 0), 3)
    RESULT_OUT_DIR = os.path.join(RESULT_DIR, 'result_' + i[:-4] +'_post_line.jpg')
    cv2.imwrite(RESULT_OUT_DIR, result_img)    

    print("inference running time")
    print(time() - map_run_time)
    

print(iter_count)
print("running_time")
print(time() - start) 
