"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
#from v2 import
import numpy as np
import os
from PIL import Image

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def rgb2gray(rgb):

    r= rgb[:,:,0]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def default_loader(id, root, target):
    root_dir = os.path.join('..', 'dataset', target, "")
    img_dir = os.path.join(root_dir, 'crop' , str(id))
    mask_dir = os.path.join(root_dir, 'label_crop', str(id))
    #root_dir = '../dataset/' + target
    #print(id)
    #img = cv2.imencode('.jpg', cv2.imread(root_dir + '/img_crop/'+ str(id)))# +'.png' ))
    img = Image.open(img_dir)# + '.png')
    #img = Image.open(root_dir+ '/rot_crop/' +str(id))# + '.png')
    img = img.convert('RGB')
    mask = cv2.imread(mask_dir ,cv2.IMREAD_GRAYSCALE)# 0)
    #mask = Image.open(root_dir + '/label_crop/' + str(id))
    #mask = mask.convert('L')
    #mask = rgb2gray(mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img)
    #print(img.shape)
    #print(img[0])
    #mask = np.array(mask)
    #print(mask.shape)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    #print(img.shape)
    #print(mask.shape)
    mask[mask>0] = 1
    mask[mask==0] = 0
    #print(mask.shape)
	#mask = abs(mask-1)
    return img, mask

def test_default_loader(id, root, target):
    root_dir = os.path.join('..', 'dataset', target, "")
    img_dir = os.path.join(root_dir, 'crop' , str(id))
    mask_dir = os.path.join(root_dir, 'label_crop', str(id))
    #root_dir = '../dataset/' + target
    #print(id)
    #img = cv2.imencode('.jpg', cv2.imread(root_dir + '/img_crop/'+ str(id)))# +'.png' ))
    #img = Image.open(root_dir+ '/img_crop/' +str(id))# + '.png')
    img = Image.open(img_dir)# + '.png')
    img = img.convert('RGB')
    mask = cv2.imread(mask_dir ,cv2.IMREAD_GRAYSCALE)# 0)
    #mask = Image.open(root_dir + '/label_crop/' + str(id))
    #mask = mask.convert('L')
    #mask = rgb2gray(mask)
    
    mask = np.expand_dims(mask, axis=2)

    img = np.array(img)
    #print(img.shape)
    #print(img[0])
    #mask = np.array(mask)
    #print(mask.shape)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    #print(img.shape)
    #print(mask.shape)
    mask[mask>0] = 1
    mask[mask==0] = 0
    #print(mask.shape)
	#mask = abs(mask-1)
    return img, mask

class ImageFolder(data.Dataset):
    def __init__(self, trainlist, root, target):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.target = target

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root, self.target)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)

class TEST_ImageFolder(data.Dataset):
    def __init__(self, trainlist, root, target):
        self.ids = trainlist
        self.loader = test_default_loader
        self.root = root
        self.target = target
    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.root, self.target)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)