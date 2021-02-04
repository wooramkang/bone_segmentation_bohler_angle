import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.net = net().cuda()
        self.net = net().to(self.device)

        if torch.cuda.is_available():
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_one_img_to_real(self, img):
        
        img.squeeze().cpu().data.numpy()
        img = np.array(img) * 255

        return img

    def test_one_mask_to_real(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 255
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0
        img = V(torch.Tensor(img).to(self.device))
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.to(self.device), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.to(self.device), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data#[0]
    
    def calc_loss(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        #loss.backward()

        return loss.data

    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(path))
        else:
            self.net.load_state_dict(torch.load(path, map_location={'cuda:0': 'cpu'}), strict=False)

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        #print >> mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr)
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
