import torch
from torch import nn
from untils.loss import DiceLoss
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import time
import torch.nn.functional as F
from untils.untils import ConfusionMatrix
from itertools import cycle

def criterion(inputs, target, weights = None):

    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=255, weight=weights)
    return loss

def criterion1(inputs, target, weights = None):
    losses = 0.0

    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss1 = nn.functional.cross_entropy(inputs, target, ignore_index=255, weight=weights)

    dceloss = DiceLoss(103)
    loss2 = dceloss(inputs,target)
    return 0.5*loss1+0.5*loss2 


def train_one_epoch(model, optimizer, train_loader,train_u=None, device = 'cuda', weights=None):

    # 训练模式
    sum_loss = 0
    model.train()
    '''
    # semi train
    loader = zip(cycle(train_loader), train_u)
    loop = tqdm(loader, desc='Train')
    for (img, target),(img_w,img_s) in loop:
        img, target,img_w,img_s = img.to(device), target.to(device),img_w.to(device),img_s.to(device)
        img_pred = model(img)
        mask = model(img_w)
        pred = model(img_s,semi=True)
        mask = mask.argmax(dim=1)
        loss1 = criterion(img_pred,target)

        loss2 = criterion(pred,mask)
        loss = 0.75*loss1 + 0.25*loss2

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
            
        sum_loss += loss
    '''
    loop = tqdm(train_loader)
    for img, target in loop:
        img, target = img.to(device), target.to(device)
        img_pred = model(img)

        loss = criterion(img_pred,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        sum_loss += loss
    
    lr = optimizer.param_groups[0]["lr"]
    return sum_loss,lr

def evaluate(model, test_loader, num_classes,epoch=1,  device='cuda'):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    
    with torch.no_grad():
        loop2 = tqdm(test_loader, desc='Test')
        for image, target in loop2:
            image, target = image.to(device), target.to(device)
            output = model(image)
            #output1 = F.interpolate(output[0],size=(target.shape[1], target.shape[2]), mode='bilinear',align_corners=False)
            #output2 = F.interpolate(output[1],size=(target.shape[1], target.shape[2]), mode='bilinear',align_corners=False)
            output = F.interpolate(output,size=(target.shape[1], target.shape[2]), mode='bilinear',align_corners=False)

            

            confmat.update(target, output.argmax(dim=1))
        confmat.reduce_from_all_processes()
        
    '''
    with torch.no_grad():
        loop2 = tqdm(test_loader, desc='Test')
        window = 224
        slide = 112
        for image, target in loop2:
            image, target = image.to(device), target.to(device)
            b,c,h,w = image.shape
            x = 4
            y = 4
            #x = h // slide + 1
            #y = w // slide + 1
            #expand_img = torch.zeros(b,c,(x+1) * slide, (y+1) * slide).cuda()
            #expand_img[:,:,:h,:w] = image
            #final_pred = torch.zeros(b,num_classes,(x+1) * slide, (y+1) * slide).cuda()
            final_pred = torch.zeros(b,num_classes,x * slide, y * slide).cuda()
            for i in range(3):
                for j in range(3):
                    pred = model(image[:,:,i * slide : i * slide + window,j * slide : j * slide + window])
                    final_pred[:,:,i * slide : i * slide + window,j * slide : j * slide + window] += (pred.softmax(dim=1))
            output = F.interpolate(final_pred,size=(target.shape[1], target.shape[2]), mode='bilinear',align_corners=False)
            confmat.update(target, output.argmax(dim=1))
        
            # 更新测试信息
        
        confmat.reduce_from_all_processes()
    '''
    return confmat
