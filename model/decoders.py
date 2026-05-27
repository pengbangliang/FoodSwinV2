import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from model.FPDA import FPDA
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim,kernel_size=1)


    def forward(self, x):
        x = self.proj(x)
        return x

class SegHead(nn.Module):
    def __init__(self,channels = [1024,512,256,128],embedding_dim=256 ,num_classes = 103):
        super(SegHead, self).__init__()

        self.linear_c1 = MLP(input_dim=channels[0], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=channels[1], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=channels[2], embed_dim=embedding_dim)
        self.linear_c4 = MLP(input_dim=channels[3], embed_dim=embedding_dim)


        self.linear_fuse1 = nn.Sequential(nn.Conv2d(embedding_dim*2,embedding_dim,kernel_size=1),
                                            nn.BatchNorm2d(embedding_dim),
                                            nn.LeakyReLU(),
                                        )
        self.linear_fuse2 = nn.Sequential(nn.Conv2d(embedding_dim*2,embedding_dim,kernel_size=1),
                                            nn.BatchNorm2d(embedding_dim),
                                            nn.LeakyReLU(),
                                        )
        self.linear_pred = nn.Conv2d(embedding_dim,num_classes,kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x  # len=4, 1/4,1/8,1/16,1/32

        n, _, h, w = c1.shape
        ############## MLP decoder on C1-C4 ###########
        c1 = self.linear_c1(c1)
        c2 = self.linear_c2(c2)
        c3 = self.linear_c3(c3)
        c4 = self.linear_c4(c4)

        c2 = F.interpolate(c2, size=(h,w), mode="bilinear", align_corners=True)
        c3 = F.interpolate(c3, size=(h,w), mode="bilinear", align_corners=True)
        c4 = F.interpolate(c4, size=(h,w), mode="bilinear", align_corners=True)

        x1 = self.linear_fuse1(torch.cat([c1,c2],dim=1))
        x2 = self.linear_fuse2(torch.cat([c3,c4],dim=1))
        x = self.linear_pred(x1+x2)
        return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

#   Efficient multi-scale convolutional attention decoding (EMCAD)
class dual_decoder(nn.Module):
    def __init__(self, channels=[512,320,128,64],num_classes = 103,  lgag_ks=3, activation='relu6'):
        super(dual_decoder,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        
        self.p1 = FPDA(channels[0])
        self.p2 = FPDA(channels[1])
        self.p3 = FPDA(channels[2])
        self.p4 = FPDA(channels[3])
        self.head = SegHead(channels=channels,num_classes=num_classes)
     
    def forward(self, x):
        x1,x2,x3,x4 = x
        c1 = self.p1(x1)
        c2 = self.p2(x2)
        c3 = self.p3(x3)
        c4 = self.p4(x4)
        p = self.head([c1,c2,c3,c4])

        return p
