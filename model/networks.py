import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from model.decoders import dual_decoder


class FoodSwinV2(nn.Module):
    def __init__(self, num_classes=103, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu'):
        super(FoodSwinV2, self).__init__()

        self.down = nn.AvgPool2d(kernel_size=2,stride=2)
        
        # backbone network initialization with pretrained weight

        self.backbone = timm.create_model('swinv2_large_window12to16_192to256',pretrained=True,features_only=True)
        
        print(self.backbone.default_cfg)

        channels = [192,384,768,1536]

        
        #   decoder initialization
        
        self.decoder = dual_decoder(channels=channels,num_classes=num_classes, lgag_ks=lgag_ks, activation=activation)

        
        
    def forward(self, x,semi=False):
        
        # if grayscale input, convert to 3 channels
        _,_,h,w = x.shape
        if x.size()[1] == 1:
            x = self.conv(x)
        #x = self.down(x)
        # encoder

        x1,x2,x3,x4 = self.backbone(x)
        

        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)
        x4 = x4.permute(0,3,1,2)
        
        #print(x1.shape, x2.shape, x3.shape, x4.shape)
        '''
        if(semi):
            x1 = nn.Dropout(0.2)(x1)
            x2 = nn.Dropout(0.2)(x2)
            x3 = nn.Dropout(0.2)(x3)
            x4 = nn.Dropout(0.2)(x4)
        '''
        # decoder
        dec_outs = self.decoder([x1,x2,x3,x4])

        p = F.interpolate(dec_outs, size=(h,w), mode="bilinear", align_corners=True)
        return p
    
               
     