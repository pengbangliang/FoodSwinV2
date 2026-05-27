import torch
import torch.nn as nn

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

class DCA(nn.Module):
    def __init__(self, in_planes):
        super(DCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.fc3 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.lambda1 = nn.Parameter(torch.zeros(in_planes, dtype=torch.float32).normal_(mean=0,std=0.1)).reshape(1,in_planes,1,1).cuda()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out1 = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out1 = avg_out1 + max_out1

        avg_out2 = self.fc4(self.relu2(self.fc3(self.avg_pool(x))))
        max_out2 = self.fc4(self.relu2(self.fc3(self.max_pool(x))))
        out2 = avg_out2 + max_out2
        
        out = self.sigmoid(out1) - self.lambda1 * self.sigmoid(out2)

        return out


class DSA(nn.Module):
    def __init__(self, kernel_size=3):
        super(DSA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.conv2 = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.lambda2 = nn.Parameter(torch.zeros(1, dtype=torch.float32).normal_(mean=0,std=0.1)).reshape(1,1,1,1).cuda()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.sigmoid(x1) - self.lambda2 * self.sigmoid(x2)

        return x


class FPDA(nn.Module):
    def __init__(self, in_channel):
        super(FPDA, self).__init__()
        self.ca2 = DCA(in_channel // 2)
        self.sa2 = DSA()

        self.ca1 = DCA(in_channel)
        self.sa1 = DSA()


    def forward(self, x):

        x4 = self.ca1(x) * x
        x4 = self.sa1(x4) * x4
        
        x = channel_shuffle(x,groups=2)
        x0,x1 = x.chunk(2,dim=1)
        x0 = self.ca2(x0)*x0
        x1 = self.ca2(x1)*x1

        
        x = torch.cat([x0,x1],dim=1)
        
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        #x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa2(x0[-2]) * x0[-2], self.sa2(x0[-1]) * x0[-1]]

        #x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa2(x1[-2]) * x1[-2], self.sa2(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        
        x = x3 + x4

        return x


if __name__ == '__main__':

    input = torch.rand(1, 192, 8, 8).cuda()
    block = FPDA(in_channel=192).cuda()
    output = block(input)

    print(input.size())
    print(output.size())
    param_num = sum(p.numel() for p in block.parameters())
    print(param_num)
