import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16

import numpy as np
import torchvision

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

class L_exp(nn.Module):

    def __init__(self, patch_size, weight):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.patch_size = patch_size
        self.weight = weight

    def forward(self, x, exp):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        # print(x.shape)
        mean = self.pool(x)
            
        return torch.mean(torch.pow(mean - exp,2)) * self.weight

class L_TV(nn.Module):
    def __init__(self, mid_val=None):
        super(L_TV,self).__init__()
        self.mid_val = mid_val

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        if self.mid_val is None:
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        else:
            h_tv = torch.pow(torch.clamp(self.mid_val - torch.abs(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]) - self.mid_val), 0, 1), 2).sum()
            w_tv = torch.pow(torch.clamp(self.mid_val - torch.abs(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]) - self.mid_val), 0, 1), 2).sum()
        return 2 * (h_tv/count_h+w_tv/count_w)/batch_size

class L_sim(nn.Module):
    def __init__(self):
        super(L_sim,self).__init__()
    
    def forward(self, feats1, feats2):
        loss = 0.
        for i in range(len(feats1)):
            feats1[i] = torch.nn.functional.normalize(feats1[i], dim=1)
            feats2[i] = torch.nn.functional.normalize(feats2[i], dim=1)
            loss += (feats1[i] * feats2[i]).sum(dim=1).mean()
            
        loss /= len(feats1)

        return loss

class L_down(nn.Module):
    def __init__(self):
        super(L_down,self).__init__()
    
    def forward(self, x):

        loss = (1-x).mean()

        return loss


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3