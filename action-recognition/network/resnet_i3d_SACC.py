from .resnet_i3d import Res_I3D
from torch import nn

import torch
import torch.nn.init as init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



class integration(nn.Module):
    def __init__(self, degree=2, P=256, C=3):
        super(integration, self).__init__()

        self.accumulation = nn.Conv1d(in_channels=P-1,
                                       out_channels=P,
                                       kernel_size=1,
                                       bias=False)
        self.P = P
        self.C = C
        self.degree = degree
        self.weights_init()

        for param in self.accumulation.parameters():
            param.requires_grad = False

    def weights_init(self):
        base_accumulation = torch.zeros(self.P-1, self.P-1)
        for i in range(self.P-1):
            base_accumulation[i, i:] = 1

        final_accumulation = torch.eye(self.P-1, self.P-1)
        for j in range(self.degree-1):
            final_accumulation = torch.matmul(final_accumulation, base_accumulation)

        last_accumulation = torch.zeros(self.P, self.P-1)
        for i in range(1, self.P):
            last_accumulation[i, :i] = 1

        final_accumulation = torch.matmul(last_accumulation, final_accumulation)
        self.accumulation.weight.data[:,:,0] = final_accumulation

    def forward(self, curve_grad):
        curve_grad = torch.reshape(curve_grad, (curve_grad.shape[0], self.P-1, self.C))
        curve = self.accumulation(curve_grad)

        # normalize to [0,1]
        curve = curve.permute(0,2,1)
        curve /= (curve.max(2, keepdim=True)[0] + 1e-10)
        return curve, curve_grad.permute(0,2,1)


class EnhanceNet(nn.Module):

    def __init__(self, degree, P=256, C=3, grad_loss_degree=1):
        super(EnhanceNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = torch.nn.Upsample(size=(16, 16))

        number_f = 3 * 8
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)

        self.d_conv1 = nn.Conv2d(number_f,number_f*2,3,2,1,bias=True)
        self.d_conv2 = nn.Conv2d(number_f*2,number_f*4,3,2,1,bias=True)
        self.d_linear = nn.Linear(number_f*4, (P-1)*C, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

        self.integration = integration(degree=degree, P=P, C=C)

        # Record degree by a buffer
        self.register_buffer("degree", torch.zeros(degree))

        self.P = P
        self.grad_loss_degree = grad_loss_degree

    def init_parms(self,):
        init_weights(self.e_conv1)
        init_weights(self.e_conv2)
        init_weights(self.e_conv3)
        init_weights(self.e_conv4)
        init_weights(self.e_conv5)
        init_weights(self.d_conv1)
        init_weights(self.d_conv2)
        init_weights(self.d_linear)

    def apply_curve(self, image, curve):
        B, C, H, W = image.shape
        image = torch.floor(image * (self.P-1)).reshape(B*C, -1)
        curve = curve.reshape(B*C, -1)

        # Based on https://github.com/alex04072000/SingleHDR/blob/master/util.py
        def sample_1d(
            img,   # RF, [b, h, c]
            y_idx, # Img, [b, n]
        ):
            b, h = img.shape
            b, n = y_idx.shape
            
            b_idx = torch.arange(start=0, end=b)    # [b]
            b_idx = b_idx.unsqueeze(1)              # [b, 1]
            b_idx = b_idx.repeat((1, n))            # [b, n]
            
            a_idx = torch.cat([b_idx.long().unsqueeze(2).to(y_idx.device), 
                               y_idx.long().unsqueeze(2)], axis=2) # [b, n, 2]

            # Based on https://github.com/ncullen93/torchsample/blob/master/torchsample/utils.py
            def th_gather_nd(
                x,          # [b, h]
                coords      # [b, n, 2]
            ):
                coords_ = coords.reshape(-1, 2)
                coords_ = coords_[:, 0] * x.shape[1]  + coords_[:, 1]
                x_gather = torch.index_select(x.reshape(-1), 0, coords_)
                return x_gather

            return th_gather_nd(img, a_idx)

        result = sample_1d(curve, image).reshape(B,C,H,W)
        return result


    def forward(self, x_ori, return_curve=False, return_loss=False):
        B,C,T,H,W = x_ori.shape
        x_ori = x_ori.view((B,C,H*T,W))

        # Input is normalized
        mean = x_ori.new_tensor([0.43216, 0.394666, 0.37645]).view(-1, 1, 1)
        std = x_ori.new_tensor([0.22803, 0.22145, 0.216989]).view(-1, 1, 1)
        x_ori = x_ori * std + mean

        x_ori = torch.clamp(x_ori, 0, 1)
        x_small = self.downsample(x_ori)

        # 3 * 16 * 16
        x1 = self.relu(self.e_conv1(x_small))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(torch.cat([x2,x3],1)))
        x5 = self.relu(self.e_conv5(torch.cat([x1,x4],1)))

        x6 = self.d_conv1(x5)
        x7 = self.d_conv2(x6)

        # number_f*4 * 4 * 4
        x7 = torch.mean(x7, [2,3])
        curve_grad = self.relu(self.d_linear(x7))
        
        curve, curve_grad = self.integration(curve_grad)

        enhance_image = self.apply_curve(x_ori, curve)

        # Back to DSFD-style normalized
        enhance_image_dsfd = (enhance_image - mean) / std 
        enhance_image_dsfd = enhance_image_dsfd.view((B,C,T,H,W))

        # if return_loss:
        #     curve_grad_grad = torch.clip(curve_grad[:,:,1:] - curve_grad[:,:,:-1], 0)
        #     grad_loss = torch.mean(curve_grad_grad)
        #     return enhance_image_dsfd, grad_loss

        if return_curve:
            return enhance_image_dsfd, curve
        else:
            return enhance_image_dsfd


class Res_I3D_SACC(nn.Module):
    
    def __init__(self, **kwargs) -> None:
        super(Res_I3D_SACC, self).__init__()
        
        self.EnhNet = EnhanceNet(degree=2)
    
        self.DetNet = Res_I3D(**kwargs)
        
    def forward(self, x):
        return self.DetNet(self.EnhNet(x))