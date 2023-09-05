import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class CSDN_Tem(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(CSDN_Tem, self).__init__()
		self.depth_conv = nn.Conv2d(
			in_channels=in_ch,
			out_channels=in_ch,
			kernel_size=3,
			stride=1,
			padding=1,
			groups=in_ch
		)
		self.point_conv = nn.Conv2d(
			in_channels=in_ch,
			out_channels=out_ch,
			kernel_size=1,
			stride=1,
			padding=0,
			groups=1
		)

	def forward(self, input):
		out = self.depth_conv(input)
		out = self.point_conv(out)
		return out

class enhance_net_nopool_v2(nn.Module):

	def __init__(self,scale_factor,curve_round=8, down_scale=0.95, encode_dim=1):
		super(enhance_net_nopool_v2, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		self.scale_factor = scale_factor
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		number_f = 64
		
		self.register_buffer('down_scale', torch.tensor(down_scale))
		self.register_buffer('curve_round', torch.tensor(curve_round, dtype=torch.long))
		self.register_buffer('encode_dim', torch.tensor(encode_dim, dtype=torch.long))

		self.e_conv0 = CSDN_Tem(3+self.encode_dim,number_f)
		self.e_conv1 = CSDN_Tem(number_f,number_f)
		self.e_conv2 = CSDN_Tem(number_f,number_f) 
		self.e_conv3 = CSDN_Tem(number_f,number_f) 
		self.e_conv4 = CSDN_Tem(number_f,number_f) 
		self.e_conv5 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv6 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv7 = CSDN_Tem(number_f*2,6)

	def enhance(self, x, x_r, down_scale):

		x = x * down_scale

		for _ in range(self.curve_round.data):
			x = x + x_r*(torch.pow(x,2)-x)

		x = x / down_scale

		return x
		
	def forward(self, x, exp):
		if self.scale_factor==1:
			x_down = x
		else:
			x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')

		b,c,h,w = x.size()
		exp_mat = exp.cuda().expand(b,1,h,w)
		
		def encode_sinusoid(x, encode_dim): # x in [0, 0.5]
			encode_x = []
			x = torch.pow(10, -10*x) # x in [1, 0]
			for i in range(encode_dim // 2):
				encode_x.append(torch.sin(x * math.pi * (2 ** i)))
				encode_x.append(torch.cos(x * math.pi * (2 ** i)))
			encode_x = torch.cat(encode_x, dim=1)
			return encode_x

		if self.encode_dim > 1:
			exp_mat = encode_sinusoid(exp_mat, self.encode_dim.data)

		# exp_mat = self.condition_conv(exp_mat)

		x_down = self.relu(self.e_conv0(torch.cat([x_down, exp_mat], dim=1)))

		x1 = self.relu(self.e_conv1(x_down))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		# x_mid = self.relu(self.e_conv4(x3))

		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		x7 = self.e_conv7(torch.cat([x1,x6],1))

		x_r = torch.tanh(x7[:,:3,:,:])
		down_scale = torch.sigmoid(x7[:,3:,:,:]) * (1 - self.down_scale.data) + self.down_scale.data
		
		# print(down_scale.shape)
		# exit()

		if self.scale_factor==1:
			x_r = x_r
		else:
			x_r = self.upsample(x_r)
		enhance_image = self.enhance(x,x_r,down_scale)
		return enhance_image, [x_r, down_scale]

	def load_state_dict(self, state_dict, strict: bool = True):
		self.encode_dim = state_dict['encode_dim']
		self.e_conv0 = CSDN_Tem(3+self.encode_dim,64).to(self.e_conv0.depth_conv.weight.device)
		return super().load_state_dict(state_dict, strict)

