"""RefineNet

RefineNet PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from wsgiref import headers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.resnet import resnet101


def batchnorm(in_planes):
	"batch norm 2d"
	return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
	"1x1 convolution"
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
					 padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
	"conv-batchnorm-relu"
	if act:
		return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
							 batchnorm(out_planes),
							 nn.ReLU6(inplace=True))
	else:
		return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups, bias=False),
							 batchnorm(out_planes))


class CRPBlock(nn.Module):

	def __init__(self, in_planes, out_planes, n_stages):
		super(CRPBlock, self).__init__()
		for i in range(n_stages):
			setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
					conv3x3(in_planes if (i == 0) else out_planes,
							out_planes, stride=1,
							bias=False))
		self.stride = 1
		self.n_stages = n_stages
		self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

	def forward(self, x):
		top = x
		for i in range(self.n_stages):
			top = self.maxpool(top)
			top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
			x = top + x
		return x


stages_suffixes = {0: '_conv',
				   1: '_conv_relu_varout_dimred'}


class RCUBlock(nn.Module):

	def __init__(self, in_planes, out_planes, n_blocks, n_stages):
		super(RCUBlock, self).__init__()
		for i in range(n_blocks):
			for j in range(n_stages):
				setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
						conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
								out_planes, stride=1,
								bias=(j == 0)))
		self.stride = 1
		self.n_blocks = n_blocks
		self.n_stages = n_stages

	def forward(self, x):
		for i in range(self.n_blocks):
			residual = x
			for j in range(self.n_stages):
				x = F.relu(x)
				x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
			x += residual
		return x


class RefineNet_BYOL(nn.Module):

	def __init__(self, num_classes, invariant=None, pretrained=True, bn_running_stats=True, m=0.99):
		# TODO fix bn_running_stats
		self.m = m

		super(RefineNet_BYOL, self).__init__()
		self.do = nn.Dropout(p=0.5)

		self.resnet = resnet101(
			pretrained=pretrained, invariant=invariant, scale=0.0, return_features=True)
		self.encoder_k = resnet101(
			pretrained=pretrained, invariant=invariant, scale=0.0, return_features=True)

		feat_out_channels = [256, 512, 1024, 2048]
		out_channel = 128
		self.head_q = ContrastiveHead(
			feat_out_channels=feat_out_channels, out_channel=out_channel)
		self.head_k = ContrastiveHead(
			feat_out_channels=feat_out_channels, out_channel=out_channel)
		self.pred = PredictionHead(heads=4, out_channel=out_channel)

		self._init_encoder_k()

		self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
		self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
		self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
		self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
		self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(
			512, 256, bias=False)
		self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
		self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
		self.adapt_stage2_b2_joint_varout_dimred = conv3x3(
			256, 256, bias=False)
		self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
		self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
		self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(
			256, 256, bias=False)

		self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
		self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
		self.adapt_stage3_b2_joint_varout_dimred = conv3x3(
			256, 256, bias=False)
		self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
		self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
		self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(
			256, 256, bias=False)

		self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
		self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
		self.adapt_stage4_b2_joint_varout_dimred = conv3x3(
			256, 256, bias=False)
		self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
		self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

		self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
								  padding=1, bias=True)

	@torch.no_grad()
	def _init_encoder_k(self):
		for param_k in self.encoder_k.parameters():
			param_k.requires_grad = False  # not update by gradient
		for param_k in self.head_k.parameters():
			param_k.requires_grad = False  # not update by gradient

		try:
			for param_q, param_k in zip(self.resnet.parameters(), self.encoder_k.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
		except:
			for param_q, param_k in zip(self.resnet.module.parameters(), self.encoder_k.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize

	def _make_crp(self, in_planes, out_planes, stages):
		layers = [CRPBlock(in_planes, out_planes, stages)]
		return nn.Sequential(*layers)

	def _make_rcu(self, in_planes, out_planes, blocks, stages):
		layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
		return nn.Sequential(*layers)

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		"""
		Momentum update of the key encoder
		"""
		try:
			for param_q, param_k in zip(self.resnet.parameters(), self.encoder_k.parameters()):
				param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		except:
			for param_q, param_k in zip(self.resnet.module.parameters(), self.encoder_k.parameters()):
				param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	@torch.no_grad()
	def _momentum_update_key_head(self):
		"""
		Momentum update of the key head
		"""
		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	def forward(self, x, y=None, proj=False, dual=False, train_head=False, similarity=False):
		if train_head:
			feats_q = list(self.resnet(x))
			feats_q2 = list(self.resnet(y))
			for i in range(len(feats_q)):
				feats_q[i] = nn.functional.normalize(feats_q[i], dim=1)
				feats_q2[i] = nn.functional.normalize(feats_q2[i], dim=1)
			return feats_q, feats_q2

		l1, l2, l3, l4 = self.resnet(x)
		if y is not None:
			# if not reverse_single: # normal single
			# 	feat_q = self.pred(self.head_q([l1,l2,l3,l4]))
			# 	for i in range(len(feat_q)):
			# 		feat_q[i] = nn.functional.normalize(feat_q[i], dim=1)
			# 	with torch.no_grad():
			# 		feat_k = self.head_k(self.encoder_k(y))
			# 		for i in range(len(feat_k)):
			# 			feat_k[i] = nn.functional.normalize(feat_k[i], dim=1)
			# else: # reverse single
			# 	feat_q = self.pred(self.head_q(self.resnet(y)))
			# 	for i in range(len(feat_q)):
			# 		feat_q[i] = nn.functional.normalize(feat_q[i], dim=1)
			# 	with torch.no_grad():
			# 		feat_k = self.head_k(self.encoder_k(x))
			# 		for i in range(len(feat_k)):
			# 			feat_k[i] = nn.functional.normalize(feat_k[i], dim=1)
			if similarity:
				feat_q = list([l1,l2,l3,l4])
				for i in range(len(feat_q)):
					feat_q[i] = nn.functional.normalize(feat_q[i], dim=1)
				with torch.no_grad():
					feat_k = list(self.resnet(y))
					for i in range(len(feat_k)):
						feat_k[i] = nn.functional.normalize(feat_k[i], dim=1)
			else:
				feat_q = self.pred(self.head_q([l1,l2,l3,l4]))
				for i in range(len(feat_q)):
					feat_q[i] = nn.functional.normalize(feat_q[i], dim=1)
				with torch.no_grad():
					feat_k = self.head_k(self.encoder_k(y))
					for i in range(len(feat_k)):
						feat_k[i] = nn.functional.normalize(feat_k[i], dim=1)	
				if dual:
					feat_q2 = self.pred(self.head_q(self.resnet(y)))
					for i in range(len(feat_q2)):
						feat_q2[i] = nn.functional.normalize(feat_q2[i], dim=1)
					with torch.no_grad():
						feat_k2 = self.head_k(self.encoder_k(x))
						for i in range(len(feat_k2)):
							feat_k2[i] = nn.functional.normalize(feat_k2[i], dim=1)
		
		l4 = self.do(l4)  # dropout
		l3 = self.do(l3)  # dropout

		x4 = self.p_ims1d2_outl1_dimred(l4)
		x4 = self.adapt_stage1_b(x4)
		x4 = F.relu(x4)
		x4 = self.mflow_conv_g1_pool(x4)
		x4 = self.mflow_conv_g1_b(x4)
		x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
		x4 = nn.Upsample(size=l3.size()[2:],
						mode='bilinear', align_corners=True)(x4)

		x3 = self.p_ims1d2_outl2_dimred(l3)
		x3 = self.adapt_stage2_b(x3)
		x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
		x3 = x3 + x4
		x3 = F.relu(x3)
		x3 = self.mflow_conv_g2_pool(x3)
		x3 = self.mflow_conv_g2_b(x3)
		x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
		x3 = nn.Upsample(size=l2.size()[2:],
						mode='bilinear', align_corners=True)(x3)

		x2 = self.p_ims1d2_outl3_dimred(l2)
		x2 = self.adapt_stage3_b(x2)
		x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
		x2 = x2 + x3
		x2 = F.relu(x2)
		x2 = self.mflow_conv_g3_pool(x2)
		x2 = self.mflow_conv_g3_b(x2)
		x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
		x2 = nn.Upsample(size=l1.size()[2:],
						mode='bilinear', align_corners=True)(x2)

		x1 = self.p_ims1d2_outl4_dimred(l1)
		x1 = self.adapt_stage4_b(x1)
		x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
		x1 = x1 + x2
		x1 = F.relu(x1)
		x1 = self.mflow_conv_g4_pool(x1)
		x1 = self.mflow_conv_g4_b(x1)

		out_size = (l1.size()[2]*4, l1.size()[3]*4)
		x0 = nn.Upsample(size=out_size, mode='bilinear',
						align_corners=True)(x1)
		x0 = self.do(x0)
		out = self.clf_conv(x0)

		if proj:
			return out, x1
		if y is not None:
			if dual:
				return out, feat_q, feat_k, feat_q2, feat_k2
			else:
				return out, feat_q, feat_k
		else:
			return out


if __name__ == '__main__':
	print('Printing RefineNet model definition, then exiting.')
	m = RefineNet_BYOL(20, pretrained=True,
					   bn_running_stats=True, invariant=None)
	print(m)
	x = torch.randn(2, 3, 1024, 2048)
	print(m(x).size())

class ContrastiveProjMLPV1(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm1d(out_channel)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(out_channel, out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)
		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.fc2.weight)


	def forward(self, x):
		x = self.conv1(x)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		x = self.bn1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x

class ContrastiveProjMLPV2(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(bottle_channel)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.conv1.weight)
		# init.kaiming_normal_(self.conv2.weight)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(self.bn1(x))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		return x

class ContrastiveProjMLPV3(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(bottle_channel)
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(bottle_channel, out_channel)
		self.bn2 = nn.BatchNorm1d(out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)
		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.conv1.weight)
		init.kaiming_normal_(self.fc2.weight)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(self.bn1(x))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = self.fc1(x.view(x.size(0), -1))
		x = self.relu(self.bn2(x))
		x = self.fc2(x)
		return x

class ContrastiveMLPConv(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal_(self.fc.weight)
		init.kaiming_normal_(self.conv1.weight)
		init.kaiming_normal_(self.conv2.weight)

	def forward(self, x):
		x = self.relu(self.conv2(self.relu(self.conv1(x))))
		x = F.adaptive_avg_pool2d(x, (1, 1))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


class ContrastiveHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel=128):
		super().__init__()
		self.single = len(feat_out_channels) == 1
		self.MLPs = []
		for in_channel in feat_out_channels:
			self.MLPs.append(ContrastiveProjMLPV3(in_channel, out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)

	def forward(self, feats, bp=True):
		if self.single:
			return self.MLPs[0](feats)
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs


class pred_head(nn.Module):
	def __init__(self, out_channel):
		super(pred_head, self).__init__()
		self.in_features = out_channel

		self.fc1 = nn.Linear(out_channel, out_channel)
		self.bn1 = nn.BatchNorm1d(out_channel)
		# self.fc2 = nn.Linear(out_channel, out_channel, bias=False)
		self.bn2 = nn.BatchNorm1d(out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)

		# init.kaiming_normal_(self.fc1.weight)
		# init.kaiming_normal_(self.fc2.weight)
		# init.eye_(self.fc1.weight)
		# init.eye_(self.fc2.weight)
  
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		# debug

		x = self.fc1(x)
		x = self.bn1(x)

		x = self.relu(x)

		x = self.fc2(x)
		# x = self.bn2(x)

		return x

class PredictionHead(nn.Module):
	def __init__(self, heads=4, out_channel=128):
		super().__init__()
		self.MLPs = []
		for i in range(heads):
			self.MLPs.append(pred_head(out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)

	def forward(self, feats, bp=True):
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs
