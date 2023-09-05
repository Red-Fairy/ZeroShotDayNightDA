from wsgiref import headers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils.resnet import resnet18


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


class ResNet_BYOL(nn.Module):

	def __init__(self, resnet, num_classes, pretrained=False, m=0.99):
		# TODO fix bn_running_stats
		self.m = m

		super(ResNet_BYOL, self).__init__()

		self.resnet = resnet
		self.encoder_k = resnet18(
			pretrained=pretrained, num_classes=num_classes)

		out_channel = 128
		self.head_q = proj_head(
			in_channel=512, out_channel=out_channel)
		self.head_k = proj_head(
			in_channel=512, out_channel=out_channel)
		self.pred = pred_head(out_channel=out_channel)

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

	def _momentum_update_key_head(self):
		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	def forward(self, x, y=None, dual=False, train_head=False, update=True, similarity=False, proj=False):
		
		if proj:
			return self.head_q(self.resnet(x, proj=True)[1])

		if similarity:
			pred, feat = self.resnet(x, proj=True)
			feat = nn.functional.normalize(feat, dim=1)
			pred_2, feat_2 = self.resnet(y, proj=True)
			feat_2 = nn.functional.normalize(feat_2, dim=1)
			return pred, pred_2, feat, feat_2

		if dual or train_head:
			pred, feat = self.resnet(x, proj=True)
			pred2, feat2 = self.resnet(y, proj=True)
			feat_q = self.pred(self.head_q(feat))
			feat_q = nn.functional.normalize(feat_q, dim=1)

			feat_q2 = self.pred(self.head_q(feat2))
			feat_q2 = nn.functional.normalize(feat_q2, dim=1)

			with torch.no_grad():
				feat_k = self.head_k(self.encoder_k(y, proj=True)[1])
				feat_k = nn.functional.normalize(feat_k, dim=1)

			with torch.no_grad():
				feat_k2 = self.head_k(self.encoder_k(x, proj=True)[1])
				feat_k2 = nn.functional.normalize(feat_k2, dim=1)

			if dual:
				self._momentum_update_key_encoder()
				return pred, pred2, feat_q, feat_k, feat_q2, feat_k2
			else:
				if update:
					self._momentum_update_key_head()
				return feat_q, feat_k, feat_q2, feat_k2
			
		else:
			return self.resnet(x, proj=False)


if __name__ == '__main__':
	print('Printing RefineNet model definition, then exiting.')
	m = ResNet_BYOL(20, pretrained=True,
					   bn_running_stats=True, invariant=None)
	print(m)
	x = torch.randn(2, 3, 1024, 2048)
	print(m(x).size())

class proj_head(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(proj_head, self).__init__()

		self.fc1 = nn.Linear(in_channel, out_channel)
		self.bn1 = nn.BatchNorm1d(out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)
		self.relu = nn.ReLU(inplace=True)

		init.kaiming_normal_(self.fc1.weight)
		init.kaiming_normal_(self.fc2.weight)

	def forward(self, x,):

		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.fc2(x)

		return x

class pred_head(nn.Module):
	def __init__(self, out_channel):
		super(pred_head, self).__init__()
		self.in_features = out_channel

		self.fc1 = nn.Linear(out_channel, out_channel)
		self.bn1 = nn.BatchNorm1d(out_channel)
		self.fc2 = nn.Linear(out_channel, out_channel)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x,):

		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.fc2(x)

		return x
