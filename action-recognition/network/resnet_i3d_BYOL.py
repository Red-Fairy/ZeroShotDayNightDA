from torch import nn
from torch.nn import init
import torch
from network.resnet_i3d import Res_I3D

class Res_I3D_BYOL(nn.Module):

	def __init__(self, m=0.99, **kwargs):
		# TODO fix bn_running_stats
		self.m = m

		super(Res_I3D_BYOL, self).__init__()

		self.encoder_q = Res_I3D(**kwargs)
		self.encoder_k = Res_I3D(**kwargs)

		out_channel = 64
		self.head_q = proj_head(
			in_channel=2048, out_channel=out_channel)
		self.head_k = proj_head(
			in_channel=2048, out_channel=out_channel)
		self.pred = pred_head(out_channel=out_channel)

		self._init_encoder_k()

	@torch.no_grad()
	def _init_encoder_k(self):
		for param_k in self.encoder_k.parameters():
			param_k.requires_grad = False  # not update by gradient
		for param_k in self.head_k.parameters():
			param_k.requires_grad = False  # not update by gradient

		try:
			for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
		except:
			for param_q, param_k in zip(self.encoder_q.module.parameters(), self.encoder_k.parameters()):
				param_k.data.copy_(param_q.data)  # initialize
		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		"""
		Momentum update of the key encoder
		"""
		try:
			for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
				param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
		except:
			for param_q, param_k in zip(self.encoder_q.module.parameters(), self.encoder_k.parameters()):
				param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	def _momentum_update_key_head(self):
		for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	def forward(self, x, y=None):

		if y is None:
			q = self.encoder_q(x)
			return q

		else:
			# compute query features
			out_q1, feat_q1 = self.encoder_q(x, return_feat=True)
			out_q2, feat_q2 = self.encoder_q(y, return_feat=True)

			feat_q1 = self.pred(self.head_q(feat_q1))
			feat_q1 = nn.functional.normalize(feat_q1, dim=1)
			feat_q2 = self.pred(self.head_q(feat_q2))
			feat_q2 = nn.functional.normalize(feat_q2, dim=1)

			with torch.no_grad():
				feat_k1 = self.head_k(self.encoder_k(y, return_feat=True)[1])
				feat_k1 = nn.functional.normalize(feat_k1, dim=1)

				feat_k2 = self.head_k(self.encoder_k(x, return_feat=True)[1])
				feat_k2 = nn.functional.normalize(feat_k2, dim=1)
			
			self._momentum_update_key_encoder()
			return out_q1, out_q2, feat_q1, feat_q2, feat_k1, feat_k2



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
