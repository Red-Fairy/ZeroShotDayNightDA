import logging
import os
from collections import OrderedDict

import torch.nn as nn
import torch
import torch.utils.checkpoint as cp


def load_state(network, state_dict):
	# customized partialy load function
	net_state_keys = list(network.state_dict().keys())
	net_state_keys_copy = net_state_keys.copy()
	sup_string = ""
	for key in state_dict.keys():
		if "backbone" in key:
			sup_string = "backbone."
		elif "module" in key:
			sup_string = "module."

	for i, _ in enumerate(net_state_keys_copy):
		name = net_state_keys_copy[i]
		if name.startswith('classifier') or name.startswith('fc'):
			continue

		if not sup_string:
			name_pretrained = name
		else:
			name_pretrained = sup_string + name
		
		if name_pretrained in state_dict.keys():
			dst_param_shape = network.state_dict()[name].shape
			if state_dict[name_pretrained].shape == dst_param_shape:
				network.state_dict()[name].copy_(state_dict[name_pretrained].view(dst_param_shape))
				net_state_keys.remove(name)
	
	# indicating missed keys
	if net_state_keys:
		num_batches_list = []
		for i in range(len(net_state_keys)):
			if 'num_batches_tracked' in net_state_keys[i]:
				num_batches_list.append(net_state_keys[i])
		pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]
		if pruned_additional_states:
			logging.info("There are layers in current network not initialized by pretrained")
			logging.warning(">> Failed to load: {}".format(pruned_additional_states))
		return False
	return True


class SimpleSpatialTemporalModule(nn.Module):
	def __init__(self, spatial_type='avg', spatial_size=7, temporal_size=1):
		super(SimpleSpatialTemporalModule, self).__init__()

		assert spatial_type in ['avg']
		self.spatial_type = spatial_type

		self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
		self.temporal_size = temporal_size
		self.pool_size = (self.temporal_size, ) + self.spatial_size

		if self.spatial_type == 'avg':
			self.op = nn.AvgPool3d(self.pool_size, stride=1, padding=0)

	def init_weights(self):
		pass

	def forward(self, input):
		return self.op(input)


class ClsHead(nn.Module):
	"""Simplest classification head"""

	def __init__(self,
				 with_avg_pool=True,
				 temporal_feature_size=1,
				 spatial_feature_size=7,
				 dropout_ratio=0.8,
				 in_channels=2048,
				 num_classes=101,
	 init_std=0.01):
	
		super(ClsHead, self).__init__()

		self.with_avg_pool = with_avg_pool
		self.dropout_ratio = dropout_ratio
		self.in_channels = in_channels
		self.dropout_ratio = dropout_ratio
		self.temporal_feature_size = temporal_feature_size
		self.spatial_feature_size = spatial_feature_size
		self.init_std = init_std

		if self.dropout_ratio != 0:
			self.dropout = nn.Dropout(p=self.dropout_ratio)
		else:
			self.dropout = None
		if self.with_avg_pool:
			self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size))
		self.fc_cls = nn.Linear(in_channels, num_classes)

	def init_weights(self):
		nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
		nn.init.constant_(self.fc_cls.bias, 0)

	def forward(self, x):
		if x.ndimension() == 4:
			x = x.unsqueeze(2)
		assert x.shape[1] == self.in_channels
		assert x.shape[2] == self.temporal_feature_size
		assert x.shape[3] == self.spatial_feature_size
		assert x.shape[4] == self.spatial_feature_size
		if self.with_avg_pool:
			x = self.avg_pool(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = x.view(x.size(0), -1)

		cls_score = self.fc_cls(x)
		return cls_score

	def loss(self,
			 cls_score,
			 labels):
		losses = dict()
		losses['loss_cls'] = F.cross_entropy(cls_score, labels)

		return losses


class _SimpleConsensus(torch.autograd.Function):
	"""Simplest segmental consensus module"""

	def __init__(self,
				 consensus_type='avg',
				 dim=1):
		super(_SimpleConsensus, self).__init__()

		assert consensus_type in ['avg']
		self.consensus_type = consensus_type
		self.dim = dim
		self.shape = None

	def forward(self, x):
		self.shape = x.size()
		if self.consensus_type == 'avg':
			output = x.mean(dim=self.dim, keepdim=True)
		else:
			output = None
		return output

	def backward(self, grad_output):
		if self.consensus_type == 'avg':
			grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
		else:
			grad_in = None
		return grad_in


class SimpleConsensus(nn.Module):
	def __init__(self, consensus_type, dim=1):
		super(SimpleConsensus, self).__init__()

		assert consensus_type in ['avg']
		self.consensus_type = consensus_type
		self.dim = dim

	def init_weights(self):
		pass

	def forward(self, input):
		return _SimpleConsensus(self.consensus_type, self.dim)(input)


class SimpleSpatialModule(nn.Module):
	def __init__(self, spatial_type='avg', spatial_size=7):
		super(SimpleSpatialModule, self).__init__()

		assert spatial_type in ['avg']
		self.spatial_type = spatial_type

		self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)

		if self.spatial_type == 'avg':
			self.op = nn.AvgPool2d(self.spatial_size, stride=1, padding=0)


	def init_weights(self):
		pass

	def forward(self, input):
		return self.op(input)