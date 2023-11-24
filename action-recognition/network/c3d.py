import logging
import os
from collections import OrderedDict

import torch.nn as nn
import torch
import torchvision

try:
	from . import initializer
	# from .utils import load_state
except: 
	import initializer
	# from utils import load_state

def load_state(network, state_dict, corresp_name):
	# customized partialy load function
	net_state_keys = list(network.state_dict().keys())
	for name, param in state_dict.items():
		if name.startswith('classifier') or name.startswith('fc') or name.startswith('nlfc'):
			continue
		if name in corresp_name:
			dst_param_shape = network.state_dict()[corresp_name[name]].shape
			if param.shape == dst_param_shape:
				network.state_dict()[corresp_name[name]].copy_(param.view(dst_param_shape))
				net_state_keys.remove(corresp_name[name])
	# indicating missed keys
	if net_state_keys:
		num_batches_list = []
		for i in range(len(net_state_keys)):
			if 'num_batches_tracked' in net_state_keys[i]:
				num_batches_list.append(net_state_keys[i])
		pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]
		logging.info("There are layers in current network not initialized by pretrained")
		logging.warning(">> Failed to load: {}".format(pruned_additional_states))
		return False
	return True


class C3D(nn.Module):
	"""
	The C3D network.
	"""
	def __init__(self, num_classes, pretrained=True):
		super(C3D, self).__init__()

		self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

		self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

		self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

		self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

		self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
		self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

		self.fc6 = nn.Linear(8192, 4096)
		self.fc7 = nn.Linear(4096, 4096)
		self.fc8 = nn.Linear(4096, num_classes)

		self.dropout = nn.Dropout(p=0.5)

		self.relu = nn.ReLU()

		#############
		# Initialization

		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/c3d_pretrained.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			corresp_name = self.__load_pretrained_weights(pretrained_model=pretrained_model)
			load_state(self, state_dict=pretrained['state_dict'], corresp_name=corresp_name)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		x = self.relu(self.conv1(x))
		x = self.pool1(x)

		x = self.relu(self.conv2(x))
		x = self.pool2(x)

		x = self.relu(self.conv3a(x))
		x = self.relu(self.conv3b(x))
		x = self.pool3(x)

		x = self.relu(self.conv4a(x))
		x = self.relu(self.conv4b(x))
		x = self.pool4(x)

		x = self.relu(self.conv5a(x))
		x = self.relu(self.conv5b(x))
		x = self.pool5(x)

		x = x.view(-1, 8192)
		x = self.relu(self.fc6(x))
		x = self.dropout(x)
		x = self.relu(self.fc7(x))
		x = self.dropout(x)

		logits = self.fc8(x)

		return logits

	def __load_pretrained_weights(self, pretrained_model):
		"""Initialiaze network."""
		corresp_name = {
						# Conv1
						"features.0.weight": "conv1.weight",
						"features.0.bias": "conv1.bias",
						# Conv2
						"features.3.weight": "conv2.weight",
						"features.3.bias": "conv2.bias",
						# Conv3a
						"features.6.weight": "conv3a.weight",
						"features.6.bias": "conv3a.bias",
						# Conv3b
						"features.8.weight": "conv3b.weight",
						"features.8.bias": "conv3b.bias",
						# Conv4a
						"features.11.weight": "conv4a.weight",
						"features.11.bias": "conv4a.bias",
						# Conv4b
						"features.13.weight": "conv4b.weight",
						"features.13.bias": "conv4b.bias",
						# Conv5a
						"features.16.weight": "conv5a.weight",
						"features.16.bias": "conv5a.bias",
						 # Conv5b
						"features.18.weight": "conv5b.weight",
						"features.18.bias": "conv5b.bias",
						# fc6
						"classifier.0.weight": "fc6.weight",
						"classifier.0.bias": "fc6.bias",
						# fc7
						"classifier.3.weight": "fc7.weight",
						"classifier.3.bias": "fc7.bias",
						}

		return corresp_name

if __name__ == "__main__":
	import torch
	logging.getLogger().setLevel(logging.DEBUG)
	# ---------
	net = C3D(num_classes=100, pretrained=True)
	data = torch.autograd.Variable(torch.randn(1,3,16,112,112))
	data = data.cuda().contiguous()
	net = net.cuda()
	output = net(data)
	# torch.save({'state_dict': net.state_dict()}, './tmp.pth')
	print (output.shape)
