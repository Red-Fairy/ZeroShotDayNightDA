# Require Pytorch Version >= 1.2.0
import logging
import os
from collections import OrderedDict

import torch.nn as nn
import torch
import torchvision

try:
	from . import initializer
	from .utils import load_state
except: 
	import initializer
	from utils import load_state


class RESNET18(nn.Module):

	def __init__(self, num_classes, pretrained=True, pool_first=True, **kwargs):
		super(RESNET18, self).__init__()

		self.resnet = torchvision.models.video.r3d_18(pretrained=False, progress=False, num_classes=num_classes, **kwargs)

		#############
		# Initialization
		initializer.xavier(net=self)

		if pretrained:
			pretrained_model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrained/r3d_18-b3b3357e.pth')
			logging.info("Network:: graph initialized, loading pretrained model: `{}'".format(pretrained_model))
			assert os.path.exists(pretrained_model), "cannot locate: `{}'".format(pretrained_model)
			pretrained = torch.load(pretrained_model)
			# load_state(self.resnet, pretrained['state_dict'])
			load_state(self.resnet, pretrained)
		else:
			logging.info("Network:: graph initialized, use random inilization!")

	def forward(self, x):

		h = self.resnet(x)

		return h

if __name__ == "__main__":
	import torch
	logging.getLogger().setLevel(logging.DEBUG)
	# ---------
	net = RESNET18(num_classes=100, pretrained=True)
	data = torch.autograd.Variable(torch.randn(1,3,16,224,224))
	output = net(data)
	# torch.save({'state_dict': net.state_dict()}, './tmp.pth')
	print (output.shape)
