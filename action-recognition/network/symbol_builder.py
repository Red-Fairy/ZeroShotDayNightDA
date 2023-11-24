import logging
import torch

# Lots of different networks
from .c3d import C3D
from .inception_v1_i3d import InceptionV1_I3D
from .resnet_i3d import Res_I3D
from .resnet_i3d_BYOL import Res_I3D_BYOL
# from .resnet_i3d_SACC import Res_I3D_SACC
if torch.__version__ >= '1.2.0':
	from .resnet_3d import RESNET18  # This require Pytorch >= 1.2.0 support

from .config import get_config

def get_symbol(name, is_dark=False, print_net=False, **kwargs):
	
	if name.upper() == "RESNET" and torch.__version__ >= '1.2.0':
		net = RESNET18(**kwargs)
	elif name.upper() == "C3D":
		net = C3D(**kwargs)
	elif name.upper() == "RES_I3D":
		net = Res_I3D(**kwargs)
	elif name.upper() == "I3D":
		net = InceptionV1_I3D(**kwargs)
	elif name.upper() == "RES_I3D_BYOL":
		net = Res_I3D_BYOL(**kwargs)
	# elif name.upper() == "RES_I3D_SACC":
	# 	net = Res_I3D_SACC(**kwargs)
	else:
		logging.error("network '{}'' not implemented".format(name))
		raise NotImplementedError()

	if print_net:
		logging.debug("Symbol:: Network Architecture:")
		logging.debug(net)

	input_conf = get_config(name, is_dark, **kwargs)
	return net, input_conf

