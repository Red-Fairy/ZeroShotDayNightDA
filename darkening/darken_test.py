import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import time
from model import enhance_net_nopool_v2 as enhance_net_nopool # we use v2 in our paper (quadratic curve)
# from model import enhance_net_nopool_v3 as enhance_net_nopool # v3 is the reciprocal curve
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from utils.data import ImageDataset
from utils.utils import *
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser(description='DCE')

parser.add_argument('--src_path', default=None, required=True, type=str)
parser.add_argument('--dst_path', default=None, required=True, type=str)
parser.add_argument('--experiment', default=None, required=True, type=str)

parser.add_argument('--exp_range', type=str, default='0-0.2')
parser.add_argument('--curve_round', type=int, default=8)
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--noise_mode', type=str, default='pixel_patch', help='add noise to input')
parser.add_argument('--noise_intensity', type=float, default=0.025, help='add noise to input')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

DCE_net = enhance_net_nopool(scale_factor=1, curve_round=args.curve_round, encode_dim=1).cuda()

DCE_net.load_state_dict(torch.load(os.path.join('checkpoints', args.experiment, 'Epoch15.pth')))
print("Loaded state dict from {}".format(os.path.join('checkpoints', args.experiment, 'Epoch15.pth')))

DCE_net.eval()
DCE_net.requires_grad = False
low_exp, high_exp = args.exp_range.split('-')
low_exp, high_exp = float(low_exp), float(high_exp)
downsample = 64

def darken(image, path):

	start = time.time()

	exp = np.random.uniform(low_exp, high_exp)
	expt = torch.tensor(exp).cuda().expand(image.shape[0], 1, image.shape[2], image.shape[3])
	if args.noise_mode is not None:
		noise = torch.zeros_like(expt)
		if 'pixel' in args.noise_mode:
			noise_raw = torch.randn_like(expt) * args.noise_intensity
			noise = noise + noise_raw
		if 'patch' in args.noise_mode:
			noise_raw = torch.randn(image.shape[0], 1, image.shape[2] // downsample, image.shape[3] // downsample).cuda() * args.noise_intensity
			noise_raw = TF.resize(noise_raw, (image.shape[2], image.shape[3]), interpolation=Image.BILINEAR)
			noise = noise + noise_raw
		expt = expt + noise
	enhanced_image, params_maps = DCE_net(image, expt)

	filename = os.path.join(*path.split('/')[-2:])
	dst_path = os.path.join(args.dst_path, filename)
	os.makedirs(os.path.dirname(dst_path), exist_ok=True)

	torchvision.utils.save_image(enhanced_image, dst_path)

	return time.time() - start


if __name__ == '__main__':

	with torch.no_grad():
		train_trans = transforms.Compose([
			transforms.Resize((512, 1024)),
			transforms.ToTensor(),
		])

		train_dataset = ImageDataset(args.src_path, transform=train_trans, path=True)

		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
		
		for data, path in train_loader:
			data = data.cuda()
			darken(data, path[0])
