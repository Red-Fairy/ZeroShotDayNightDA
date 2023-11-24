import cv2
import numpy as np
from network.zero_dce import enhance_net_nopool
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import os
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./dataset/NormalLight/raw/data/')
parser.add_argument('--output', type=str, default='./dataset/NormalLight/raw/data_darken_test/')
parser.add_argument('--file_txt', type=str, default='./dataset/NormalLight/raw/list_cvt/ARID_split1_train.txt')
parser.add_argument('--darkening_model', type=str, help='path the darkening model',default='')
parser.add_argument('--gpu_id', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def get_paths(file_txt):
	f = open(file_txt, 'r')
	paths = []
	for line in f:
		path = line.split('\t')[-1].strip()
		paths.append(os.path.join(args.input, path))
	return paths

paths = get_paths(args.file_txt)
paths.sort()

model = enhance_net_nopool(1, curve_round=8).cuda().eval()
model.requires_grad_(False)
model.load_state_dict(torch.load(args.darkening_model, map_location='cpu'))

for path in tqdm(paths):
	videoCapture = cv2.VideoCapture(path)

	outpath = os.path.join(args.output, path.split('/')[-1])

	success, frame = videoCapture.read()
	fps = videoCapture.get(cv2.CAP_PROP_FPS)
	
	if path.split('/')[-1].endswith('.avi'):
		videowriter = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame.shape[1], frame.shape[0]))
	else:
		videowriter = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
 
	exp = np.random.uniform(0, 0.2)
	expt = torch.tensor(exp).cuda().expand(1, 1, frame.shape[0], frame.shape[1])

	noise_pixel = torch.randn_like(expt).cuda() * 0.025
	noise_patch = torch.randn(1, 1, frame.shape[0] // 16, frame.shape[1] // 16).cuda() * 0.025
	noise_patch = TF.resize(noise_patch, (frame.shape[0], frame.shape[1]), interpolation=Image.BILINEAR)
	noise = noise_patch + noise_pixel
	expt = expt + noise

	while success:
		# to tensor
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = to_tensor(frame).unsqueeze(0).cuda()

		frame, _  = model(frame, expt)
		# to numpy
		frame = to_pil_image(frame.squeeze(0).cpu())
		frame = np.array(frame)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		videowriter.write(frame)
		success, frame = videoCapture.read()

	videowriter.release()
	# print(path)