import torch
from torch import nn
import argparse
import os
from torchvision import transforms
from utils.utils import *
from utils.resnet_BYOL import ResNet_BYOL
from utils.resnet import resnet18
from codan import CODaN
from utils.zero_model import *

parser = argparse.ArgumentParser(description='Zero Shot Day Night Domain Adaptation -- Classification')

parser.add_argument('--checkpoint', type=str, required=True,
					help='location for checkpoint')

parser.add_argument('--batch-size', default=32, type=int, help='batch size')                    
parser.add_argument('--gpu_id', default='0', type=str)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def validate(model,test_day_loader, test_night_loader):
	model.eval()
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	top1 = AverageMeter()
	losses = AverageMeter()

	acc = []
	for loader in [test_day_loader,test_night_loader]:
		losses.reset()
		top1.reset()
		for images, labels in loader:
			images = images.cuda()
			labels = labels.cuda()

			with torch.no_grad():
				outputs = model(images)
				loss = criterion(outputs,labels)

			prec1 = accuracy(outputs.data, labels)[0]
			top1.update(prec1.item(), images.size(0))
			losses.update(float(loss.detach().cpu()))
		acc.append(top1.avg)

	return acc

def main():

	transforms_test = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
	test_day_dataset = CODaN(split='test_day',transform=transforms_test)
	test_night_dataset = CODaN(split='test_night',transform=transforms_test)

	test_night_loader = torch.utils.data.DataLoader(test_night_dataset,
			num_workers=16,
			batch_size=args.batch_size,
			shuffle=False)
	test_day_loader = torch.utils.data.DataLoader(test_day_dataset,
			num_workers=16,
			batch_size=args.batch_size,
			shuffle=False)    

	state_dict = torch.load(args.checkpoint)

	resnet = resnet18(num_classes=10)
	model = ResNet_BYOL(num_classes=10, resnet=resnet)
	model.load_state_dict(state_dict['state_dict'], strict=True)
	model.cuda()

	acc = validate(model, test_day_loader, test_night_loader)

	print('Daytime accuracy: {:.2f}%'.format(acc[0]))
	print('Nighttime accuracy: {:.2f}%'.format(acc[1]))

if __name__ == '__main__':
	main()
