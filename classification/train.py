import torch
from torch import nn
# from torchvision.models import resnet18
import argparse
import os
import time
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.resnet_BYOL import ResNet_BYOL
from utils.resnet import resnet18
from codan import CODaN
from tqdm import tqdm
from torch.backends import cudnn
from utils.zero_model import *
import random
from utils.dataset import AlignedDataset
import torchvision.transforms.functional as TF
from PIL import Image

parser = argparse.ArgumentParser(description='Zero Shot Day Night Domain Adaptation -- Classification')
parser.add_argument('--experiment', type=str,
					help='location for saving trained models')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/baseline_resnet/model_best.pt',
					help='location for pre-trained daytime model')
parser.add_argument('--darkening_model',type=str,default=None,help='PATH to the darkening model checkpoint')
parser.add_argument('--epochs', default=90, type=int,
					help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')                    
parser.add_argument('--lr', default=1e-3, type=float, help='optimizer lr')
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--scheduler', default='step', type=str, help='multistep or linear or cosine')
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--use_BYOL', action='store_true', help='use BYOL loss')
parser.add_argument('--BYOL_weight', default=0.1, type=float, help='BYOL loss weight')
parser.add_argument('--noise_intensity', type=float, default=0.025, help='add noise to input')
parser.add_argument('--noise_mode', type=str, default='pixel_patch', help='add noise to input')
parser.add_argument('--decreasing_lr',default='30,60', type=str, help='decreasing lr at which epoch')
parser.add_argument('--exp_range', type=str, default='0-0.2')

args = parser.parse_args()
writer = SummaryWriter(os.path.join('checkpoints', args.experiment,'tensorboard'))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
save_dir = os.path.join('checkpoints', args.experiment)
if os.path.exists(save_dir) is not True:
	os.system("mkdir -p {}".format(save_dir))
log = logger(path=save_dir)
log.info(str(args))


def train(model,train_loader,optimizer,scheduler,
				epoch,log,use_BYOL=False, darkening_model=None, exp_range=None):
	model.train()
	losses_BYOL,losses_day,losses_night,losses_total = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
	criterion = torch.nn.CrossEntropyLoss()
	st = time.time()

	for i, (images, labels) in enumerate(tqdm(train_loader)):
		images = images.cuda()
		labels = labels.cuda()
		d = images.shape
		images = images.view(d[0]*2, d[2], d[3], d[4])
							
		day_images = images[0::2]
		night_images = images[1::2]

		if darkening_model or args.gamma or args.brightness:
			with torch.no_grad():
				if darkening_model:
					exp = torch.rand(d[0]) * (exp_range[1] - exp_range[0]) + exp_range[0]
					exp = exp.cuda().view(-1,1,1,1).expand(d[0], 1, d[3], d[4])
					downsample = 16
					if 'patch' in args.noise_mode:
						noise = torch.randn(day_images.shape[0], 1, day_images.shape[2] // downsample, day_images.shape[3] // downsample).cuda() * args.noise_intensity
						noise = TF.resize(noise, (day_images.shape[2], day_images.shape[3]), interpolation=Image.BILINEAR)
						exp = exp + noise
					if 'pixel' in args.noise_mode:
						exp = exp + torch.randn_like(exp).cuda() * args.noise_intensity
					night_images, _ = darkening_model(night_images, exp)
					
				mean = (0.485, 0.456, 0.406)
				std = (0.229, 0.224, 0.225)

				jitter = 0.3
				bf = random.uniform(1-jitter,1+jitter)
				cf = random.uniform(1-jitter,1+jitter)
				sf = random.uniform(1-jitter,1+jitter)
				hf = random.uniform(-jitter,jitter)
				night_images = TF.adjust_brightness(night_images, bf)
				night_images = TF.adjust_contrast(night_images, cf)
				night_images = TF.adjust_saturation(night_images, sf)
				night_images = TF.adjust_hue(night_images, hf)

				bf = random.uniform(1-jitter,1+jitter)
				cf = random.uniform(1-jitter,1+jitter)
				sf = random.uniform(1-jitter,1+jitter)
				hf = random.uniform(-jitter,jitter)
				day_images = TF.adjust_brightness(day_images, bf)
				day_images = TF.adjust_contrast(day_images, cf)
				day_images = TF.adjust_saturation(day_images, sf)
				day_images = TF.adjust_hue(day_images, hf)

				day_images = TF.normalize(day_images, mean=mean, std=std)
				night_images = TF.normalize(night_images, mean=mean, std=std)

		outputs_day, outputs_night, feat_q, feat_k, feat_q2, feat_k2 =\
								model(x=day_images, y=night_images, dual=True)

		loss = 0.

		loss_ce_night = criterion(outputs_night,labels)
		loss += loss_ce_night
		losses_night.update(float(loss_ce_night.detach().cpu()), labels.shape[0])

		loss_ce_day = criterion(outputs_day,labels)
		loss += loss_ce_day
		losses_day.update(float(loss_ce_day.detach().cpu()), labels.shape[0])
		
		if use_BYOL:
			loss_BYOL = 0
			loss_BYOL += (2 - 2*(feat_q*feat_k).sum(dim=1).mean())
			loss_BYOL += (2 - 2*(feat_q2*feat_k2).sum(dim=1).mean())
			loss_BYOL *= (0.5 * args.BYOL_weight)
			loss += loss_BYOL
			losses_BYOL.update(float(loss_BYOL.detach().cpu()), labels.shape[0])

		losses_total.update(float(loss.detach().cpu()), labels.shape[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_time = time.time() - st
	log.info(f'Epoch: {epoch}\t  Time: {train_time:.2f}')
	log.info(f'Loss_BYOL: {losses_BYOL.avg:.4f}\t Loss_Day: {losses_day.avg:.4f}\t \
				Loss_Night: {losses_night.avg:.4f} \t Loss_Total: {losses_total.avg:.4f} ')
	writer.add_scalar('loss/BYOL', losses_BYOL.avg, epoch)
	writer.add_scalar('loss/day', losses_day.avg, epoch)
	writer.add_scalar('loss/night', losses_night.avg, epoch)
	writer.add_scalar('loss/total', losses_total.avg, epoch)
	scheduler.step()

def validate(model, log, test_day_loader, test_night_loader):
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
		log.info(f"Accuracy: {top1.avg:.2f}\t Loss: {losses.avg:.4f}")

	return acc


def main():
	transforms_train = transforms.Compose([transforms.RandomResizedCrop(224,(0.5,1.0)),
								transforms.RandomHorizontalFlip(p=0.5),
								transforms.ToTensor(),
								])
	transforms_test = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
	rootA = 'data/train'
	train_loader = torch.utils.data.DataLoader(AlignedDataset(
		rootA, rootA, maxImgNum = 1e5, transform=transforms_train), batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True,num_workers=8)

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

	start_epoch = 0
	if args.checkpoint is not None:
		resnet = resnet18(num_classes=10)
		state_dict = torch.load(args.checkpoint)
		if not args.resume:
			if 'state_dict' in state_dict:
				state_dict = state_dict['state_dict']
			resnet.load_state_dict(state_dict, strict=False)
			model = ResNet_BYOL(num_classes=10, resnet=resnet)
		else:
			model = ResNet_BYOL(num_classes=10, resnet=resnet)
			model.load_state_dict(state_dict['state_dict'], strict=True)
			start_epoch = state_dict['epoch']
		model.cuda()
	else:
		assert False
		
	model.cuda()

	if args.darkening_model is not None:
		darkening_model = enhance_net_nopool_v2(scale_factor=1).cuda()
		darkening_model.load_state_dict(torch.load(args.darkening_model))
		darkening_model.eval()
		darkening_model.requires_grad = False
	else:
		darkening_model = None
	
	cudnn.benchmark = True

	if args.optimizer == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
										weight_decay=1e-5)
	elif args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
								momentum=0.9, weight_decay=1e-5)
	else:
		assert False
	
	if args.scheduler == 'linear':
		lambdalr = lambda epoch: 1 if (epoch < args.epochs // 2) else (1 - epoch / args.epochs)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdalr)
	elif args.scheduler == 'step':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, milestones=[int(i) for i in args.decreasing_lr.split(',')], gamma=0.1)
	elif args.scheduler == 'cosine':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
	else:
		assert False
	
	for _ in range(start_epoch):
		scheduler.step()

	best_acc = 0.0
	night_best_acc = 0.0
 
	acc = validate(model,log,test_day_loader,test_night_loader)

	exp_range = [float(i) for i in args.exp_range.split('-')]
	for epoch in range(start_epoch+1, args.epochs+1):
		log.info("current lr is {}".format(
			optimizer.state_dict()['param_groups'][0]['lr']))

		loss = train(model,train_loader,optimizer,scheduler,
					epoch,log,use_BYOL=args.use_BYOL,darkening_model=darkening_model,exp_range=exp_range)

		acc = validate(model,log,test_day_loader,test_night_loader)
		writer.add_scalar("Accuracy/test_day", acc[0], epoch)
		writer.add_scalar("Accuracy/test_night", acc[1], epoch)

		if(best_acc < acc[0]):
			best_acc = acc[0]
			night_best_acc = acc[1]
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optim': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'best_acc': best_acc,
				'night_best_acc': night_best_acc,
			}, filename=os.path.join(save_dir, 'model_best.pt'))
		
		if epoch % 10 == 0:
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optim': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}, filename=os.path.join(save_dir, f'model_{epoch}.pt'))
	
	log.info(f"Best accuracy: {best_acc:.2f}, corresponding night accuracy: {night_best_acc:.2f}")

if __name__ == '__main__':
	main()
