import argparse
import os
import shutil
import time
import math
import pickle

import numpy as np

import torch
import torch.optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.models as models

from torchsummary import summary

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.layers.loss import ContrastiveLoss, TripletLoss
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime
import torchvision.transforms.functional as TF
from PIL import Image

from cirtorch.utils.zero_model import enhance_net_nopool_m4 as enhance_net_nopool

training_dataset_names = ['retrieval-SfM-120k']
test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', '247tokyo1k']
test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

model_names = sorted([name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])] + ['w_resnet101','w_vgg16'])

pool_names = ['mac', 'spoc', 'gem', 'gemmp']
loss_names = ['contrastive', 'triplet']
optimizer_names = ['sgd', 'adam']

class logger(object):
	def __init__(self, path):
		self.path = path

	def info(self, msg):
		print(msg)
		with open(os.path.join(self.path, "log.txt"), 'a') as f:
			f.write(msg + "\n")

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

# export directory, training and val datasets, test datasets
parser.add_argument('directory', metavar='EXPORT_DIR',
					help='destination where trained network should be saved')
parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='retrieval-SfM-120k', choices=training_dataset_names,
					help='training dataset: ' +
						' | '.join(training_dataset_names) +
						' (default: retrieval-SfM-120k)')
parser.add_argument('--no-val', dest='val', action='store_false',
					help='do not run validation')
parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='247tokyo1k',
					help='comma separated list of test datasets: ' +
						' | '.join(test_datasets_names) +
						' (default: roxford5k,rparis6k)')
parser.add_argument('--test-whiten', metavar='DATASET', default='', choices=test_whiten_names,
					help='dataset used to learn whitening for testing: ' +
						' | '.join(test_whiten_names) +
						' (default: None)')
parser.add_argument('--test-freq', default=1, type=int, metavar='N',
					help='run test evaluation every N epochs (default: 1)')

# network architecture and initialization options
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet101)')
parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
					help='pooling options: ' +
						' | '.join(pool_names) +
						' (default: gem)')
parser.add_argument('--local-whitening', '-lw', dest='local_whitening', action='store_true',
					help='train model with learnable local whitening (linear layer) before the pooling')
parser.add_argument('--regional', '-r', dest='regional', action='store_true',
					help='train model with regional pooling using fixed grid')
parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
					help='train model with learnable whitening (linear layer) after the pooling')
parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
					help='initialize model with random weights (default: pretrained on imagenet)')
parser.add_argument('--loss', '-l', metavar='LOSS', default='contrastive',
					choices=loss_names,
					help='training loss options: ' +
						' | '.join(loss_names) +
						' (default: contrastive)')
parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
					help='loss margin: (default: 0.7)')
parser.add_argument('--mean', default='0.485,0.456,0.406',
					type=lambda s: [float(item) for item in s.split(',')],
					help='dataset mean: (default: \'0.485,0.456,0.406\')')
parser.add_argument('--std', default='0.229,0.224,0.225',
					type=lambda s: [float(item) for item in s.split(',')],
					help='dataset std: (default: \'0.229, 0.224, 0.225\')')

# train/val options specific for image retrieval learning
parser.add_argument('--image-size', default=1024, type=int, metavar='N',
					help='maximum size of longer image side used for training (default: 1024)')
parser.add_argument('--neg-num', '-nn', default=5, type=int, metavar='N',
					help='number of negative image per train/val tuple (default: 5)')
parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
					help='number of queries randomly drawn per one train epoch (default: 2000)')
parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
					help='size of the pool for hard negative mining (default: 20000)')

# standard train/val options
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
					help='gpu id used for training (default: 0)')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
					help='number of total epochs to run (default: 5)')
parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N',
					help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
parser.add_argument('--update-every', '-u', default=1, type=int, metavar='N',
					help='update model weights every N batches, used to handle really large batches, ' +
						'batch_size effectively becomes update_every x batch_size (default: 1)')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
					choices=optimizer_names,
					help='optimizer options: ' +
						' | '.join(optimizer_names) +
						' (default: adam)')
parser.add_argument('--lr', '--learning-rate', default=5e-7, type=float,
					metavar='LR', help='initial learning rate (default: 5e-7)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
					metavar='W', help='weight decay (default: 1e-6)')
parser.add_argument('--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/mnt/netdisk/luord/CVPR23/retrieval-modify/checkpoint/baseline-30-baseline-inpaper/model_epoch30.pth.tar', type=str, metavar='FILENAME',
					help='adapt model from which checkpoint')
parser.add_argument('--darkening_model', default='/mnt/netdisk/luord/CVPR23/Zero-DCE-modify/checkpoints_retrieval/scale0.95PixelReverseEst-w5/Iter15000.pth', type=str, metavar='FILENAME',
					help='path of darkening model')
parser.add_argument('--noise_mode', type=str, default='pixel_patch')
parser.add_argument('--noise_intensity', type=float, default=0.025)

min_loss = float('inf')

def main():
	global args, min_loss, log
	args = parser.parse_args()

	print(args)

	# manually check if there are unknown test datasets
	for dataset in args.test_datasets.split(','):
		if dataset not in test_datasets_names:
			raise ValueError('Unsupported or unknown test dataset: {}!'.format(dataset))

	# check if test dataset are downloaded
	# and download if they are not
	download_train(get_data_root())
	download_test(get_data_root())

	# create export dir if it doesnt exist
	# directory = "{}".format(args.training_dataset)
	# directory += "_{}".format(args.arch)
	# directory += "_{}".format(args.pool)
	# if args.local_whitening:
	#     directory += "_lwhiten"
	# if args.regional:
	#     directory += "_r"
	# if args.whitening:
	#     directory += "_whiten"
	# if not args.pretrained:
	#     directory += "_notpretrained"
	# directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
	# directory += "_{}_lr{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.weight_decay)
	# directory += "_nnum{}_qsize{}_psize{}".format(args.neg_num, args.query_size, args.pool_size)
	# directory += "_bsize{}_uevery{}_imsize{}".format(args.batch_size, args.update_every, args.image_size)

	# args.directory = os.path.join(args.directory, directory)
	# print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
	args.directory = os.path.join('checkpoints', args.directory)
	if not os.path.exists(args.directory):
		os.makedirs(args.directory)

	log = logger(args.directory)
	log.info(str(args))

	# set cuda visible device
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

	# set random seeds
	# TODO: maybe pass as argument in future implementation?
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)
	np.random.seed(0)

	# initialize model
	if args.pretrained:
		print(">> Using pre-trained model '{}'".format(args.arch))
	else:
		print(">> Using model from scratch (random weights) '{}'".format(args.arch))
	model_params = {}
	model_params['architecture'] = args.arch
	model_params['pooling'] = args.pool
	model_params['local_whitening'] = args.local_whitening
	model_params['regional'] = args.regional
	model_params['whitening'] = args.whitening
	model_params['mean'] = args.mean
	model_params['std'] = args.std
	model_params['pretrained'] = args.pretrained
	model = init_network(model_params)

	# move network to gpu
	model.cuda()

	# print(model)
	# summary(model, (3,224,224))

	# define loss function (criterion) and optimizer
	if args.loss == 'contrastive':
		criterion = ContrastiveLoss(margin=args.loss_margin).cuda()
	elif args.loss == 'triplet':
		criterion = TripletLoss(margin=args.loss_margin).cuda()
	else:
		raise(RuntimeError("Loss {} not available!".format(args.loss)))

	# parameters split into features, pool, whitening
	# IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
	parameters = []
	# add feature parameters
	parameters.append({'params': model.features.parameters()})
	# add local whitening if exists
	if model.lwhiten is not None:
		parameters.append({'params': model.lwhiten.parameters()})
	# add pooling parameters (or regional whitening which is part of the pooling layer!)
	if not args.regional:
		# global, only pooling parameter p weight decay should be 0
		if args.pool == 'gem':
			parameters.append({'params': model.pool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
		elif args.pool == 'gemmp':
			parameters.append({'params': model.pool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
	else:
		# regional, pooling parameter p weight decay should be 0,
		# and we want to add regional whitening if it is there
		if args.pool == 'gem':
			parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*10, 'weight_decay': 0})
		elif args.pool == 'gemmp':
			parameters.append({'params': model.pool.rpool.parameters(), 'lr': args.lr*100, 'weight_decay': 0})
		if model.pool.whiten is not None:
			parameters.append({'params': model.pool.whiten.parameters()})
	# add final whitening if exists
	if model.whiten is not None:
		parameters.append({'params': model.whiten.parameters()})

	# define optimizer
	if args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'adam':
		optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

	# define learning rate decay schedule
	# TODO: maybe pass as argument in future implementation?
	exp_decay = math.exp(-0.01)
	# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

	# optionally resume from a checkpoint
	start_epoch = 1
	
	checkpoint = torch.load(args.resume)
	model.load_state_dict(checkpoint['state_dict'])
	log.info(f'loaded checkpoint from {args.resume}')

	# Data loading code
	normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
	transform = transforms.Compose([
		transforms.ToTensor(),
		normalize,
	])
	train_dataset = TuplesDataset(
		name=args.training_dataset,
		mode='train',
		imsize=args.image_size,
		nnum=args.neg_num,
		qsize=args.query_size,
		poolsize=args.pool_size,
		transform=transform,
		night=True,
	)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True, sampler=None,
		drop_last=True, collate_fn=collate_tuples
	)
	if args.val:
		val_dataset = TuplesDataset(
			name=args.training_dataset,
			mode='val',
			imsize=args.image_size,
			nnum=args.neg_num,
			qsize=float('Inf'),
			poolsize=float('Inf'),
			transform=transform
		)
		val_loader = torch.utils.data.DataLoader(
			val_dataset, batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True,
			drop_last=True, collate_fn=collate_tuples
		)

	# uncomment to evaluate the network before starting
	# test(args.test_datasets, model)

	if args.darkening_model is not None:
		darkening_model = enhance_net_nopool(scale_factor=1)
		darkening_model.load_state_dict(torch.load(args.darkening_model))
		darkening_model.eval().cuda()
		darkening_model.requires_grad = False
	else:
		darkening_model = None

	for epoch in range(start_epoch, args.epochs+1):

		# set manual seeds per epoch
		np.random.seed(epoch)
		torch.manual_seed(epoch)
		torch.cuda.manual_seed_all(epoch)

		# # debug printing to check if everything ok
		# lr_feat = optimizer.param_groups[0]['lr']
		# lr_pool = optimizer.param_groups[1]['lr']
		# print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))

		# train for one epoch on train set
		loss = train(train_loader, model, criterion, optimizer, epoch, darkening_model)
		log.info(f'Train epoch: {epoch} loss: {loss}')

		# evaluate on validation set
		# if args.val:
		#     with torch.no_grad():
		#         loss = validate(val_loader, model, criterion, epoch)
		#         log.info(f'Epoch: {epoch} - Validation loss: {loss:.4f}')

		# evaluate on test datasets every test_freq epochs
		if (epoch + 1) % args.test_freq == 0:
			with torch.no_grad():
				map = test(args.test_datasets, model)
				log.info(f'Epoch: {epoch} - MAP: {map:.3f}')

		# remember best loss and save checkpoint
		is_best = loss < min_loss
		min_loss = min(loss, min_loss)

		save_checkpoint({
			'epoch': epoch,
			'meta': model.meta,
			'state_dict': model.state_dict(),
			'min_loss': min_loss,
			'optimizer' : optimizer.state_dict(),
		}, is_best, args.directory)
		
		# scheduler.step()

def train(train_loader, model, criterion, optimizer, epoch, darkening_model=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	# create tuples for training
	avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)

	# switch to train mode
	model.train()
	model.apply(set_batchnorm_eval)

	# zero out gradients
	optimizer.zero_grad()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		nq = len(input) # number of training tuples
		ni = len(input[0]) # number of images per tuple

		for q in range(nq):
			if darkening_model is not None:
				img = input[q][1]
				h, w = img.shape[-2:]
				# denormalize
				img = img.cuda()
				img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda() + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
				# img = torch.clamp(img, 0, 0.8)
				# img = img.unsqueeze(0)
				exp = np.random.uniform(0, 0.2)
				exp = torch.tensor(exp).cuda().expand(1, 1, h, w)
				if 'patch' in args.noise_mode:
					noise = torch.randn(1, 1, h // 32, w // 32).cuda() * args.noise_intensity
					noise = TF.resize(noise, (h, w), interpolation=Image.BILINEAR)
					exp = exp + noise
				if 'pixel' in args.noise_mode:
					exp = exp + torch.randn_like(exp).cuda() * args.noise_intensity
				img, _  = darkening_model(img, exp)
				# normalize
				img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
				# img = TF.resize(img, (h, w), interpolation=Image.BILINEAR)
				input[q][1] = img
				
			output = torch.zeros(model.meta['outputdim'], ni).cuda()
			for imi in range(ni):

				# compute output vector for image imi
				output[:, imi] = model(input[q][imi].cuda()).squeeze()

			# reducing memory consumption:
			# compute loss for this query tuple only
			# then, do backward pass for one tuple only
			# each backward pass gradients will be accumulated
			# the optimization step is performed for the full batch later
			loss = criterion(output, target[q].cuda())
			losses.update(loss.item())
			loss.backward()

		if (i + 1) % args.update_every == 0:
			# do one step for multiple batches
			# accumulated gradients are used
			optimizer.step()
			# zero out gradients so we can
			# accumulate new ones over batches
			optimizer.zero_grad()
			# print('>> Train: [{0}][{1}/{2}]\t'
			#       'Weight update performed'.format(
			#        epoch+1, i+1, len(train_loader)))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(train_loader):
			log.info('>> Train: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
				   epoch, i+1, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses))

	return losses.avg


def validate(val_loader, model, criterion, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()

	# create tuples for validation
	avg_neg_distance = val_loader.dataset.create_epoch_tuples(model)

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):

		nq = len(input) # number of training tuples
		ni = len(input[0]) # number of images per tuple
		output = torch.zeros(model.meta['outputdim'], nq*ni).cuda()

		for q in range(nq):

			for imi in range(ni):
				# compute output vector for image imi of query q
				output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()

		# no need to reduce memory consumption (no backward pass):
		# compute loss for the full batch
		loss = criterion(output, torch.cat(target).cuda())

		# record loss
		losses.update(loss.item()/nq, nq)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if (i+1) % args.print_freq == 0 or i == 0 or (i+1) == len(val_loader):
			print('>> Val: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
				   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses))

	return losses.avg

def test(datasets, net):

	print('>> Evaluating network on test datasets...')

	# for testing we use image size of max 1024
	image_size = 1024

	# moving network to gpu and eval mode
	net.cuda()
	net.eval()
	# set up the transform
	normalize = transforms.Normalize(
		mean=net.meta['mean'],
		std=net.meta['std']
	)
	transform = transforms.Compose([
		transforms.ToTensor(),
		normalize
	])

	# compute whitening
	if args.test_whiten:
		start = time.time()

		print('>> {}: Learning whitening...'.format(args.test_whiten))

		# loading db
		db_root = os.path.join(get_data_root(), 'train', args.test_whiten)
		ims_root = os.path.join(db_root, 'ims')
		db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.test_whiten))
		with open(db_fn, 'rb') as f:
			db = pickle.load(f)
		images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

		# extract whitening vectors
		print('>> {}: Extracting...'.format(args.test_whiten))
		wvecs = extract_vectors(net, images, image_size, transform, print_freq=args.print_freq*100)  # implemented with torch.no_grad

		# learning whitening
		print('>> {}: Learning...'.format(args.test_whiten))
		wvecs = wvecs.numpy()
		m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
		Lw = {'m': m, 'P': P}

		print('>> {}: elapsed time: {}'.format(args.test_whiten, htime(time.time()-start)))
	else:
		Lw = None

	# evaluate on test datasets
	datasets = args.test_datasets.split(',')
	for dataset in datasets:
		start = time.time()

		print('>> {}: Extracting...'.format(dataset))

		# prepare config structure for the test dataset
		cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
		images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
		qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
		bbxs = [(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

		# extract database and query vectors
		print('>> {}: database images...'.format(dataset))
		vecs = extract_vectors(net, images, image_size, transform, print_freq=args.print_freq*100)  # implemented with torch.no_grad
		print('>> {}: query images...'.format(dataset))
		qvecs = extract_vectors(net, qimages, image_size, transform, bbxs, print_freq=args.print_freq*100)  # implemented with torch.no_grad

		print('>> {}: Evaluating...'.format(dataset))

		# convert to numpy
		vecs = vecs.numpy()
		qvecs = qvecs.numpy()

		# search, rank, and print
		scores = np.dot(vecs.T, qvecs)
		ranks = np.argsort(-scores, axis=0)
		map = compute_map_and_print(dataset, ranks, cfg['gnd'])

		if Lw is not None:
			# whiten the vectors
			vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
			qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

			# search, rank, and print
			scores = np.dot(vecs_lw.T, qvecs_lw)
			ranks = np.argsort(-scores, axis=0)
			map = compute_map_and_print(dataset + ' + whiten', ranks, cfg['gnd'])

		print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))
	
		return map


def save_checkpoint(state, is_best, directory):
	filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
	torch.save(state, filename)
	if is_best:
		filename_best = os.path.join(directory, 'model_best.pth.tar')
		shutil.copyfile(filename, filename_best)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def set_batchnorm_eval(m):
	classname = m.__class__.__name__
	if classname.find('BatchNorm') != -1:
		# freeze running mean and std:
		# we do training one image at a time
		# so the statistics would not be per batch
		# hence we choose freezing (ie using imagenet statistics)
		m.eval()
		# # freeze parameters:
		# # in fact no need to freeze scale and bias
		# # they can be learned
		# # that is why next two lines are commented
		# for p in m.parameters():
			# p.requires_grad = False


if __name__ == '__main__':
	main()
