import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from . import flow_transforms
from .video_iterator import VideoIter
from .flow_iterator import FlowIter


def get_arid(data_root='./dataset/ARID',
			   clip_length=8,
			   segments=3,
			   train_interval=2,
			   val_interval=2,
			   mean=[0.485, 0.456, 0.406],
			   std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
			   night=False,
			   **kwargs):
	""" data iter for ucf-101
	"""
	logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
				clip_length, train_interval, val_interval, seed))

	normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	# train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	if not night:
		train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
						txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID_split1_train.txt'),
						sampler=train_sampler,
						force_color=True,
						video_transform=transforms.Compose([
											transforms.RandomScale(make_square=True,
																	aspect_ratio=[0.8, 1./0.8],
																	slen=[224, 288]),
																	# slen=[112, 144]), # For C3D only
											# transforms.Resize((112, 112)),
											transforms.RandomCrop((224, 224)), # insert a resize if needed
											# transforms.RandomCrop((112, 112)), # For C3D only
											transforms.RandomHorizontalFlip(),
											transforms.RandomHLS(vars=[15, 35, 25]),
											transforms.ToTensor(), 
											normalize,
										],
										aug_seed=(seed+1)),
						name='train',
						shuffle_list_seed=(seed+2),
						)

	else:
		train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
						video_prefix_night=os.path.join(data_root, 'raw', 'data_darken'),
						txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID_split1_train.txt'),
						sampler=train_sampler,
						force_color=True,
						video_transform=transforms.Compose([
											transforms.RandomScale(make_square=True,
																	aspect_ratio=[0.8, 1./0.8],
																	slen=[224, 288]),
																	# slen=[112, 144]), # For C3D only
											# transforms.Resize((112, 112)),
											transforms.RandomCrop((224, 224)), # insert a resize if needed
											# transforms.RandomCrop((112, 112)), # For C3D only
											transforms.RandomHorizontalFlip(),
											transforms.RandomHLS(vars=[15, 35, 25]),
											transforms.ToTensor(),
											normalize,
										],
										aug_seed=(seed+1)),
						name='train',
						shuffle_list_seed=(seed+2),
						)

	val_sampler = sampler.SequentialSampling(num=clip_length, interval=val_interval, fix_cursor=True, shuffle=True)
	# val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, interval=val_interval, fix_cursor=True, shuffle=True)
	val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID_split1_test.txt'),
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 # transforms.Resize((128, 128)), # For C3D Only
										 # transforms.CenterCrop((112, 112)), # For C3D Only
										 # transforms.Resize((256, 256)),
						  				 transforms.RandomScale(make_square=False,
											aspect_ratio=[1.0, 1.0],
											slen=[256, 256]),
										 transforms.CenterCrop((224, 224)),
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  )
	return (train, val)


def get_arid_flow(data_root='./dataset/ARID',
			   clip_length=8,
			   segments=3,
			   train_interval=2,
			   val_interval=2,
			   mean=[0.485, 0.456, 0.406],
			   std=[0.229, 0.224, 0.225],
			   seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
			   **kwargs):
	""" data iter for ucf-101
	"""
	logging.debug("FlowIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
				clip_length, train_interval, val_interval, seed))

	# normalize = transforms.Normalize(mean=mean, std=std)

	train_sampler = sampler.RandomSampling(num=clip_length, interval=train_interval, speed=[1.0, 1.0], seed=(seed+0))
	# train_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, interval=train_interval, fix_cursor=False, shuffle=True, seed=(seed+0))
	train = FlowIter(video_prefix=os.path.join(data_root, 'raw', 'flow'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID_split1_train.txt'),
					  sampler=train_sampler,
					  force_gray=True,
					  flow_transforms=flow_transforms.Compose([
										 flow_transforms.RandomScale(make_square=True,
																aspect_ratio=[0.8, 1./0.8],
																slen=[224, 288]),
										 flow_transforms.RandomCrop((224, 224)), # insert a resize if needed
										 flow_transforms.RandomHorizontalFlip(),
										 flow_transforms.ToTensor(),
										 # normalize,
									  ],
									  aug_seed=(seed+1)),
					  name='train',
					  shuffle_list_seed=(seed+2),
					  )

	val_sampler = sampler.SequentialSampling(num=clip_length, interval=val_interval, fix_cursor=True, shuffle=True)
	# val_sampler = sampler.SegmentalSampling(num_per_seg=clip_length, segments=segments, interval=val_interval, fix_cursor=True, shuffle=True)
	val   = FlowIter(video_prefix=os.path.join(data_root, 'raw', 'flow'),
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'ARID_split1_test.txt'),
					  sampler=val_sampler,
					  force_gray=True,
					  flow_transforms=flow_transforms.Compose([
										 # flow_transforms.Resize((256, 256)),
						  				 transforms.RandomScale(make_square=False,
											aspect_ratio=[1.0, 1.0],
											slen=[256, 256]),
										 flow_transforms.CenterCrop((224, 224)), 
										 flow_transforms.ToTensor(),
										 # normalize,
									  ]),
					  name='test',
					  )

	return (train, val)


def creat(name, batch_size, use_flow, num_workers=16, **kwargs):

	kwargs["data_root"] = "./dataset/" + name

	if name.upper() == 'ARID' or name == 'NormalLight' and not use_flow:
		train, val = get_arid(**kwargs)
	elif name.upper() == 'ARID' or name == 'NormalLight' and use_flow:
		train, val = get_arid_flow(**kwargs)
	else:
		assert NotImplementedError("iter {} not found".format(name))


	train_loader = torch.utils.data.DataLoader(train,
		batch_size=batch_size, shuffle=True,
		num_workers=num_workers, pin_memory=False)

	val_loader = torch.utils.data.DataLoader(val,
		batch_size=2*torch.cuda.device_count(), shuffle=False,
		num_workers=num_workers, pin_memory=False)

	return (train_loader, val_loader)
