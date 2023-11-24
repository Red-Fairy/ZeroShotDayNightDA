import sys
sys.path.append("..")

import os
import time
import json
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms
from data import flow_transforms
from data.dual_iterator import DualIter
from network.symbol_builder import get_symbol

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation) default UCF101")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
					help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='ARID', help="path to dataset")
parser.add_argument('--clip-length-rgb', type=int, default=16)
parser.add_argument('--clip-length-flow', type=int, default=16)
parser.add_argument('--frame-interval', type=int, default=2)    
parser.add_argument('--task-name-rgb', type=str, default='../exps/models/archive/arid-i3d-0429/ARID_PyTorch')    
parser.add_argument('--task-name-flow', type=str, default='../exps/models/archive/ARID_PyTorch')
parser.add_argument('--model-dir', type=str, default="./")
parser.add_argument('--log-file', type=str, default="./eval-hmdb51-0324.log")
# device
parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6,7")
# algorithm
parser.add_argument('--network-rgb', type=str, default='i3d')
parser.add_argument('--network-flow', type=str, default='i3d_flow')
# evaluation
parser.add_argument('--load-epoch', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=8)

#other changes
parser.add_argument('--list-file', type=str, default='ARID_split1_test.txt')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--test-rounds', type=int, default=20)
parser.add_argument('--is-dark', type=bool, default=True)
parser.add_argument('--use-flow', type=bool, default=True)
parser.add_argument('--weight-rgb', type=float, default=1.0)
parser.add_argument('--weight-flow', type=float, default=1.0)
parser.add_argument('--use-segments', default=True, type=bool)
parser.add_argument('--segments', default=3, type=int)


def autofill(args):
	# customized
	args.model_prefix_rgb = os.path.join(args.model_dir, args.task_name_rgb)
	args.model_prefix_flow = os.path.join(args.model_dir, args.task_name_flow)
	return args

def set_logger(log_file='', debug_mode=False):
	if log_file:
		if not os.path.exists("./"+os.path.dirname(log_file)):
			os.makedirs("./"+os.path.dirname(log_file))
		handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
	else:
		handlers = [logging.StreamHandler()]

	""" add '%(filename)s' to format show source file """
	logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
				format='%(asctime)s %(levelname)s: %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S',
				handlers = handlers)


if __name__ == '__main__':

	# set args
	args = parser.parse_args()
	args = autofill(args)

	set_logger(log_file=args.log_file, debug_mode=args.debug_mode)
	logging.info("Start evaluation with args:\n" +
				 json.dumps(vars(args), indent=4, sort_keys=True))

	# set device states
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # before using torch
	assert torch.cuda.is_available(), "CUDA is not available"

	# load dataset related configuration
	dataset_cfg = dataset.get_config(name=args.dataset)

	# creat model
	dark = args.is_dark
	sym_net_rgb, input_config = get_symbol(name=args.network_rgb, is_dark=dark, **dataset_cfg)
	sym_net_flow, input_config = get_symbol(name=args.network_flow, is_dark=dark, **dataset_cfg)
	
	# network
	if torch.cuda.is_available():
		cudnn.benchmark = True
		sym_net_rgb = torch.nn.DataParallel(sym_net_rgb).cuda()
		sym_net_flow = torch.nn.DataParallel(sym_net_flow).cuda()
		criterion = torch.nn.CrossEntropyLoss().cuda()
	else:
		sym_net_rgb = torch.nn.DataParallel(sym_net_rgb)
		sym_net_flow = torch.nn.DataParallel(sym_net_flow)
		criterion = torch.nn.CrossEntropyLoss()
	net_rgb = static_model(net=sym_net_rgb, criterion=criterion, model_prefix=args.model_prefix_rgb)
	net_flow = static_model(net=sym_net_flow, criterion=criterion, model_prefix=args.model_prefix_flow)
	net_rgb.load_checkpoint(epoch=args.load_epoch)
	net_flow.load_checkpoint(epoch=args.load_epoch)
	
	# data iterator:
	data_root = "../dataset/{}".format(args.dataset)
	
	video_location = os.path.join(data_root, 'raw', 'data')
	flow_location = os.path.join(data_root, 'raw', 'flow')
	
	normalize = video_transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
	if not args.use_segments:
		val_video_sampler = sampler.RandomSampling(num=args.clip_length_rgb, interval=args.frame_interval, speed=[1.0, 1.0], seed=1)
		val_flow_sampler = sampler.RandomSampling(num=args.clip_length_flow, interval=args.frame_interval, speed=[1.0, 1.0], seed=1)
	else:
		val_video_sampler = sampler.SegmentalSampling(num_per_seg=args.clip_length_rgb, segments=args.segments, interval=args.frame_interval, fix_cursor=True, shuffle=True, seed=1)
		val_flow_sampler = sampler.SegmentalSampling(num_per_seg=args.clip_length_flow, segments=args.segments, interval=args.frame_interval, fix_cursor=True, shuffle=True, seed=1)
	val_loader = DualIter(video_prefix=video_location,
					  flow_prefix=flow_location,
					  txt_list=os.path.join(data_root, 'raw', 'list_cvt', args.list_file), 
					  video_sampler=val_video_sampler,
					  flow_sampler=val_flow_sampler,
					  force_color=True,
					  force_gray=True,
					  flow_transforms=flow_transforms.Compose([
										 flow_transforms.Resize((256,256)),
										 flow_transforms.RandomCrop((224,224)),
										 # flow_transforms.CenterCrop((224, 224)), # we did not use center crop in our paper
										 # flow_transforms.RandomHorizontalFlip(), # we did not use mirror in our paper
										 flow_transforms.ToTensor(),
									  ]),

					  video_transforms=video_transforms.Compose([
										 video_transforms.Resize((256,256)),
										 video_transforms.RandomCrop((224,224)),
										 # video_transforms.CenterCrop((224, 224)), # we did not use center crop in our paper
										 # video_transforms.RandomHorizontalFlip(), # we did not use mirror in our paper
										 video_transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )
					  
	eval_iter = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=True, 
											num_workers=args.workers, pin_memory=True)

	# eval metrics
	metrics = metric.MetricList(metric.Loss(name="loss-ce"), metric.Accuracy(topk=1, name="top1"), 
								metric.Accuracy(topk=5, name="top5"))
	metrics.reset()

	# main loop
	net_rgb.net.eval()
	net_flow.net.eval()
	avg_score = {}
	sum_batch_elapse = 0.
	sum_batch_inst = 0
	duplication = 1
	softmax = torch.nn.Softmax(dim=1)

	total_round = args.test_rounds # change this part accordingly if you do not want an inf loop
	weight_rgb = args.weight_rgb
	weight_flow = args.weight_flow
	for i_round in range(total_round):
		i_batch = 0
		logging.info("round #{}/{}".format(i_round, total_round))
		for rgb, flow, target, video_subpath in eval_iter:
			batch_start_time = time.time()

			if 'VGG' in net_rgb.net.module.__class__.__name__:
				rgb = rgb.reshape(-1, rgb.shape[1]*rgb.shape[2], rgb.shape[3], rgb.shape[4])
			if 'VGG' in net_flow.net.module.__class__.__name__:
				flow = flow.reshape(-1, flow.shape[1]*flow.shape[2], flow.shape[3], flow.shape[4])

			outputs_rgb, losses = net_rgb.forward(rgb, target)
			outputs_flow, losses = net_flow.forward(flow, target)
			outputs = [output_rgb + output_flow for output_rgb, output_flow in zip(outputs_rgb, outputs_flow)]

			sum_batch_elapse += time.time() - batch_start_time
			sum_batch_inst += 1

			# recording
			output = softmax(outputs[0]).data.cpu()
			target = target.cpu()
			losses = losses[0].data.cpu()
			# logging.info("output is {}, target is {}".format(output, target))
			for i_item in range(0, output.shape[0]):
				output_i = output[i_item,:].view(1, -1)
				target_i = torch.LongTensor([target[i_item]])
				loss_i = losses
				video_subpath_i = video_subpath[i_item]
				if video_subpath_i in avg_score:
					avg_score[video_subpath_i][2] += output_i
					avg_score[video_subpath_i][3] += 1
					duplication = 0.92 * duplication + 0.08 * avg_score[video_subpath_i][3]
				else:
					avg_score[video_subpath_i] = [torch.LongTensor(target_i.numpy().copy()), 
												  torch.FloatTensor(loss_i.numpy().copy()), 
												  torch.FloatTensor(output_i.numpy().copy()),
												  1] # the last one is counter

			# show progress
			if (i_batch % 100) == 0:
				metrics.reset()
				for _, video_info in avg_score.items():
					target, loss, pred, _ = video_info
					metrics.update([pred], target, [loss])
				name_value = metrics.get_name_value()
				logging.info("{:.1f}%, {:.1f} \t| Batch [0,{}]    \tAvg: {} = {:.5f}, {} = {:.5f}, {} = {:.5f}".format(
							float(100*i_batch) / eval_iter.__len__(), \
							duplication, \
							i_batch, \
							name_value[0][0][0], name_value[0][0][1], \
							name_value[1][0][0], name_value[1][0][1], \
							name_value[2][0][0], name_value[2][0][1]))
			i_batch += 1


	# finished
	logging.info("Evaluation Finished!")

	metrics.reset()
	for _, video_info in avg_score.items():
		target, loss, pred, _ = video_info
		metrics.update([pred], target, [loss])

	logging.info("Total time cost: {:.1f} sec".format(sum_batch_elapse))
	logging.info("Speed: {:.4f} samples/sec".format(
			args.batch_size * sum_batch_inst / sum_batch_elapse ))
	logging.info("Accuracy:")
	logging.info(json.dumps(metrics.get_name_value(), indent=4, sort_keys=True))
