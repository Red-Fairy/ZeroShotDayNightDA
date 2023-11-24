import sys
sys.path.append("..")

import os
import time
import json
import logging
import argparse
import random

import torch
import torch.backends.cudnn as cudnn

import dataset
from train.model import static_model
from train import metric
from data import video_sampler as sampler
from data import video_transforms as transforms
from data.video_iterator import VideoIter
from network.symbol_builder import get_symbol

GLOBAL_SEED = 0
torch.manual_seed(GLOBAL_SEED) # cpu
torch.cuda.manual_seed(GLOBAL_SEED) #gpu
random.seed(GLOBAL_SEED) #random and transforms

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Parser (Evaluation) default UCF101")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
					help="print all setting for debugging.")
# io
parser.add_argument('--dataset', default='NormalLight', help="path to dataset")
parser.add_argument('--clip-length', default=16,
					help="define the length of each input sample.")    
parser.add_argument('--frame-interval', type=int, default=2,
					help="define the sampling interval between frames.")    
parser.add_argument('--task-name', type=str, default='../exps/models/archive/ARID_v1-master',
					help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./",
					help="set logging file.")
parser.add_argument('--log-file', type=str, default="./eval-hmdb51-0324.log",
					help="set logging file.")
# device
parser.add_argument('--gpus', type=str, default="0",
					help="define gpu id")
# algorithm
parser.add_argument('--network', type=str, default='RES_I3D',
					help="choose the base network")
# evaluation
parser.add_argument('--load-epoch', type=int, default=47,
					help="resume trained model")
parser.add_argument('--batch-size', type=int, default=4,
					help="batch size")

#other changes
parser.add_argument('--workers', type=int, default=4, help='num_workers during evaluation data loading')
parser.add_argument('--test-rounds', type=int, default=100, help='number of testing rounds')
parser.add_argument('--is-dark', type=bool, default=False)


def autofill(args):
	# customized
	if not args.task_name:
		args.task_name = os.path.basename(os.getcwd())
	# fixed
	args.model_prefix = os.path.join(args.model_dir, args.task_name)
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
	sym_net, input_config = get_symbol(name=args.network, is_dark=dark, **dataset_cfg)
	
	# network
	if torch.cuda.is_available():
		cudnn.benchmark = True
		sym_net = torch.nn.DataParallel(sym_net).cuda()
		criterion = torch.nn.CrossEntropyLoss().cuda()
	else:
		sym_net = torch.nn.DataParallel(sym_net)
		criterion = torch.nn.CrossEntropyLoss()
	net = static_model(net=sym_net,
					   criterion=criterion,
					   model_prefix=args.model_prefix)
	net.load_checkpoint(args.load_epoch)
	
	# data iterator:
	data_root = "../dataset/{}".format(args.dataset)
	video_location = os.path.join(data_root, 'SACC-17000-videos')
	# StableLLVE-videos, RUAS-videos, SACC-17000-videos

	normalize = transforms.Normalize(mean=input_config['mean'], std=input_config['std'])
	val_sampler = sampler.RandomSampling(num=args.clip_length,
										 interval=args.frame_interval,
										 speed=[1.0, 1.0], seed=GLOBAL_SEED)
	val_loader = VideoIter(video_prefix=video_location,
					  txt_list="./ARID_split1_test-custom.txt", 
					  sampler=val_sampler,
					  force_color=True,
					  video_transform=transforms.Compose([
										 transforms.Resize((256,256)),
										 transforms.RandomCrop((224,224)),
										 # transforms.Resize((128, 128)), # For C3D Only
										 # transforms.RandomCrop((112, 112)), # For C3D Only
										 # transforms.CenterCrop((224, 224)), # we did not use center crop in our paper
										 # transforms.RandomHorizontalFlip(), # we did not use mirror in our paper
										 transforms.ToTensor(),
										 normalize,
									  ]),
					  name='test',
					  return_item_subpath=True,
					  )
					  
	eval_iter = torch.utils.data.DataLoader(val_loader,
					  batch_size=args.batch_size,
					  shuffle=True,
					  num_workers=args.workers,
					  pin_memory=True)

	# eval metrics
	metrics = metric.MetricList(metric.Loss(name="loss-ce"),
								metric.Accuracy(topk=1, name="top1"),
								metric.Accuracy(topk=5, name="top5"))
	metrics.reset()

	# main loop
	net.net.eval()
	avg_score = {}
	sum_batch_elapse = 0.
	sum_batch_inst = 0
	duplication = 1
	softmax = torch.nn.Softmax(dim=1)

	total_round = args.test_rounds # change this part accordingly if you do not want an inf loop
	for i_round in range(total_round):
		i_batch = 0
		logging.info("round #{}/{}".format(i_round, total_round))
		for data, target, video_subpath in eval_iter:
			batch_start_time = time.time()

			outputs, losses = net.forward(data, target)

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

				# pred_i = output_i.data.argmax()

				# if pred_i == target_i:
				# 	print(video_subpath_i, pred_i, target_i)

			# show progress
			if (i_batch % 100) == 0:
				metrics.reset()
				for _, video_info in avg_score.items():
					target, loss, pred, _ = video_info
					metrics.update([pred], target, [loss])
				name_value = metrics.get_name_value()
				print(video_location)
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
