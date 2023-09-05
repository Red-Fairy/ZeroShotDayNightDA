# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:38:31 2019 by Attila Lengyel - attila@lengyel.nl
"""

from numpy import interp
import torch
import time
import os
import cv2
from zmq import PROTOCOL_ERROR_ZAP_BAD_REQUEST_ID

from utils.helpers import AverageMeter, ProgressMeter, visim, vislbl
from utils.get_iou import iouCalc
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import torch.distributed as dist
import torchvision.transforms.functional as TF
import torchvision
import copy
from datasets.cityscapes_ext import CityscapesExt
import random

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= nprocs
	return rt

def train_epoch(dataloader, model, criterion, optimizer, epoch, log, void=-1):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, data_time, loss_running, acc_running],
		prefix="Epoch: [{}]".format(epoch))
	
	# set model in training mode
	model.train()
	
	end = time.time()
	
	with torch.set_grad_enabled(True):
		# Iterate over data.
		for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
			data_time.update(time.time()-end)

			# input resolution
			res = inputs.shape[2]*inputs.shape[3]
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()

			# zero the parameter gradients
			optimizer.zero_grad()
	
			# forward
			outputs = model(inputs)
			preds = torch.argmax(outputs, 1)
			loss = criterion(outputs, labels)
			
			# backward
			loss.backward()
			optimizer.step()
			
			# Statistics
			bs = inputs.size(0) # current batch size
			loss = loss.item()
			loss_running.update(loss, bs)
			corrects = torch.sum(preds == labels.data)
			nvoid = int((labels==void).sum())
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			
			# Output training info
			progress.display(epoch_step)
			# Append current stats to csv
			with open('logs/log_batch.csv', 'a') as log_batch:
				log_batch.write('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(epoch,
								epoch_step, loss/bs, loss_running.avg,
								acc, acc_running.avg))
							
			batch_time.update(time.time() - end)
			end = time.time()
			
	log.info('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,loss_running.avg,acc_running.avg))
	
	return loss_running.avg, acc_running.avg

def train_epoch_BYOL_dual(dataloader, model, criterion, optimizer, epoch, log, void=-1, BYOL_weight=0.1):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	loss_day_running = AverageMeter('Loss_day', ':.4e')
	loss_night_running = AverageMeter('Loss_night', ':.4e')
	loss_BYOL_running = AverageMeter('Loss_BYOL', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	acc_night_running = AverageMeter('Acc_night', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, data_time, loss_running, loss_day_running, 
			loss_night_running, loss_BYOL_running, acc_running, acc_night_running],
		prefix="Epoch: [{}]".format(epoch))
	
	# set model in training mode
	model.train()
	
	end = time.time()
	
	with torch.set_grad_enabled(True):
		# Iterate over data.
		for epoch_step, (inputs, inputs_night, labels) in enumerate(dataloader):
			bs = inputs.size(0) # current batch size
			data_time.update(time.time()-end)

			# input resolution
			res = inputs.shape[2]*inputs.shape[3]
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()

			try:
				model._momemtum_update_key_encoder()
			except:
				model.module._momentum_update_key_encoder()

			optimizer.zero_grad()
			# forward & backward day data (pass day data to normal route)
			loss, loss_BYOL = 0., 0.
			outputs, feats_q, feats_k = model(inputs, inputs_night, dual=False) # encoder_k forwards night data
			for i in range(len(feats_q)):
				loss_BYOL += (2-2 * (feats_q[i] * feats_k[i]).sum(dim=1).mean()) * BYOL_weight
			loss_BYOL /= 4
			loss += loss_BYOL

			preds = torch.argmax(outputs, 1)
			loss_day = criterion(outputs, labels)
			loss += loss_day
			
			loss.backward()

			loss = loss.item()
			loss_running.update(loss, bs)
			loss_day_running.update(loss_day.item(), bs)
			loss_BYOL_running.update(loss_BYOL.item(), bs)

			# forward & backward night data (pass day night to normal route)
			# optimizer.zero_grad()
			loss, loss_BYOL = 0., 0.
			outputs_night, feats_q, feats_k = model(inputs_night, inputs, dual=False) 
			for i in range(len(feats_q)):
				loss_BYOL += (2-2 * (feats_q[i] * feats_k[i]).sum(dim=1).mean()) * BYOL_weight
			loss_BYOL /= 4
			loss += loss_BYOL
			
			preds_night = torch.argmax(outputs_night, 1)
			loss_night = criterion(outputs_night, labels)
			loss += loss_night

			loss.backward()
			
			loss = loss.item()
			loss_running.update(loss, bs)
			loss_night_running.update(loss_night.item(), bs)
			loss_BYOL_running.update(loss_BYOL.item(), bs)
			
			# update parameters
			optimizer.step()

			# Statistics
			nvoid = int((labels==void).sum())
			corrects = torch.sum(preds == labels.data)
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			corrects_night = torch.sum(preds_night == labels.data)
			acc_night = corrects_night.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_night_running.update(acc_night, bs)
			
			# Output training info
			progress.display(epoch_step)
			# Append current stats to csv
			with open('logs/log_batch.csv', 'a') as log_batch:
				log_batch.write('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(epoch,
								epoch_step, loss/bs, loss_running.avg, loss_day_running.avg, loss_night_running.avg, 
								loss_BYOL_running.avg, acc_night_running.avg, acc_running.avg))
							
			batch_time.update(time.time() - end)
			end = time.time()
			
	log.info('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,loss_running.avg,acc_running.avg))
	
	return loss_running.avg, loss_day_running.avg, loss_night_running.avg ,loss_BYOL_running.avg, acc_running.avg, acc_night_running.avg

def train_epoch_sim_dual(dataloader, model, criterion, optimizer, epoch, log, void=-1, sim_weight=0.1):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	loss_day_running = AverageMeter('Loss_day', ':.4e')
	loss_night_running = AverageMeter('Loss_night', ':.4e')
	loss_sim_running = AverageMeter('Loss_sim', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	acc_night_running = AverageMeter('Acc_night', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, data_time, loss_running, loss_day_running, 
			loss_night_running, loss_sim_running, acc_running, acc_night_running],
		prefix="Epoch: [{}]".format(epoch))
	
	# set model in training mode
	model.train()
	
	end = time.time()
	
	with torch.set_grad_enabled(True):
		# Iterate over data.
		for epoch_step, (inputs, inputs_night, labels) in enumerate(dataloader):
			bs = inputs.size(0) # current batch size
			data_time.update(time.time()-end)

			# input resolution
			res = inputs.shape[2]*inputs.shape[3]
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()

			optimizer.zero_grad()
			# forward & backward day data (pass day data to normal route)
			loss, loss_sim = 0., 0.
			outputs, feats_day = model(inputs, return_feats=True)
			outputs_night, feats_night = model(inputs_night, return_feats=True)
			
			for i in range(len(feats_day)):
				loss_sim += (1 - (feats_day[i] * feats_night[i]).sum(dim=1).mean()) * sim_weight
			loss_sim /= 4
   
			loss_day = criterion(outputs, labels)
			loss_night = criterion(outputs_night, labels)
			loss += loss_day + loss_night + loss_sim
			
			loss.backward()

			loss = loss.item()
			loss_running.update(loss, bs)
			loss_day_running.update(loss_day.item(), bs)
			loss_sim_running.update(loss_sim.item(), bs)
   
			preds = torch.argmax(outputs, 1)
			preds_night = torch.argmax(outputs_night, 1)
			
			# update parameters
			optimizer.step()

			# Statistics
			nvoid = int((labels==void).sum())
			corrects = torch.sum(preds == labels.data)
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			corrects_night = torch.sum(preds_night == labels.data)
			acc_night = corrects_night.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_night_running.update(acc_night, bs)
			
			# Output training info
			progress.display(epoch_step)
			# Append current stats to csv
			with open('logs/log_batch.csv', 'a') as log_batch:
				log_batch.write('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(epoch,
								epoch_step, loss/bs, loss_running.avg, loss_day_running.avg, loss_night_running.avg, 
								loss_sim_running.avg, acc_night_running.avg, acc_running.avg))
							
			batch_time.update(time.time() - end)
			end = time.time()
			
	log.info('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,loss_running.avg,acc_running.avg))
	
	return loss_running.avg, loss_day_running.avg, loss_night_running.avg ,loss_sim_running.avg, acc_running.avg, acc_night_running.avg

def train_epoch_BYOL_dual_online(dataloader, model, darkening_model, criterion, optimizer, epoch, log, void=-1, BYOL_weight=0.1):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	loss_day_running = AverageMeter('Loss_day', ':.4e')
	loss_night_running = AverageMeter('Loss_night', ':.4e')
	loss_BYOL_running = AverageMeter('Loss_BYOL', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	acc_night_running = AverageMeter('Acc_night', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, data_time, loss_running, loss_day_running, 
			loss_night_running, loss_BYOL_running, acc_running, acc_night_running],
		prefix="Epoch: [{}]".format(epoch))
	
	# set model in training mode
	model.train()
	
	end = time.time()
	
	with torch.set_grad_enabled(True):
		# Iterate over data.
		for epoch_step, (inputs, inputs_night, labels) in enumerate(dataloader):
			bs = inputs.size(0) # current batch size
			data_time.update(time.time()-end)

			# input resolution
			res = inputs.shape[2]*inputs.shape[3]
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()

			try:
				model._momemtum_update_key_encoder()
			except:
				model.module._momentum_update_key_encoder()

			with torch.no_grad():
				exp = np.random.uniform(0, 0.2)
				d = inputs.shape
				exp = torch.tensor(exp).cuda().expand(d[0], 1, d[2], d[3])
				noise = torch.randn(d[0], 1, d[2] // 64, d[3] // 64).cuda() * 0.025
				noise = TF.resize(noise, (d[2], d[3]), interpolation=Image.BILINEAR)
				exp = noise + torch.randn_like(exp).cuda() * 0.025
				# inputs_night = torch.clamp(inputs_night, 0, 0.8)
				inputs_night, _ = darkening_model(inputs_night, exp)
				inputs_night = (inputs_night * 255).byte() / 255

				# color jitter and normalize
				mean = (0.485, 0.456, 0.406)
				std = (0.229, 0.224, 0.225)

				jitter = 0.3
				bf = random.uniform(1-jitter,1+jitter)
				cf = random.uniform(1-jitter,1+jitter)
				sf = random.uniform(1-jitter,1+jitter)
				hf = random.uniform(-jitter,jitter)
				inputs_night = TF.adjust_brightness(inputs_night, bf)
				inputs_night = TF.adjust_contrast(inputs_night, cf)
				inputs_night = TF.adjust_saturation(inputs_night, sf)
				inputs_night = TF.adjust_hue(inputs_night, hf)

				bf = random.uniform(1-jitter,1+jitter)
				cf = random.uniform(1-jitter,1+jitter)
				sf = random.uniform(1-jitter,1+jitter)
				hf = random.uniform(-jitter,jitter)
				inputs = TF.adjust_brightness(inputs, bf)
				inputs = TF.adjust_contrast(inputs, cf)
				inputs = TF.adjust_saturation(inputs, sf)
				inputs = TF.adjust_hue(inputs, hf)

				inputs = TF.normalize(inputs, mean=mean, std=std)
				inputs_night = TF.normalize(inputs_night, mean=mean, std=std)
				
			optimizer.zero_grad()
			# forward & backward day data (pass day data to normal route)
			loss, loss_BYOL = 0., 0.
			outputs, feats_q, feats_k = model(inputs, inputs_night, dual=False) # encoder_k forwards night data
			for i in range(len(feats_q)):
				loss_BYOL += (2-2 * (feats_q[i] * feats_k[i]).sum(dim=1).mean()) * BYOL_weight

			preds = torch.argmax(outputs, 1)
			loss_day = criterion(outputs, labels)
			loss += loss_day

			# forward & backward night data (pass day night to normal route)
			outputs_night, feats_q, feats_k = model(inputs_night, inputs, dual=False) 
			for i in range(len(feats_q)):
				loss_BYOL += (2-2 * (feats_q[i] * feats_k[i]).sum(dim=1).mean()) * BYOL_weight
			loss_BYOL /= 8
			loss += loss_BYOL
			
			preds_night = torch.argmax(outputs_night, 1)
			loss_night = criterion(outputs_night, labels)
			loss += loss_night

			loss.backward()
			loss = loss.item()

			loss_running.update(loss, bs)
			loss_day_running.update(loss_day.item(), bs)
			loss_night_running.update(loss_night.item(), bs)
			loss_BYOL_running.update(loss_BYOL.item(), bs)
			
			# update parameters
			optimizer.step()

			# Statistics
			nvoid = int((labels==void).sum())
			corrects = torch.sum(preds == labels.data)
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			corrects_night = torch.sum(preds_night == labels.data)
			acc_night = corrects_night.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_night_running.update(acc_night, bs)
			
			# Output training info
			progress.display(epoch_step)
			# Append current stats to csv
			with open('logs/log_batch.csv', 'a') as log_batch:
				log_batch.write('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(epoch,
								epoch_step, loss/bs, loss_running.avg, loss_day_running.avg, loss_night_running.avg, 
								loss_BYOL_running.avg, acc_night_running.avg, acc_running.avg))
							
			batch_time.update(time.time() - end)
			end = time.time()
			
	log.info('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,loss_running.avg,acc_running.avg))
	
	return loss_running.avg, loss_day_running.avg, loss_night_running.avg ,loss_BYOL_running.avg, acc_running.avg, acc_night_running.avg


def evaluate(dataloader, model, criterion, epoch, classLabels, validClasses, log=None, void=-1, maskColors=None, mean=None, std=None):
	iou = iouCalc(classLabels, validClasses, voidClass = void)
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, loss_running, acc_running],
		prefix='Test: ')
	
	# set model in training mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
			data_time.update(time.time()-end)

			# input resolution
			res = inputs.shape[2]*inputs.shape[3]
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()
	
			# forward
			outputs = model(inputs)
			preds = torch.argmax(outputs, 1)
			loss = criterion(outputs, labels)
			
			# Statistics
			bs = inputs.size(0) # current batch size
			loss = loss.item()
			loss_running.update(loss, bs)
			corrects = torch.sum(preds == labels.data)
			nvoid = int((labels==void).sum())
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			# Calculate IoU scores of current batch
			iou.evaluateBatch(preds, labels)
			
			# Save visualizations of first batch
			if epoch_step == 0 and maskColors is not None:
				for i in range(inputs.size(0)):
					filename = os.path.splitext(os.path.basename(filepath[i]))[0]
					# Only save inputs and labels once
					if epoch == 0:
						img = visim(inputs[i,:,:,:], mean, std)
						label = vislbl(labels[i,:,:], maskColors)
						if len(img.shape) == 3:
							cv2.imwrite('images/{}.png'.format(filename),img[:,:,::-1])
						else:
							cv2.imwrite('images/{}.png'.format(filename),img)
						cv2.imwrite('images/{}_gt.png'.format(filename),label[:,:,::-1])
					# Save predictions
					pred = vislbl(preds[i,:,:], maskColors)
					cv2.imwrite('images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# print progress info
			progress.display(epoch_step)
		
		miou, iou_summary, confMatrix = iou.outputScores(epoch=epoch)
		if log is not None:
			log.info(' * Acc {:.3f}'.format(acc_running.avg))
		print(iou_summary)

	return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary

def eval_evaluate(dataloader, model, criterion, classLabels, validClasses, log=None, void=-1, maskColors=None, mean=None, std=None, save_root='',save_suffix='', save=False):
	save_path = os.path.join('results', save_root)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	paths = [os.path.join(save_path,'gt'), os.path.join(save_path,'visuals'), os.path.join(save_path,'original'), 
				# os.path.join(save_path,'labelTrainIds_invalid'),os.path.join(save_path,'confidence'), 
    			# os.path.join(save_path,'labelTrainIds')
    		]
	for p in paths:
		if not os.path.exists(p):
			os.makedirs(p)

	iou = iouCalc(classLabels, validClasses, voidClass = void)
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	acc_running = AverageMeter('Acc', ':6.2f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, loss_running, acc_running],
		prefix='Test: ')
	
	# set model in training mode
	model.eval()


	with torch.no_grad():
		end = time.time()
		for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
			data_time.update(time.time()-end)

			# input resolution
			h,w = labels.shape[-2:]
			res = h*w
			
			inputs = inputs.float().cuda()
			labels = labels.long().cuda()
	
			# forward
			outputs = model(inputs)
			preds = torch.argmax(outputs, 1)
			confidence = torch.softmax(outputs, 1).max(1)[0]
			loss = criterion(outputs, labels)
			# preds = transforms.Resize((h,w), interpolation=Image.NEAREST)(preds)
			# loss = criterion(outputs, transforms.Resize((512,1024), interpolation=Image.NEAREST)(labels))
			
			# Statistics
			bs = inputs.size(0) # current batch size
			loss = loss.item()
			loss_running.update(loss, bs)
			corrects = torch.sum(preds == labels.data)
			nvoid = int((labels==void).sum())
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			# Calculate IoU scores of current batch
			iou.evaluateBatch(preds, labels)
			
			# Save visualizations
			if save:
				for i in range(inputs.size(0)):
					filename = os.path.splitext(os.path.basename(filepath[i]))[0]

					img = visim(inputs[i,:,:,:], mean, std)
					label = vislbl(labels[i,:,:], maskColors)
					pred = vislbl(preds[i,:,:], maskColors)
		
					cv2.imwrite(save_path + f'/visuals/{filename}-{save_suffix}-{acc:.2f}.png',pred[:,:,::-1])
					cv2.imwrite(save_path + f'/visuals/{filename}-original.png',img[:,:,::-1])
					cv2.imwrite(save_path + f'/visuals/{filename}-gt.png',label[:,:,::-1])
					# Save predictions
					# preds = transforms.Resize(input_shape, interpolation=Image.NEAREST)(preds)
					pred = vislbl(preds[i,:,:], maskColors)
					cv2.imwrite(save_path + '/labelTrainIds/{}.png'.format(filename),pred[:,:,::-1])
					cv2.imwrite(save_path + '/labelTrainIds_invalid/{}.png'.format(filename),pred[:,:,::-1])
					con = (confidence[i].cpu().numpy()*65536).astype(np.uint16)
					cv2.imwrite(save_path + '/confidence/{}.png'.format(filename),con)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# print progress info
			progress.display(epoch_step)
		
		miou, iou_summary, confMatrix = iou.outputScores(epoch=0)
		if log is not None:
			log.info(' * Acc {:.3f}'.format(acc_running.avg))
		print(iou_summary)

	return acc_running.avg, loss_running.avg, miou, confMatrix, iou_summary


def test_evaluate(dataloader, model, criterion, classLabels, validClasses, log=None, void=-1, maskColors=None, mean=None, std=None, save_root='',input_shape=(1080,1920)):
	save_path = os.path.join('results', save_root)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	paths = [os.path.join(save_path,'labelTrainIds'), 
			# os.path.join(save_path,'img'),
			os.path.join(save_path,'labelTrainIds_invalid'),
			os.path.join(save_path,'confidence'),
			os.path.join(save_path,'visuals')
   			]
	for p in paths:
		if not os.path.exists(p):
			os.makedirs(p)

	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	progress = ProgressMeter(
		len(dataloader),
		[batch_time],
		prefix='Test: ')
	
	# set model in training mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for epoch_step, (inputs, filepath) in enumerate(dataloader):
			data_time.update(time.time()-end)
			
			inputs = inputs.float().cuda()
	
			# forward
			outputs = model(inputs)
			preds = torch.argmax(outputs, 1)

			confidence = torch.softmax(outputs, 1).max(1)[0]
			# loss = criterion(outputs, transforms.Resize((512,1024), interpolation=Image.NEAREST)(labels))
			
			# Statistics
			bs = inputs.size(0) # current batch size
			
			# Save visualizations
			for i in range(inputs.size(0)):
				filename = os.path.splitext(os.path.basename(filepath[i]))[0]

				# img = visim(inputs[i,:,:,:], mean, std)
				# if len(img.shape) == 3:
				# 	cv2.imwrite(save_path + '/visuals/{}.png'.format(filename),img[:,:,::-1])
				# else:
				# 	cv2.imwrite(save_path + '/visuals/{}.png'.format(filename),img)
				# Save predictions
				preds_i = transforms.Resize(input_shape, interpolation=Image.NEAREST)(preds[i:i+1]) # resize to original size (1080,1920)
				preds_i = preds_i.permute(1,2,0).cpu().numpy()

				cv2.imwrite(save_path + '/labelTrainIds/{}.png'.format(filename),preds_i)
				cv2.imwrite(save_path + '/labelTrainIds_invalid/{}.png'.format(filename),preds_i) # dummy
				cv2.imwrite(save_path + '/confidence/{}.png'.format(filename),preds_i) # dummy

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			
			# print progress info
			progress.display(epoch_step)
			
		os.chdir(save_path)
		os.system('zip -r results.zip labelTrainIds labelTrainIds_invalid confidence')

def train_epoch_multi(dataloader, model, criterion, optimizer, epoch, log, void=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    lossCity_running = AverageMeter('LossCity', ':.4e')
    lossZurich_running = AverageMeter('LossZurich', ':.4e')
    accCity_running = AverageMeter('Acc', ':6.2f')
    accZurich_running = AverageMeter('AccZurich', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, lossCity_running, lossZurich_running, accCity_running, accZurich_running],
        prefix="Epoch: [{}]".format(epoch))
    
    # set model in training mode
    model.train()
    
    end = time.time()
    
    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputsCity, inputsZurich, labels) in enumerate(dataloader):
            d = inputsCity.shape
            data_time.update(time.time()-end)

            # input resolution
            res = d[-1]*d[-2]
            
            labels = labels.long().cuda()
            loss = 0.

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward day
            inputsCity = inputsCity.float().cuda()
            outputsCity = model(inputsCity)
            predsCity = torch.argmax(outputsCity, 1)
            lossCity = criterion(outputsCity, labels)
            loss += lossCity
            lossCity.backward()

            # forward night
            inputsZurich = inputsZurich.float().cuda()
            outputsZurich = model(inputsZurich)
            predsZurich = torch.argmax(outputsZurich, 1)
            lossZurich = criterion(outputsZurich, labels)
            loss += lossZurich
            lossZurich.backward()

            # update parameters
            optimizer.step()
            
            # Statistics
            bs = d[0] # current batch size
            lossCity = lossCity.item()
            lossZurich = lossZurich.item()
            lossCity_running.update(lossCity, bs)
            lossZurich_running.update(lossZurich, bs)

            correctsCity = torch.sum(predsCity == labels.data)
            correctsZurich = torch.sum(predsZurich == labels.data)

            nvoid = int((labels==void).sum())
            accCity = correctsCity.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            accZurich = correctsZurich.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            accCity_running.update(accCity, bs)
            accZurich_running.update(accZurich, bs)
            
            # Output training info
            progress.display(epoch_step)
            # Append current stats to csv
            with open('logs/log_batch.csv', 'a') as log_batch:
                log_batch.write('{}, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n'.format(epoch,
                                epoch_step, loss/bs, lossCity_running.avg, lossZurich_running.avg,
                                accCity, accCity_running.avg, accZurich, accZurich_running.avg))
                            
            batch_time.update(time.time() - end)
            end = time.time()
            
    log.info('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,lossCity_running.avg,lossZurich_running.avg,
                                    accCity_running.avg, accZurich_running.avg))
    
    return lossCity_running.avg, lossZurich_running.avg, accCity_running.avg, accZurich_running.avg
