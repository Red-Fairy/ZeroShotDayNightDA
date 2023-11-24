import os
import torch

from data import iterator_factory

import losses_darkening
from network.zero_dce import enhance_net_nopool

import argparse
from utils.utils import *

from network.resnet_i3d import Res_I3D

def train(args):
    
    train_iter, _ = iterator_factory.creat(name='NormalLight',
                                                batch_size=args.batch_size,
                                                clip_length=16,
                                                train_interval=2,
                                                val_interval=2,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225],
                                                seed=0,
                                                use_flow=False)

    device = torch.device(f"cuda:{args.gpu_id}")
    scale_factor = args.scale_factor
    DCE_net = enhance_net_nopool(scale_factor, curve_round=args.curve_round,
                                 encode_dim=args.encode_dim,
                                 down_scale=args.down_scale).cuda(device)
    
    L_color = losses_darkening.L_color()
    L_down = losses_darkening.L_down()
    L_exp = losses_darkening.L_exp(args.patch_size, args.exp_weight)
    L_tv = losses_darkening.L_TV(mid_val=args.tv_mid)
    
    ltv, lcol, lexp, lsim, ldown, ltotal = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    if args.sim:
        L_sim = losses_darkening.L_sim()
        resnet_I3D = Res_I3D().cuda(device).eval()
        state_dict = torch.load(args.feature_extractor)['state_dict']
        state_dict_new = {}
        for k, v in state_dict.items():
            state_dict_new[k.replace('module.','')] = v
        resnet_I3D.load_state_dict(state_dict_new, strict=True)
        resnet_I3D.requires_grad_(False)
    else:
        L_sim = None

    optimizer = torch.optim.Adam(DCE_net.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(i) for i in args.decreasing_lr.split(',')], gamma=0.1)

    DCE_net.train()
    low_exp, high_exp = args.exp_range.split('-')
    low_exp, high_exp = float(low_exp), float(high_exp)
    
    iteration = 1

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    while iteration <= args.max_iters:
        for video, _ in train_iter:
            video = video.cuda(device)
            d = video.shape
            full_frames = video.permute(0,2,1,3,4).reshape(d[0]*d[2],d[1],d[3],d[4])
            # denormalize
            full_frames = full_frames * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda(device) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda(device)
            exp = torch.rand(d[0]*d[2], 1, 1, 1).cuda(device) * (high_exp - low_exp) + low_exp

            full_frames = torch.clamp(full_frames, 0, max=args.max_value)

            darkened_frames, [enhance_r, down_scale] = DCE_net(full_frames, exp)

            loss = 0.
            loss_TV = args.tv_weight * L_tv(enhance_r)
            loss += loss_TV
            ltv.update(loss_TV.item(), args.batch_size)

            loss_col = args.color_weight*torch.mean(L_color(darkened_frames))
            loss += loss_col
            lcol.update(loss_col.item(), args.batch_size)

            darkened_video = darkened_frames.reshape(d[0],d[2],d[1],d[3],d[4]).permute(0,2,1,3,4)
            video = full_frames.reshape(d[0],d[2],d[1],d[3],d[4]).permute(0,2,1,3,4)
            # normalize

            if L_sim is not None:
                out_ori = resnet_I3D(video, return_feat=True)[1]
                out_low = resnet_I3D(darkened_video, return_feat=True)[1]
                loss_sim = L_sim([out_ori], [out_low]) * args.sim_weight
                loss += loss_sim
                lsim.update(loss_sim.item(), args.batch_size)
            else:
                loss_sim = torch.zeros(1)

            loss_exp = torch.mean(L_exp(darkened_frames, exp))
            loss += loss_exp
            lexp.update(loss_exp.item(), args.batch_size)

            loss_down = args.down_weight*L_down(down_scale)
            loss += loss_down
            ldown.update(loss_down.item(), args.batch_size)

            ltotal.update(loss.item(), args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(
                DCE_net.parameters(), args.grad_clip_norm)
            optimizer.step()

            if ((iteration+1) % args.display_iter) == 0:
                log.info('Iter [{}/{}] Loss: {:.4f} Loss_TV: {:.4f} Loss_down: {:.4f} Loss_col: {:.4f} Loss_exp: {:.4f}, Loss_sim {:4f}'.format(
                            iteration+1, args.max_iters, loss.item(), loss_TV.item(), loss_down.item(), loss_col.item(), loss_exp.item(), loss_sim.item()))
            
            iteration += 1
            scheduler.step()
            
            if iteration > args.max_iters:
                break

            if iteration % 1000 == 0:
                torch.save(DCE_net.state_dict(
                    ), 'checkpoints/' + args.experiment + "/Iter" + str(iteration) + '.pth')

        scheduler.step()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--decreasing_lr', default='1000,3000', type=str)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--experiment', type=str,
                        required=True, help="Name of the folder where the checkpoints will be saved")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--exp_range', type=str, default='0-0.5')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)

    parser.add_argument('--exp_weight', type=float, default=10)
    parser.add_argument('--max_value', type=float, default=0.8)
    parser.add_argument('--color_weight',type=float,default=25)

    parser.add_argument('--tv_weight',type=float,default=1600)
    parser.add_argument('--tv_mid', type=float, default=0.02)
    parser.add_argument('--sim',action='store_true')
    parser.add_argument('--sim_weight',type=float,default=0.1)
    parser.add_argument('--curve_round',type=int,default=8)

    parser.add_argument('--encode_dim',type=int,default=1)
    parser.add_argument('--down_scale',type=float,default=0.95)
    parser.add_argument('--down_weight',type=float,default=1)

    parser.add_argument('--feature_extractor',type=str,default='/mnt/netdisk/luord/CVPR23/action/experiments/baseline-bsz48/action_ep-0050.pth')

    args = parser.parse_args()

    if not os.path.exists(os.path.join('checkpoints', args.experiment)):
        os.makedirs(os.path.join('checkpoints', args.experiment))
    if not os.path.exists(os.path.join('checkpoints', args.experiment, 'outputs')):
        os.makedirs(os.path.join('checkpoints', args.experiment, 'outputs'))

    log = logger(os.path.join('checkpoints', args.experiment))
    log.info(str(args))

    train(args)


