import torch
from torch import nn
import torchvision
import torch.optim
import os
import argparse
from utils.data import ImageDataset
import loss_func
import numpy as np
from torchvision import transforms
from utils.utils import *
from utils.resnet import resnet18

from model import enhance_net_nopool_v2 as enhance_net_nopool # we use v2 in our paper (quadratic curve)

def train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    scale_factor = args.scale_factor

    
    DCE_net = enhance_net_nopool(scale_factor, curve_round=args.curve_round,
                                 encode_dim=args.encode_dim,
                                 down_scale=args.down_scale).cuda()

    DCE_net.train()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(args.lowlight_images_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    L_color = loss_func.L_color()
    L_down = loss_func.L_down()
    L_exp = loss_func.L_exp(args.patch_size, args.exp_weight)
    L_tv = loss_func.L_TV(mid_val=args.tv_mid)

    if args.sim:
        L_sim = loss_func.L_sim()
        feat_extractor = resnet18(pretrained=False, return_last_feature=True).cuda().eval()
        state_dict = torch.load(args.sim_model_dir)['state_dict']
        feat_extractor.load_state_dict(state_dict, strict=False)
        feat_extractor.requires_grad_ = False
    else:
        L_sim = None

    optimizer = torch.optim.Adam(DCE_net.parameters(
        ), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(i) for i in args.decreasing_lr.split(',')], gamma=0.1)

    DCE_net.train()
    low_exp, high_exp = args.exp_range.split('-')
    low_exp, high_exp = float(low_exp), float(high_exp)

    for epoch in range(1, args.num_epochs+1):
        ltv, ldown, lcol, lexp, lsim = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        ltotal = AverageMeter()
        for iteration, (img) in enumerate(train_loader):
            
            exp = torch.rand(args.train_batch_size, 1, 1, 1).cuda() * (high_exp - low_exp) + low_exp

            img = img.cuda()
            img = torch.clamp(img, 0, max=args.max_value)

            darkened_img, [enhance_r, down_scale] = DCE_net(img, exp)
            real_exp = darkened_img[0].mean().item()
            ori_exp = img[0].mean().item()
            if iteration % 50 == 0:
                torchvision.utils.save_image(
                    darkened_img[0], 'checkpoints/'+args.experiment+'/outputs/'+str(epoch)+'_'+str(iteration)+f'-{(exp[0].item()):.2f}-{real_exp:.2f}.png')
                torchvision.utils.save_image(
                    img[0], 'checkpoints/'+args.experiment+'/outputs/'+str(epoch)+'_'+str(iteration)+f'-gt-{ori_exp:.2f}.png')

            loss = 0.
            loss_TV = args.tv_weight * L_tv(enhance_r)
            loss += loss_TV
            ltv.update(loss_TV.item(), args.train_batch_size)

            loss_col = args.color_weight*torch.mean(L_color(darkened_img))
            loss += loss_col
            lcol.update(loss_col.item(), args.train_batch_size)

            loss_down = args.down_weight*L_down(down_scale)
            loss += loss_down
            ldown.update(loss_down.item(), args.train_batch_size)

            if L_sim is not None:
                out_ori = feat_extractor(img)
                out_low = feat_extractor(darkened_img)
                loss_sim = L_sim([out_ori], [out_low]) * args.sim_weight
                loss += loss_sim
                lsim.update(loss_sim.item(), args.train_batch_size)
            else:
                loss_sim = torch.zeros(1)

            loss_exp = torch.mean(L_exp(darkened_img, exp))
            loss += loss_exp
            lexp.update(loss_exp.item(), args.train_batch_size)

            ltotal.update(loss.item(), args.train_batch_size)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(
                DCE_net.parameters(), args.grad_clip_norm)
            optimizer.step()

            if ((iteration+1) % args.display_iter) == 0:
                log.info('Epoch [{}/{}], Iter [{}/{}] Loss: {:.4f} Loss_TV: {:.4f} Loss_down: {:.4f} Loss_col: {:.4f} Loss_exp: {:.4f}, Loss_sim {:4f}'.format(
                            epoch, args.num_epochs, iteration+1, len(train_loader), loss.item(), loss_TV.item(), loss_down.item(), loss_col.item(), loss_exp.item(), loss_sim.item()))

        if  epoch % 5 == 0:
            torch.save(DCE_net.state_dict(
                ), 'checkpoints/' + args.experiment + "/Epoch" + str(epoch) + '.pth')

        scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,
                        default="../classification/data/train",
                        )
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--decreasing_lr', default='5,10', type=str)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--experiment', type=str,
                        required=True, help="Name of the folder where the checkpoints will be saved")
    parser.add_argument('--exp_range', type=str, default='0-0.5')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--exp_weight', type=float, default=10)

    parser.add_argument('--max_value', type=float, default=1)
    parser.add_argument('--color_weight',type=float,default=25)

    parser.add_argument('--tv_weight',type=float,default=1600)
    parser.add_argument('--tv_mid', type=float, default=0.02)
    parser.add_argument('--sim',action='store_true')
    parser.add_argument('--sim_weight',type=float,default=0.1)
    parser.add_argument('--sim_model_dir', type=str, default='../classification/checkpoints/baseline_resnet/model_best.pt')
    parser.add_argument('--curve_round',type=int,default=8)

    parser.add_argument('--encode_dim',type=int,default=1)
    parser.add_argument('--down_scale',type=float,default=0.95)
    parser.add_argument('--down_weight',type=float,default=5)

    args = parser.parse_args()

    if not os.path.exists(os.path.join('checkpoints', args.experiment)):
        os.makedirs(os.path.join('checkpoints', args.experiment))
    if not os.path.exists(os.path.join('checkpoints', args.experiment, 'outputs')):
        os.makedirs(os.path.join('checkpoints', args.experiment, 'outputs'))

    log = logger(os.path.join('checkpoints', args.experiment))
    log.info(str(args))

    train(args)
