import torchvision
import torch.optim
import os
import argparse
import time
# from dataloader import ImageDataset
from model import enhance_net_nopool_v2 as enhance_net_nopool # we use v2 in our paper (quadratic curve)
import loss_func
import numpy as np
from torchvision import transforms
import torch
from utils.utils import *
import numpy as np
from utils.resnet import resnet101
from PIL import Image

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)
    print(len(images))
    return images

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root='', height=256, width=256, transform=None, path=False):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        self.paths = sorted(make_dataset(root))
        self.path = path

        self.size = len(self.paths)  # get the size of dataset A

    def __getitem__(self, index):
        # make sure index is within then range
        path = self.paths[index]
        A_img = Image.open(path).convert('RGB')
        # apply image transformation
        A = self.transform(A_img)

        # print(A.shape,B.shape)
        if self.path:
            return A, path
        else:
            return A

    def __len__(self):
        return len(self.paths)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    scale_factor = args.scale_factor
    DCE_net = enhance_net_nopool(scale_factor, curve_round=args.curve_round,
                                 encode_dim=args.encode_dim,
                                 down_scale=args.down_scale).cuda()

    DCE_net.train()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224,224)),
        transforms.ToTensor(),
    ])
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_dataset = ImageDataset(args.lowlight_images_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    L_color = loss_func.L_color()
    L_down = loss_func.L_down()
    L_exp = loss_func.L_exp(args.patch_size, args.exp_weight)
    L_tv = loss_func.L_TV(mid_val=args.tv_mid)

    if args.sim:
        L_sim = loss_func.L_sim()
        resnet = resnet101(pretrained=True, return_last_feature=True).cuda().eval()
        resnet.requires_grad_ = False
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
    ltv, ldown, lcol, lexp, lsim = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    ltotal = AverageMeter()
    while iteration <= args.max_iters:
        for img in train_loader:
            exp = torch.rand(args.train_batch_size, 1, 1, 1).cuda() * (high_exp - low_exp) + low_exp

            img = img.cuda()

            darkened_img, [enhance_r, down_scale] = DCE_net(img, exp)
            real_exp = darkened_img[0].mean().item()
            ori_exp = img[0].mean().item()
            if iteration % 50 == 0:
                torchvision.utils.save_image(
                    darkened_img[0], 'checkpoints/'+args.experiment+'/outputs/'+str(iteration)+f'-{(exp[0].item()):.2f}-{real_exp:.2f}.png')
                torchvision.utils.save_image(
                    img[0], 'checkpoints/'+args.experiment+'/outputs/'+str(iteration)+f'-gt-{ori_exp:.2f}.png')

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
                # normalize
                img_norm = (img - torch.tensor(mean).view(3, 1, 1).cuda()) / torch.tensor(std).view(3, 1, 1).cuda()
                darkened_img_norm = (darkened_img - torch.tensor(mean).view(3, 1, 1).cuda()) / torch.tensor(std).view(3, 1, 1).cuda()
                out_ori = resnet(img_norm)
                out_low = resnet(darkened_img_norm)
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
                log.info('Iter [{}/{}] Loss: {:.4f} Loss_TV: {:.4f} Loss_down: {:.4f} Loss_col: {:.4f} Loss_exp: {:.4f}, Loss_sim {:4f}'.format(
                            iteration+1, args.max_iters, loss.item(), loss_TV.item(), loss_down.item(), loss_col.item(), loss_exp.item(), loss_sim.item()))
            
            iteration += 1
            scheduler.step()
            
            if iteration > args.max_iters:
                break

            if iteration % 1000 == 0:
                torch.save(DCE_net.state_dict(
                    ), 'checkpoints/' + args.experiment + "/Iter" + str(iteration) + '.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str,
                        default="../visual-place-recognition/dataset/train/120k",
                        )
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--decreasing_lr', default='1000,3000', type=str)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=8)
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
    parser.add_argument('--color_weight',type=float,default=25)

    parser.add_argument('--tv_weight',type=float,default=1600)
    parser.add_argument('--tv_mid', type=float, default=0.02)
    parser.add_argument('--sim',action='store_true')
    parser.add_argument('--sim_weight',type=float,default=0.1)
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
