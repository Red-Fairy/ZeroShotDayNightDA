"""
Author: Rundong Luo, rundongluo2002@gmail.com
"""

import os
from utils.helpers import get_test_trans, get_eval_trans
from utils.routines import evaluate, test_evaluate, eval_evaluate
from datasets.cityscapes_ext import CityscapesExt
from datasets.nighttime_driving import NighttimeDrivingDataset
from datasets.dark_zurich import DarkZurichTestDataset, DarkZurichDataset
from datasets.night_city import NightCityDataset
from models.refinenet import RefineNet

import torch
import torch.nn as nn

def main(args):

    print(args)

    # Define data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    target_size = (512,1024)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    cs_path = '../Cityscapes'
    nd_path = '../NighttimeDrivingTest'
    dz_val_path = '../DarkZurichVal'
    dz_path = '../DarkZurichTest/night'
    nc_path = '../NightCity'

    valset = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
    testset_day = CityscapesExt(cs_path, split='test', target_type='semantic', transforms=test_trans)
    testset_nd = NighttimeDrivingDataset(nd_path, transforms=test_trans)
    valset_dz = DarkZurichDataset(dz_val_path, transforms=test_trans)
    testset_dz = DarkZurichTestDataset(dz_path, transforms=test_trans)
    testset_nc = NightCityDataset(nc_path, transforms=test_trans)

    dataloaders = {}
    dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_nd'] = torch.utils.data.DataLoader(testset_nd, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['val_dz'] = torch.utils.data.DataLoader(valset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_dz'] = torch.utils.data.DataLoader(testset_dz, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_nc'] = torch.utils.data.DataLoader(testset_nc, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)
    model = RefineNet(num_classes, pretrained=False)

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Load weights from checkpoint
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Validate
    print('--- Validation - daytime ---')
    val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(dataloaders['val'],
        model, criterion, 0, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
    print(miou_cs)
    
    print('--- Validation - Nighttime Driving ---')
    test_acc_nd, test_loss_nd, miou_nd, confmat_nd, iousum_nd = eval_evaluate(dataloaders['test_nd'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'ND'), save=args.save)
    print(miou_nd)
    
    print('--- Validation - Dark Zurich ---')
    test_acc_dz, test_loss_dz, miou_dz, confmat_dz, iousum_dz = eval_evaluate(dataloaders['val_dz'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'DZ_Val'), save=args.save)
    print(miou_dz)
    
    print('--- Test - Dark Zurich ---')
    with torch.no_grad():
        test_evaluate(dataloaders['test_dz'],
            model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
            void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'DZ_Test'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('--weight', type=str, required=True,
                        help='load weight file')
    parser.add_argument('--model', type=str, default='refinenet',
                        help='model (refinenet or deeplabv3)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--save_path', default='',type=str)
    parser.add_argument('--save', action='store_true', help='save visual results')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    main(args)
