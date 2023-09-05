# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:51:21 2019 by Attila Lengyel - attila@lengyel.nl
"""

import os
import numpy as np

from PIL import Image

from torchvision.datasets import Cityscapes
import torch
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, maxImgNum=1e5, name=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and ((name in fname) if name is not None else 1):
                path = os.path.join(root, fname)
                images.append(path)
                # print(path)
    # random.shuffle(images)
    print(len(images))
    return images[:min(maxImgNum, len(images))]

class CityZurich(Cityscapes):
    
    voidClass = 19

    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to ids
    trainid2id = np.arange(len(id2trainid))[np.argsort(id2trainid)]
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')

    def __init__(self, root, rootZurich='/mnt/netdisk/luord/datasets/CycleGAN_generate/city2zurich200', split='train', target_type='semantic', transforms=None):
        super().__init__(root, split=split, target_type=target_type, transforms=transforms)
        self.rootZurich = rootZurich
        self.imagesZurich = []

        for city in os.listdir(self.rootZurich):
            img_dir = os.path.join(self.rootZurich, city)
            for file_name in os.listdir(img_dir):
                self.imagesZurich.append(os.path.join(img_dir, file_name))

        print(len(self.imagesZurich))
        assert (len(self.images) == len(self.imagesZurich))


    def __getitem__(self, index):
        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')
        filepathZurich = self.imagesZurich[index]
        imageZurich = Image.open(filepathZurich).convert('RGB')

        # print(filepath, filepathZurich)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, imageZurich, target = self.transforms(image, imageZurich, target)
            
        target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW

        return image, imageZurich, target

class DualNight(Cityscapes): # override original cityscapes inputs
    
    voidClass = 19

    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to ids
    trainid2id = np.arange(len(id2trainid))[np.argsort(id2trainid)]
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')

    def __init__(self, root, rootNight1, rootNight2, split='train', target_type='semantic', transforms=None):
        super().__init__(root, split=split, target_type=target_type, transforms=transforms)
        self.rootNight1 = rootNight1
        self.rootNight2 = rootNight2
        self.imagesNight1 = [] # imagesNight2 are stored in self.images

        for city in os.listdir(self.rootNight1):
            img_dir = os.path.join(self.rootNight1, city)
            for file_name in os.listdir(img_dir):
                self.imagesNight1.append(os.path.join(img_dir, file_name))
        
        self.images = []

        for city in os.listdir(self.rootNight2):
            img_dir = os.path.join(self.rootNight2, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))

    def __getitem__(self, index):

        filepath1 = self.imagesNight1[index]
        imageNight1 = Image.open(filepath1).convert('RGB')
        filepath2 = self.images[index]
        imageNight2 = Image.open(filepath2).convert('RGB')

        # print(filepath, filepathZurich)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            imageNight1, imageNight2, target = self.transforms(imageNight1, imageNight2, target)
            
        target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW

        return imageNight1, imageNight2, target
    
class CityCity(Cityscapes):
    
    voidClass = 19

    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to ids
    trainid2id = np.arange(len(id2trainid))[np.argsort(id2trainid)]
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')

    def __init__(self, root, split='train', target_type='semantic', transforms=None):
        super().__init__(root, split=split, target_type=target_type, transforms=transforms)

    def __getitem__(self, index):
        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')
        image2 = image.copy()

        # print(filepath, filepathZurich)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, image2, target = self.transforms(image, image2, target)
            
        target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW

        return image, image2, target


class CityNegative(Cityscapes):
    
    voidClass = 19

    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass

    # Convert train_ids to ids
    trainid2id = np.arange(len(id2trainid))[np.argsort(id2trainid)]
    
    # Convert train_ids to colors
    mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    mask_colors.append([0,0,0])
    mask_colors = np.array(mask_colors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')

    def __init__(self, root, split='train', target_type='semantic', transforms=None):
        super().__init__(root, split=split, target_type=target_type, transforms=transforms)
        self.len = len(self.images)

    def __getitem__(self, index):
        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')
        index2 = random.randint(0, self.len - 1)
        image2 = Image.open(self.images[index2]).convert('RGB')

        # print(filepath, filepathZurich)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, image2, target = self.transforms(image, image2, target)
            
        target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW

        return image, image2, target