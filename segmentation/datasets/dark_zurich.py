"""
Author: Rundong Luo, rundongluo2002@gmail.com
"""

import os
import numpy as np

from PIL import Image

from torchvision.datasets import Cityscapes

class DarkZurichDataset(Cityscapes):
    
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
    
    def __init__(self, root, split='val', transforms=None, add=''):
        self.transforms = transforms
        
        self.root = root
        self.split = split
        
        if split == 'val':
            if add != '':
                add = '_' + add
            self.imgs_root = os.path.join(root,f'rgb_anon/val/night{add}/GOPR0356')
            self.masks_root = os.path.join(root,'gt/val/night/GOPR0356')
            self.masks = [mask for mask in list(sorted(os.listdir(self.masks_root))) if 'labelIds' in mask]
        else:
            self.imgs_root = os.path.join(root,'rgb_anon/test/night')

        self.imgs = list(sorted(os.listdir(self.imgs_root)))

        if split=='val':
            assert len(self.imgs) == len(self.masks), 'Number of images and masks must be equal'
        
        assert len(self.imgs) != 0, 'No images found'
    
    def __getitem__(self, idx):
        # Define image and mask path
        img_path = os.path.join(self.imgs_root, self.imgs[idx])
        image = Image.open(img_path).convert('RGB')

        if self.split == 'val':
            mask_path = os.path.join(self.masks_root, self.masks[idx])
            target = Image.open(mask_path)
        else:
            target = None
            
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.split == 'val':
            target = self.id2trainid[target] # Convert class ids to train_ids and then to tensor: SLOW
            return image, target, img_path
        else:
            return image, img_path
        
    def __len__(self):
        return len(self.imgs)


class DarkZurichTestDataset(Cityscapes):
    
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
    
    def __init__(self, root, transforms=None):
        self.transforms = transforms
        
        self.root = root
        
        self.imgs = list(sorted(os.listdir(self.root)))
        
        assert len(self.imgs) != 0, 'No images found'
    
    def __getitem__(self, idx):
        # Define image and mask path
        img_path = os.path.join(self.root, self.imgs[idx])
        image = Image.open(img_path).convert('RGB')
            
        if self.transforms is not None:
            image= self.transforms(image)

        return image, img_path
        
    def __len__(self):
        return len(self.imgs)