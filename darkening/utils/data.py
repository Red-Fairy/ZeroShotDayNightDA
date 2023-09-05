import os
from torchvision import transforms
import torch
from PIL import Image
import csv
import numpy as np
from torchvision.datasets import Cityscapes
import torchvision.transforms.functional as TF


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, maxImgNum=1e5):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    print(len(images))
    return images[:min(maxImgNum, len(images))]

def get_index(cls_list, cls):
    for i, c in enumerate(cls_list):
        if c == cls:
            return i
    return -1

def make_labeled_dataset(dir, cls_list, maxImgNum=1e5):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    targets = []
    for root, _, fnames in sorted(os.walk(dir)):
        cls_num = get_index(cls_list, root.split('/')[-1])
        if cls_num != -1:
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
                    targets.append(cls_num)
    # random.shuffle(images)
    print(len(images))
    return images[:min(maxImgNum, len(images))], targets[:min(maxImgNum, len(targets))]

class CityscapesExt(Cityscapes):
    
    voidClass = 19
    
    # Convert ids to train_ids
    id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
    id2trainid[np.where(id2trainid==255)] = voidClass
    
    # Convert train_ids to colors
    maskColors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
    maskColors.append([0,0,0])
    maskColors = np.array(maskColors)
    
    # List of valid class ids
    validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
    validClasses[np.where(validClasses==255)] = voidClass
    validClasses = list(validClasses)
    
    # Create list of class names
    classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
    classLabels.append('void')
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        filepath = self.images[index]
        image = Image.open(filepath).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)  

        return image, filepath

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
    
