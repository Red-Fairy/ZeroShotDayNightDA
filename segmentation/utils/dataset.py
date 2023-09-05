import os
from torchvision import transforms
import torch
from PIL import Image
import csv
from utils.paired_transform import unpaired_random_crop, augment


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
                # print(path)
    # random.shuffle(images)
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


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, rootA='', rootB='', unaligned=False, maxImgNum = 1000, width=224, height=224):
        self.transform = transforms.Compose([
            transforms.Resize(
                int(height*1.2), interpolation=Image.BICUBIC),  # 调整输入图片的大小
            transforms.RandomCrop((height,width)),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # for CycleGAN
        ])
        self.unaligned = unaligned

        # load images from '/path/to/data/trainA'
        self.A_paths = sorted(make_dataset(rootA, maxImgNum))
        # load images from '/path/to/data/trainB'
        self.B_paths = sorted(make_dataset(rootB, maxImgNum))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        assert self.A_size == self.B_size, 'The number of images in A and B should be equal.'

    def __getitem__(self, index):
        # make sure index is within then range
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform(A_img)
        B = self.transform(B_img)

        # print(A.shape,B.shape)

        return A, B

    def __len__(self):
        return self.A_size

class ZurichPairedDataset(torch.utils.data.Dataset):
    def __init__(self, root='/mnt/netdisk/Datasets/078-DarkZurich/Dark_Zurich_train_anon/rgb_anon/train/night', 
                    ref='/home/luord/cic/experiments/3_segmentation/utils/DarkZurichPairNight.csv',transforms=None):
        self.transform = transforms
        self.A_paths = sorted(make_dataset(root))
        self.B_paths = []

        with open(ref, 'r', encoding="UTF-8") as f:
            reader = csv.reader(f)
            for path in self.A_paths:
                ref = next(reader)
                self.B_paths.append(path.replace(ref[0], ref[1]))
            f.close()
        
        assert len(self.A_paths) == len(self.B_paths), 'The number of images in A and B should be equal.'
        self.size = len(self.A_paths)
        print(self.size)

    def __getitem__(self, index):
        # read images
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform(A_img)
        B = self.transform(B_img)

        # apply paired random flip
        A, B = augment([A, B], rotation=False)
        # print(type(A),type(B))
        A, B = unpaired_random_crop(A, B)

        # print(A.shape,B.shape)

        return A, B

    def __len__(self):
        return self.size

class CityCycleZurich(torch.utils.data.Dataset):
    def __init__(self, rootA='/mnt/netdisk/wangwenjing/Datasets/Cityscapes/leftImg8bit/train', 
                    rootB='/mnt/netdisk/luord/datasets/CycleGAN_generate/city2zurich200',transforms=None, transforms2=None):
        self.transform = transforms
        self.transform2 = transforms2 if transforms2 is not None else self.transform
        self.A_paths = sorted(make_dataset(rootA))
        self.B_paths = sorted(make_dataset(rootB))
        
        assert len(self.A_paths) == len(self.B_paths), 'The number of images in A and B should be equal.'
        self.size = len(self.A_paths)
        print(self.size)

    def __getitem__(self, index):
        # make sure index is within then range
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        # print(A_path, B_path)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform(A_img)
        B = self.transform2(B_img)

        # apply paired image transformation
        A, B = augment([A, B], rotation=True)
        # print(type(A),type(B))
        A, B = unpaired_random_crop(A, B)

        # print(A.shape,B.shape)

        return A, B

    def __len__(self):
        return self.size


class CityMultiview(torch.utils.data.Dataset):
    def __init__(self, root='/mnt/netdisk/wangwenjing/Datasets/Cityscapes/leftImg8bit/train', transforms=None):
        self.transform = transforms

        self.A_paths = sorted(make_dataset(root))
        
        self.size = len(self.A_paths)
        print(self.size)

    def __getitem__(self, index):
        # make sure index is within then range
        A_path = self.A_paths[index]
        # print(A_path, B_path)
        A_img = Image.open(A_path).convert('RGB')

        # apply image transformation
        img1 = self.transform(A_img)
        img2 = self.transform(A_img)

        # apply paired image transformation
        img1, img2 = augment([img1, img2], rotation=True) # hflip, vflip, rotation
        # print(type(A),type(B))
        img1, img2 = unpaired_random_crop(img1, img2)

        return img1, img2

    def __len__(self):
        return self.size
