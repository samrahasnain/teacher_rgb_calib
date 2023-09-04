import os
import cv2
import torch
from torch.utils import data
import numpy as np
import random
import glob
import albumentations as albu
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
#from pytorch_lightning.trainer.supporters import CombinedLoader
from torchsampler import ImbalancedDatasetSampler
random.seed(10)

class DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, image_size):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*'))
        self.image_size = image_size


    def __getitem__(self, idx):
        im_name = self.images[idx]
        gt_name = self.gts[idx]
        sal_image , im_size= load_image( im_name, self.image_size)
        sal_label,sal_edge = load_sal_label(gt_name, self.image_size)

        sal_image, sal_label = cv_random_crop_rgb(sal_image,  sal_label, self.image_size)
        sal_image = sal_image.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))


        image = torch.Tensor(sal_image)
        mask = torch.Tensor(sal_label)
        #print('duts',image.shape,mask.shape)
 
        #sample1 = {'rgb_image': image,'rgb_label': mask}
        return image,mask
        

    def __len__(self):
        return len(self.images)

class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list,image_size):
        self.sal_root = data_root
        self.sal_source = data_list
        self.image_size = image_size

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        de_name = self.sal_list[item % self.sal_num].split()[1]
        gt_name = self.sal_list[item % self.sal_num].split()[2]
        sal_image , im_size= load_image(os.path.join(self.sal_root, im_name), self.image_size)
        sal_depth, im_size = load_image(os.path.join(self.sal_root, de_name), self.image_size)
        sal_label,sal_edge = load_sal_label(os.path.join(self.sal_root, gt_name), self.image_size)

        sal_image, sal_depth, sal_label = cv_random_crop(sal_image, sal_depth, sal_label, self.image_size)
        sal_image = sal_image.transpose((2, 0, 1))
        sal_depth = sal_depth.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))
        #sal_edge = sal_edge.transpose((2, 0, 1))

        sal_image = torch.Tensor(sal_image)
        sal_depth = torch.Tensor(sal_depth)
        sal_label = torch.Tensor(sal_label)
        #sal_edge = torch.Tensor(sal_edge)
        dq=depth_quality_score(sal_depth)
        #print('rgbd',sal_image.shape,sal_depth.shape,sal_label.shape)
        #sample = {'sal_image': sal_image, 'sal_depth': sal_depth, 'sal_label': sal_label, 'sal_edge': sal_edge,'depth_quality_score':dq,'name': self.sal_list[item % self.sal_num].split()[0].split('/')[1]}
        return sal_image,sal_label

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list,image_size):
        self.data_root = data_root
        self.data_list = data_list
        self.image_size = image_size
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item].split()[0]), self.image_size)
        depth, de_size = load_image_test(os.path.join(self.data_root, self.image_list[item].split()[1]), self.image_size)
        image = torch.Tensor(image)
        depth = torch.Tensor(depth)
        return {'image': image, 'name': self.image_list[item % self.image_num].split()[0].split('/')[1],
                'size': im_size, 'depth': depth}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset1 = ImageDataTrain(config.train_root, config.train_list, config.image_size)
        dataset2 = DatasetGenerate(config.img_folder, config.gt_folder,config.image_size)
        
        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        train_loader = data.DataLoader(dataset,
                               batch_size=config.batch_size,
                               num_workers=config.num_thread,
                               shuffle=True,
                               collate_fn=None,  # Modify this if custom collation function is needed
                               pin_memory=True)

        print(len(train_loader))
        return train_loader
      
    else:
        dataset = ImageDataTest(config.test_root, config.test_list, config.image_size)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                                      num_workers=config.num_thread, pin_memory=pin)
    
        return data_loader



def load_image(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    return in_,im_size



def load_image_test(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (image_size, image_size))
    in_ = Normalization(in_)
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path,image_size):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #gradient
    gX = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    combined = np.array(combined , dtype=np.float32)
    combined  = cv2.resize(combined , (image_size, image_size))
    combined  = combined  / 255.0
    combined  = combined [..., np.newaxis]

    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label, (image_size, image_size))
    label = label / 255.0
    label = label[..., np.newaxis]
    return label,combined


def cv_random_crop(image, depth, label,image_size):
    crop_size = int(0.0625*image_size)
    croped = image_size - crop_size
    top = random.randint(0, crop_size)  #crop rate 0.0625
    left = random.randint(0, crop_size)

    image = image[top: top + croped, left: left + croped, :]
    depth = depth[top: top + croped, left: left + croped, :]
    label = label[top: top + croped, left: left + croped, :]
    image = cv2.resize(image, (image_size, image_size))
    depth = cv2.resize(depth, (image_size, image_size))
    label = cv2.resize(label, (image_size, image_size))
    label = label[..., np.newaxis]
    return image, depth, label

def cv_random_crop_rgb(image,  label,image_size):
    crop_size = int(0.0625*image_size)
    croped = image_size - crop_size
    top = random.randint(0, crop_size)  #crop rate 0.0625
    left = random.randint(0, crop_size)

    image = image[top: top + croped, left: left + croped, :]
    
    label = label[top: top + croped, left: left + croped, :]
    image = cv2.resize(image, (image_size, image_size))

    label = cv2.resize(label, (image_size, image_size))
    label = label[..., np.newaxis]
    return image, label

def Normalization(image):
    in_ = image[:, :, ::-1]
    in_ = in_ / 255.0
    in_ -= np.array((0.485, 0.456, 0.406))
    in_ /= np.array((0.229, 0.224, 0.225))
    return in_


def depth_quality_score(depth):
    #print('depth',depth)
    score_m=torch.mean(depth)
    #print('mean',score_m)
    score_CV=score_m/torch.std(depth)
    #print('std',score_CV)
    P1=torch.tensor(torch.numel((depth<=0.4).nonzero(as_tuple=False))/torch.numel(depth))
    P2=torch.tensor(torch.numel(((depth>0.4)&(depth<=0.6)).nonzero(as_tuple=False))/torch.numel(depth))
    P3=torch.tensor(torch.numel((depth>0.6).nonzero(as_tuple=False))/torch.numel(depth))
    #print(P1,P2,P3)
    constant=torch.tensor(3)
    H=-(P1*(torch.log(P1)/torch.log(constant)))-(P2*(torch.log(P2)/torch.log(constant)))-(P3*(torch.log(P3)/torch.log(constant)))
    #print(H)
    lammda=torch.exp((1-score_m)*score_CV*H)-1
    #print(lammda)
    return lammda

def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if ver == 2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90()
            ], p=0.5),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transforms

def gt_to_tensor(gt):
    gt = cv2.imread(gt)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) / 255.0
    gt = np.where(gt > 0.5, 1.0, 0.0)
    gt = torch.tensor(gt, device='cuda', dtype=torch.float32)
    gt = gt.unsqueeze(0).unsqueeze(1)

    return gt
