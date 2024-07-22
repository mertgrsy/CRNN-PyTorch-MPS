#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
#import lmdb
import os
import six
import sys
from utils import mytools
from PIL import Image
import PIL.ImageOps
import numpy as np

class lmdbDataset(Dataset):
    def __init__(self, root: str, transform=True, target_transform=None):
        if not os.path.isdir(root):
            raise RuntimeError(f"Given input dataset path, {root}, not exist!")

        self.data_list = [root + "/" + file for file in os.listdir(root) if file.split(".")[-1] in ["jpg", "jpeg", "png", "JPG", "JPEG"]]
        if len(self.data_list) == 0:
            raise RuntimeError(f"Data_list is empty!")

        self.nSamples = len(self.data_list)

        data_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        # transforms.RandomResizedCrop(size=(image_height, image_width), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
        self.transform = data_transforms
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        convert_tensor = transforms.ToTensor()
        file = self.data_list[index]
        file = file.replace(".jpg",".def").replace(".jpeg",".def").replace(".png",".def").replace(".JPG",".def").replace(".JPEG",".def")

        with open(file, 'r') as f:
            data = f.read() 

        data = data.split('\n')
        plate = data[1].split("PLATE=")
        plate = plate[1]

        tmp = data[2].split("YOLO_PLT_3CLASS_1=")
        dims = tmp[1].split(" ")
        lbl, x_center, y_center, w, h = dims
        x_center = float(x_center)
        y_center = float(y_center)
        w = float(w)
        h = float(h)

        # img = Image.open(self.data_list[index]).convert('L')  # read -> frame -> grayscale
        img = Image.open(self.data_list[index])
        img_w, img_h = img.size
        x1, y1, x2, y2 = mytools.converter(x_center, y_center, w, h, img_w, img_h)

        # file = self.data_list[index].split(".")
        
        img = img.crop((x1, y1, x2, y2))
        if lbl == "1":
            img = PIL.ImageOps.invert(img)
            # img.show()

        # print(f"label: {data[0]}")
        plate = plate.upper()
        label = plate.encode("utf-8")

        if self.transform is not None:
            # print("yes transform is happening!")
            img = self.transform(img)
            # img.show()

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img.convert('L'), label)

class ValDataset(Dataset):
    def __init__(self, root: str, transform=True, target_transform=None):
        if not os.path.isdir(root):
            raise RuntimeError(f"Given input dataset path, {root}, not exist!")

        self.data_list = [root + "/" + file for file in os.listdir(root) if file.split(".")[-1] in ["jpg", "jpeg", "png"]]
        if len(self.data_list) == 0:
            raise RuntimeError(f"Data_list is empty!")

        self.nSamples = len(self.data_list)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        convert_tensor = transforms.ToTensor()
        file = self.data_list[index]
        file = file.replace(".jpg",".def").replace(".jpeg",".def").replace(".png",".def").replace(".JPG",".def").replace(".JPEG",".def")


        with open(file, 'r') as f:
            data = f.read() 

        data = data.split('\n')
        plate = data[1].split("PLATE=")
        plate = plate[1]

        tmp = data[2].split("YOLO_PLT_3CLASS_1=")
        dims = tmp[1].split(" ")
        lbl, x_center, y_center, w, h = dims
        x_center = float(x_center)
        y_center = float(y_center)
        w = float(w)
        h = float(h)

        # img = Image.open(self.data_list[index]).convert('L')  # read -> frame -> grayscale
        img = Image.open(self.data_list[index]).convert('L')
        img_w, img_h = img.size
        x1, y1, x2, y2 = mytools.converter(x_center, y_center, w, h, img_w, img_h)

        # file = self.data_list[index].split(".")
        
        img = img.crop((x1, y1, x2, y2))
        if lbl == "1":
            img = PIL.ImageOps.invert(img)
            # img.show()

        # print(f"label: {data[0]}")
        plate = plate.upper()
        label = plate.encode("utf-8")
        # print(f"label: {label}")
        # img.show()
        if self.transform is not None:
            # print("yes transform is happening!")
            img = self.transform(img)
            # img.show()

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
