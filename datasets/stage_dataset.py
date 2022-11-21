import os
import torch
import torch.utils.data as data
import numpy as np
import sys
import glob
from natsort import ns, natsorted
import random
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm
import math
import torchvision.datasets as datasets
import importlib
import inspect
from collections import OrderedDict
import shutil

IMG_EXTENSIONS = ['.jpg', '.JPG', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_stage1(data_dir, class_to_idx, mode):
    images = []
    data_dir = os.path.expanduser(data_dir)
    N_SAMPLES_PER_CLASS_BASE = [0]*len(class_to_idx.keys())
    for class_type in class_to_idx.keys():
        patient_list = os.listdir(os.path.join(data_dir, 'Stage1', mode, class_type))
        for patient_name in patient_list:
            group_list = os.listdir(os.path.join(data_dir, 'Stage1', mode, class_type, patient_name))
            for group_name in group_list:
                imglist_dir = os.path.join(data_dir, 'Stage1', mode, class_type, patient_name, group_name)
                if os.path.isdir(imglist_dir):
                    imglist = [item for item in os.listdir(imglist_dir) if is_image_file(item)]
                    for fname in sorted(imglist):
                        path = os.path.join(data_dir, 'Stage1', mode, class_type, patient_name, group_name, fname)
                        item = (path, class_to_idx[class_type])
                        N_SAMPLES_PER_CLASS_BASE[class_to_idx[class_type]] += 1
                        images.append(item)
    return images, N_SAMPLES_PER_CLASS_BASE


class StageDataset(data.Dataset):
    def __init__(self, dataset_cfg=None, state='train'):
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        self.exp_name = self.dataset_cfg.exp_name
        self.data_dir = self.dataset_cfg.data_dir
        self.mode = state
        self.imgsize = self.dataset_cfg.imgsize
        self.data_augmentation = self.dataset_cfg.data_augmentation
        if self.mode == 'train':
            self.out_name = False
        else:
            self.out_name = True
        self.stage = self.dataset_cfg.stage_num
        if self.stage == 2:
            self.imgseq_num = self.dataset_cfg.imgseq_num
        self.classes = ['negative', 'positive']
        self.class_to_idx = {'negative': 0, 'positive': 1}

        if self.mode == 'train' and self.data_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.Resize(self.imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.34224), (0.14083894))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.imgsize),
                transforms.ToTensor(),
                transforms.Normalize((0.34224), (0.14083894))
            ])
        if self.stage == 1:
            print('Stage 1')
            imgs, n_samples_per_class = make_dataset_stage1(self.data_dir, self.class_to_idx, self.mode)
            if len(imgs) == 0:
                raise(RuntimeError('Found 0 images in subfolders of: ' + self.root + '\n'
                                    'Supported image extensions are: ' + ', '.join(IMG_EXTENSIONS)))
            self.imgs = imgs
            self.n_samples_per_class = n_samples_per_class
        else:
            raise ValueError('stage num error!')
    
    def __len__(self):
        if self.stage == 1:
            return len(self.imgs)
        else:
            raise ValueError('stage num error!')
    
    def __getitem__(self, index):
        if self.stage == 1:
            path, target = self.imgs[index]
            img = Image.open(path)
            img = img.crop((0, 0, 384, 384))
            img = self.transform(img)
            if self.out_name:
                group = os.path.split(path)[-2]
                group = os.path.split(group)[-1]
                filename = os.path.split(path)[-1].split('.')[0]
                filename = group + '_' + filename
                return img, target, filename
            else:
                return img, target
        else:
            raise ValueError('stage num error!')

    def get_oversampled_data(self):
        if self.stage == 1:
            length = len(self.imgs)
            num_samples = list(self.n_samples_per_class)
            selected_list = []
            for i in range(length):
                _, target = self.imgs[i]
                selected_list.append(1/num_samples[target])
            return selected_list
        else:
            raise ValueError('stage num error!')

def save_img_batch_from_ndarray(out_dir, img_array, patient_name, patient_label, image_ids, img_size=384):
    print('Saving %s in %s' % (patient_name, out_dir))
    label_to_class = {0: 'negative', 1: 'positive'}
    norm_mean = 0.34224; norm_std = 0.14083894
    if os.path.exists(os.path.join(out_dir, label_to_class[patient_label]+'_'+patient_name)):
        shutil.rmtree(os.path.join(out_dir, label_to_class[patient_label]+'_'+patient_name))
    os.mkdir(os.path.join(out_dir, label_to_class[patient_label]+'_'+patient_name))
    for i in tqdm(range(len(img_array))):
        output_imgfilename = os.path.join(out_dir, label_to_class[patient_label]+'_'+patient_name, '%s(%d)%d.jpg'%(patient_name.split('[')[0], image_ids[i][0].item(), image_ids[i][1].item()))
        # print(output_imgfilename)
        img = (img_array[i]*norm_std+norm_mean).reshape(img_size, img_size)*255
        saveimg = Image.fromarray(img.astype(np.uint8))
        saveimg.save(output_imgfilename)

if __name__ == '__main__':
    from addict import Dict
    # stage 1
    # dataset_cfg = Dict({'data_dir': './Data',
    #                     'imgsize': 224,
    #                     'data_augmentation': False,
    #                     'stage_num': 1,
    #                     'imgseq_num': 7})
    # trainset = StageDataset(dataset_cfg, state='train')
    # testset = StageDataset(dataset_cfg, state='test')
    # valset = StageDataset(dataset_cfg, state='val')
    # print(trainset.n_samples_per_class, len(trainset))
    # print(testset.n_samples_per_class, len(testset))
    # print(valset.n_samples_per_class, len(valset))
    # for i in range(len(trainset)):
    #     img, target = trainset[i]
    #     print('train img size = %s, target = %d' % (img.shape, target))
    #     break