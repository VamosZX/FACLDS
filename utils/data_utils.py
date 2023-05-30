import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from skimage.transform import resize
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp


import torch
from torchvision import transforms

import torch.utils.data as data

Image.LOAD_TRUNCATED_IMAGES = True


class DatasetFAC(data.Dataset):
    def __init__(self, args, phase ):
        super(DatasetFAC, self).__init__()
        self.phase = phase

        self.transform = transforms.Compose([       
            transforms.ToTensor()                
        ])

        if args.dataset == 'afew':
            data_all = np.load('/data/afew.npy', allow_pickle=True)
            data_all = data_all.item()
            self.data_all = data_all[args.split_type]
            if self.phase == 'train':
                self.data = self.data_all['data'][args.single_client]
                self.labels = self.data_all['target'][args.single_client] 
            else:
                self.test_data = np.load('/data/afew_test.npy', allow_pickle=True).item()
                self.data = self.test_data['data']
                self.labels = self.test_data['target']  

        elif args.dataset == 'mead':
            data_all = np.load('/data/mead.npy', allow_pickle=True)
            data_all = data_all.item()
            self.data_all = data_all[args.split_type]
            if self.phase == 'train':
                self.data = self.data_all['data'][args.single_client]
                self.labels = self.data_all['target'][args.single_client] 
            else:
                self.test_data = np.load('/data/mead_test.npy', allow_pickle=True).item()
                self.data = self.test_data['data']
                self.labels = self.test_data['target']

        elif args.dataset == 'youtube':
            data_all = np.load('/data/youtube.npy', allow_pickle=True)
            data_all = data_all.item()
            self.data_all = data_all[args.split_type]
            if self.phase == 'train':
                self.data = self.data_all['data'][args.single_client]
                self.labels = self.data_all['target'][args.single_client] 
            else:
                self.test_data = np.load('/data/youtube_test.npy', allow_pickle=True).item()
                self.data = self.test_data['data']
                self.labels = self.test_data['target']

        self.args = args


    def __getitem__(self, index):
       
        video_tensor_path = self.data[index]
        video = torch.load(video_tensor_path)
        target = self.labels[index]
    
        return video, torch.tensor(target)

    def __len__(self):

        return len(self.data)


def create_dataset_and_evalmetrix(args):
  
    if args.dataset == 'afew':
        args.num_classes = 4
        data_all = np.load('/data/afew.npy', allow_pickle=True).item()
        data_all = data_all[args.split_type] 
        args.dis_cvs_files = [key for key in data_all['data'].keys() if 'train' in key]
        args.clients_with_len = {name: len(data_all['data'][name]) for name in args.dis_cvs_files}

    elif args.dataset == 'mead':
        args.num_classes = 8
        data_all = np.load('/data/mead.npy', allow_pickle=True).item()
        data_all = data_all[args.split_type] 
        args.dis_cvs_files = [key for key in data_all['data'].keys() if 'train' in key]
        args.clients_with_len = {name: len(data_all['data'][name]) for name in args.dis_cvs_files}
    
    elif args.dataset == 'youtube':
        args.num_classes = 24
        data_all = np.load('/data/youtube.npy', allow_pickle=True).item()
        data_all = data_all[args.split_type] 
        args.dis_cvs_files = [key for key in data_all['data'].keys() if 'train' in key]
        args.clients_with_len = {name: len(data_all['data'][name]) for name in args.dis_cvs_files}



    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False 
    args.best_eval_loss = {}

    for single_client in args.dis_cvs_files:
        args.best_acc[single_client] = 0 if args.num_classes > 1 else 999
        args.current_acc[single_client] = []
        args.current_test_acc[single_client] = []

        args.best_eval_loss[single_client] = 9999






