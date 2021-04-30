from __future__ import division, absolute_import

import torch
import torch.utils.data as data
from torchvision import transforms

import os
import numpy as np
import torch
import random
from PIL import Image
import cv2

from DatasetLoader.img_utils import pil_loader, read_pfm

class KITTI_dataset(data.Dataset):
    def __init__(self, dataset_path, iskitti2012=False):
        self.dataset_path    = dataset_path

        ## Load file names into those
        self.right_img_names = []
        self.left_img_names  = []
        self.left_dsp_names  = [] 

        ## folder names
        left_folder, right_folder, disp_left = 'image_2', 'image_3', 'disp_occ_0'
        if iskitti2012 == True:
            left_folder, right_folder, disp_left = 'colored_0', 'colored_1', 'disp_occ'
        

        ## read the names
        for fname in os.listdir(os.path.join(self.dataset_path, disp_left)):
            self.left_img_names.append(os.path.join(dataset_path, left_folder, fname))
            self.left_dsp_names.append(os.path.join(dataset_path, disp_left, fname))
            self.right_img_names.append(os.path.join(dataset_path, right_folder, fname))

        self.length = len(self.left_img_names)
        print('Loaded', self.length, 'image paths')
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        left_img  = pil_loader( self.left_img_names[index])
        right_img = pil_loader(self.right_img_names[index])

        disparity = np.array(Image.open(self.left_dsp_names[index])).astype(float)/256

        inputs = {'left_image' : ( self.to_tensor(left_img) - 0.45) / 0.255,
                  'right_image': (self.to_tensor(right_img) - 0.45) / 0.255,
                  'disparity'  : torch.from_numpy(disparity).float()}

        return inputs