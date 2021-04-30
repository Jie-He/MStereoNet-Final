## Loads middlebury dataset
## no image effect/crop is applied

import os
import numpy as np
from os import listdir
from os.path import join

import torch
import torch.utils.data as data
from torchvision import transforms
from DatasetLoader.img_utils import pil_loader, read_pfm

class MIDDLEBURY_dataset(data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        self.left_image_names  = []
        self.right_image_names = []
        self.left_disparity  = []
        self.right_disparity = []

        self.names = [ 'im0.png' , 'im1.png', ## left, right (img)
                       'disp0.pfm', 'disp1.pfm']## left, right (disp)

        ## Load the subdirectories
        for fname in listdir(dataset_path):
            self.left_image_names.append( join(fname, self.names[0]) )
            self.right_image_names.append(join(fname, self.names[1]) )
            self.left_disparity.append(   join(fname, self.names[2]) )
            self.right_disparity.append(  join(fname, self.names[3]) )
        
        print('Loaded', len(self.left_image_names), 'paths')
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.left_image_names)
    
    def __getitem__(self, indx):
        left_image = pil_loader( join(self.dataset_path, self.left_image_names[indx]))
        right_image= pil_loader( join(self.dataset_path, self.right_image_names[indx]))

        left_disparity = read_pfm( join(self.dataset_path, self.left_disparity[indx]))
        right_disparity= read_pfm( join(self.dataset_path, self.right_disparity[indx]))

        left_disparity[left_disparity == np.inf] = 0
        right_disparity[right_disparity == np.inf] = 0

        left_disparity = np.ascontiguousarray( left_disparity)
        right_disparity= np.ascontiguousarray(right_disparity)

        inputs = {'left_image':  (self.to_tensor( left_image) - 0.45) / 0.255,
                  'right_image': (self.to_tensor(right_image) - 0.45) / 0.255,
                  'disparity':   torch.from_numpy( left_disparity).float(),
                  'rdisparity':  torch.from_numpy(right_disparity).float()}

        return inputs