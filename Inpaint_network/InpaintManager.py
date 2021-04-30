import os
import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from Inpaint_network.model.networks import Generator
from Inpaint_network.utils.tools import get_config, random_bbox, mask_image, is_image_file,\
                        default_loader, normalize, get_model_list, unnormalise

class InpaintManager():
    def __init__(self, _width, _height):
        self.config = get_config('Inpaint_network/inpaint_config.yaml')
        print('[INPAINT]: Initialising...')

        self.cuda = self.config['cuda']
        device_ids = self.config['gpu_ids']
        print('[INPAINT]: Cuda', self.cuda)
        if self.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
            device_ids = list(range(len(device_ids)))
            self.config['gpu_ids'] = device_ids
            cudnn.benchmark = True
        print('[INPAINT]: SIZE', self.config['image_shape'])
        self.TrResizer = transforms.Resize(self.config['image_shape'][:-1])
        self.TrCentral = transforms.CenterCrop(self.config['image_shape'][:-1])

        self.TrTensor = transforms.ToTensor()

        last_model_name = get_model_list('Inpaint_network/checkpoints/imagenet/hole_benchmark/'\
                                        , "gen", iteration=0)
        print('[INPAINT]: Loading Weights...')
        self.NGenerator = Generator(self.config['netG'], self.cuda, device_ids)
        self.NGenerator.load_state_dict(torch.load(last_model_name))
        model_iteration = int(last_model_name[-11:-3])
        print("[INPAINT]: Loaded with weights {}".format( model_iteration))

        if self.cuda:
            self.NGenerator = self.NGenerator.cuda()

        ## define upscaler 
        self.upscale = transforms.Resize((_height, _width))

    def preprocess(self, image, mask):
        ## resize image and mask to self sizes
        image = Image.fromarray(image)
        mask  = Image.fromarray(mask.squeeze() * 255 , 'L')

        image = self.TrCentral(self.TrResizer(image))
        mask  = self.TrCentral(self.TrResizer(mask ))

        image = self.TrTensor(image)
        mask  = self.TrTensor(mask)[0].unsqueeze(dim=0)

        image = normalize(image)
        image = image * (1. - mask)
        image = image.unsqueeze(dim=0)
        mask  = mask.unsqueeze(dim=0)
        
        return image, mask

    def inference(self, image, mask):
        with torch.no_grad():
            image, mask = self.preprocess(image, mask)

            if self.cuda:
                image = image.cuda()
                mask  = mask.cuda()

            x1, x2, offset_flow = self.NGenerator(image, mask)
            inpainted_result = x2 * mask + image * (1. - mask)
            inpainted_result = self.upscale( unnormalise(inpainted_result) )
        return inpainted_result