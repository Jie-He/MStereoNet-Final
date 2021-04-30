from __future__ import division, absolute_import

import os
import numpy as np
import torch
import random
from PIL import Image
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

from skimage.filters import gaussian, sobel
from skimage.color import rgb2grey

from scipy.interpolate import griddata
from DatasetLoader.img_utils import pil_loader, read_pfm
from utils import tensor2img

import time
import matplotlib.pyplot as plt

class MSCOCO_dataset(data.Dataset):
    def __init__(self, dataset_path, max_disparity, corp_height, crop_width,
                 is_training=True, background_mode='background'):
        self.dataset_path    = dataset_path
        self.max_disparity   = max_disparity
        self.crop_height     = corp_height
        self.crop_width      = crop_width
        self.background_mode = background_mode
        self.is_training     = is_training

        ## Load file names into those
        self.image_names     = []
        self.dispairty_names = []

        ## folder names
        image_folder = 'coco_img'
        depth_folder = 'coco_depth'

        ## read the names
        for fname in os.listdir(os.path.join(self.dataset_path, image_folder)):
            img_path = os.path.join(self.dataset_path, image_folder, fname)
            dsp_path = os.path.join(self.dataset_path, depth_folder, fname)
            ## read the sub folders
            for img_name in os.listdir(img_path):
                self.image_names.append(os.path.join(img_path, img_name))
                self.dispairty_names.append( os.path.join(dsp_path, (img_name[:-3] + 'pfm')))
        
        self.process_width = self.crop_width + self.max_disparity
        self.xs, self.ys = np.meshgrid(np.arange(self.process_width), np.arange(self.crop_height))

        self.length = len(self.image_names)
        self.index = 0
        print('Loaded', self.length, 'image paths')

        self.keep_aspect_ratio = True
        self.disable_synthetic_augmentation = False
        self.to_tensor = transforms.ToTensor()

        try:
            self.stereo_brightness = (0.8, 1.2)
            self.stereo_contrast = (0.8, 1.2)
            self.stereo_saturation = (0.8, 1.2)
            self.stereo_hue = (-0.01, 0.01)
            transforms.ColorJitter.get_params(
                self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
                self.stereo_hue)
        except TypeError:
            self.stereo_brightness = 0.2
            self.stereo_contrast = 0.2
            self.stereo_saturation = 0.2
            self.stereo_hue = 0.01

    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(self.process_width - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask

    def warp_image(self, inputs):

        image = np.array( inputs['left_image'] )
        if self.background_mode == 'background':
            background_image = np.array( inputs['background'] )

        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        pix_locations = self.xs - inputs['disparity']

        mask = self.get_occlusion_mask(pix_locations)
        masked_pix_locations = pix_locations * mask - self.process_width * (1 - mask)

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, self.crop_height, self.process_width)) * 10000

        for col in range(self.process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(self.crop_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(self.crop_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(self.crop_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(self.crop_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(self.crop_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(self.crop_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        # now fill occluded regions with random background
        
        if self.background_mode == 'background':
            warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]
            warped_image = warped_image.astype(np.uint8)
        else:
            _h, _w = warped_image.shape[:2]
            inpaint_mask = np.zeros(shape=(_h, _w, 1), dtype=np.uint8)

            cmask = (warped_image.max(-1) == 0)
            inpaint_mask[cmask] = 1
            inputs['mask'] = inpaint_mask
            if self.background_mode == 'opencv':
                warped_image[cmask] = 0
                warped_image = warped_image.astype(np.uint8)
                warped_image = cv2.inpaint(warped_image, inpaint_mask, 3, cv2.INPAINT_TELEA )
            else:
                warped_image[cmask] = 128
                warped_image = warped_image.astype(np.uint8)

            inputs['mask'] = inputs['mask'][:, :self.crop_width] 
                ## Do this outside
                # inference = self.Inpainter.inference(warped_image, inpaint_mask)
                # inference = (tensor2img(inference) * 255).astype(int)
                # warped_image = np.where(inpaint_mask == 0, warped_image, inference)
        # plt.imshow(warped_image)
        # plt.show()
        # exit()

        return warped_image

    def process_disparity(self, disparity, max_disparity_range=(40, 196)):
        """ Depth predictions have arbitrary scale - need to convert to a pixel disparity"""

        disparity = disparity.copy()

        # make disparities positive
        min_disp = disparity.min()
        if min_disp < 0:
            disparity += np.abs(min_disp)

        if random.random() < 0.01:
            # make max warped disparity bigger than network max -> will be clipped to max disparity,
            # but will mean network is robust to disparities which are too big
            max_disparity_range = (self.max_disparity * 1.05, self.max_disparity * 1.15)
        ## Incase some corrput data
        if disparity.max() == 0:
            disparity += 0.01
        disparity /= disparity.max()  # now 0-1

        scaling_factor = (max_disparity_range[0] + random.random() *
                          (max_disparity_range[1] - max_disparity_range[0]))
        disparity *= scaling_factor

        # now find disparity gradients and set to nearest - stop flying pixels
        # sharpen disparity
        edges = sobel(disparity) > 3
        disparity[edges] = 0
        mask = disparity > 0

        try:
            disparity = griddata(np.stack([self.ys[mask].ravel(), self.xs[mask].ravel()], 1),
                                    disparity[mask].ravel(), np.stack([self.ys.ravel(),
                                                                    self.xs.ravel()], 1),
                                    method='nearest').reshape(self.crop_height, self.process_width)
        except (ValueError, IndexError) as e:
            pass  # just return disparity

        return disparity

    def crop_all(self, inputs):
        # get crop parameters
        height, width, _ = np.array(inputs['left_image']).shape
        top = int(random.random() * (height - self.crop_height))
        left = int(random.random() * (width - self.process_width))
        right, bottom = left + self.process_width, top + self.crop_height

        mlist = ['left_image']
        if self.background_mode =='background':
            mlist = ['left_image', 'background']

        for key in mlist:
            inputs[key] = inputs[key].crop((left, top, right, bottom))
        inputs['disparity'] = inputs['disparity'][top:bottom, left:right]

        return inputs

    @staticmethod
    def resize_all(inputs, height, width, bg_mode):

        # images
        img_resizer = transforms.Resize(size=(height, width))

        mlist = ['left_image']
        if bg_mode =='background':
            mlist = ['left_image', 'background']

        for key in mlist:
            inputs[key] = img_resizer(inputs[key])
        # disparity - needs rescaling
        disp = inputs['disparity']
        disp *= width / disp.shape[1]

        disp = cv2.resize(disp.astype(float), (width, height))  # ensure disp is float32 for cv2
        inputs['disparity'] = disp

        return inputs    

    def prepare_sizes(self, inputs):
        height, width, _ = np.array(inputs['left_image']).shape
        if self.keep_aspect_ratio:
            if self.crop_height <= height and self.process_width <= width:
                # can simply crop the image
                target_height = height
                target_width = width

            else:
                # check the constraint
                current_ratio = height / width
                target_ratio = self.crop_height / self.process_width

                if current_ratio < target_ratio:
                    # height is the constraint
                    target_height = self.crop_height
                    target_width = int(self.crop_height / height * width)

                elif current_ratio > target_ratio:
                    # width is the constraint
                    target_height = int(self.process_width / width * height)
                    target_width = self.process_width

                else:
                    # ratio is the same - just resize
                    target_height = self.crop_height
                    target_width = self.process_width

        else:
            target_height = self.crop_height
            target_width = self.process_width

        inputs = self.resize_all(inputs, target_height, target_width, self.background_mode)

        # now do cropping
        if target_height == self.crop_height and target_width == self.process_width:
            # we are already at the correct size - no cropping
            pass
        else:
            self.crop_all(inputs)

        return inputs

    def augment_synthetic_image(self, image):
        if self.disable_synthetic_augmentation or \
           self.background_mode == 'inpaint':
            return Image.fromarray(image.astype(np.uint8))

        # add some noise to stereo image
        noise = np.random.randn(self.crop_height, self.process_width, 3) / 50
        image = np.clip(image / 255 + noise, 0, 1) * 255

        # add blurring
        if random.random() > 0.5:
            image = gaussian(image,
                             sigma=random.random(),
                             multichannel=True)

        image = np.clip(image, 0, 255)

        # color augmentation
        stereo_aug = transforms.ColorJitter.get_params(
            self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
            self.stereo_hue)

        image = stereo_aug(Image.fromarray(image.astype(np.uint8)))

        return image
    
    def augment_image(self, image):
        if self.is_training and np.random.rand() > 0.5:
            color_aug = transforms.ColorJitter.get_params(
                self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
                self.stereo_hue)

            image = color_aug(image)

        return image

    def __len__(self):
        return self.length

    def getitem(self, index=-1):
        if index == -1:
            index = np.random.randint(self.length, size=1)[0]

        left_image = pil_loader( os.path.join(self.dataset_path, self.image_names[index]))
        disparity  = read_pfm( os.path.join(self.dataset_path, self.dispairty_names[index]))

        ### 50% chance to flip the image left right
        if self.is_training and np.random.rand() > 0.5:
            left_image = left_image.transpose(Image.FLIP_LEFT_RIGHT)
            disparity  = disparity[:, ::-1]

        inputs = { 'left_image' : left_image,
                   'disparity'  : disparity }
        
        if self.background_mode == 'background':
            bg_image = np.random.choice(self.image_names, size=1)[0]
            inputs['background'] = pil_loader( os.path.join(self.dataset_path, bg_image))

        ## crop the image to crop size
        inputs = self.prepare_sizes(inputs)

        inputs['disparity'] = self.process_disparity(inputs['disparity'], max_disparity_range=(50, self.max_disparity))
        inputs['disparity'] = inputs['disparity'].astype(float)
        inputs['right_image'] = self.warp_image(inputs)

        ## Add some effects to right image
        inputs['right_image'] = self.augment_synthetic_image(inputs['right_image'])       ## Deal with

        # the reduce process width to crop width
        for key in ['left_image', 'right_image']:
            inputs[key] = inputs[key].crop((0, 0, self.crop_width, self.crop_height))
        inputs['disparity'] = inputs['disparity'][:, :self.crop_width] 

        ## convert to tensor .. 
        ## and normalise with ImageNet
        if self.background_mode == 'background':  inputs.pop('background')

        ## Add some effect to left img
        inputs['left_image']  = self.augment_image(inputs['left_image'])                  ## Leave this out is fine

        ## Normalise with ImageNet mean std.
        if self.background_mode == 'inpaint':
            inputs['right_image'] = np.array(inputs['right_image'])
        else:
            inputs['right_image'] = (self.to_tensor(inputs['right_image']) - 0.45) / 0.225
            
        inputs['left_image']  = (self.to_tensor(inputs['left_image'])  - 0.45) / 0.225
        # inputs['right_image'] = (self.to_tensor(inputs['right_image']) - 0.45) / 0.225    ## Do this one at training
        inputs['disparity']   = torch.from_numpy(inputs['disparity']).float()

        return inputs

    def __getitem__(self, index):
        inputs = self.getitem(index)
        return inputs
