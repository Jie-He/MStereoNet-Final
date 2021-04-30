VERSION='4.2.5'

import time, copy, os, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from options import Options
from utils import *

import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage.filters import gaussian

from DatasetLoader.mscoco_dataset import MSCOCO_dataset 
from DatasetLoader.middlebury_dataset import MIDDLEBURY_dataset
from my_sampler import MySampler

from Network.MStereoNet    import MSNet
from Network.MStereoNetNew import MSNetNew

from DatasetLoader.img_utils import pil_loader
from Inpaint_network.InpaintManager import InpaintManager

from dataset_paths import data_paths

OPTIONS = Options()
OPTIONS = OPTIONS.parse()

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_path = 'models/weights'

# data_paths = {'mscoco':'/mnt/seaweed/homes/jh2699/Project/MDataset',
#               'middlebury': '/media/xps/Toshiba/StereoTrainingSet/Middlebury'}

## For testing
data_paths = {'mscoco':'C:/Users/jiehe/Desktop/coco_test',
              'kitti2015': '/home/xps/Desktop/KITTI_2015'}

## Cannot parallel the inpainter, so using this instead
## and post process the image during training while 4 workers prep the image
class StereoImagePostProcess():
    def __init__(self, painter, height, width):
        self.painter = painter
        self.to_tensor = transforms.ToTensor()
        self.crop_height = height
        self.crop_width  = width
        try:
            self.stereo_brightness = (0.8, 1.2)
            self.stereo_contrast = (0.8, 1.2)
            self.stereo_saturation = (0.8, 1.2)
            self.stereo_hue = (-0.01, 0.01)
            transforms.ColorJitter.get_params(
                self.stereo_brightness, self.stereo_contrast, 
                self.stereo_saturation, self.stereo_hue)
        except TypeError:
            self.stereo_brightness = 0.2
            self.stereo_contrast = 0.2
            self.stereo_saturation = 0.2
            self.stereo_hue = 0.01
        print('POST [-]: Inpaint Post Processor Initialised!')

    def postprocess(self, image, mask):
        inference = self.painter.inference(image, mask)
        inference = (tensor2img(inference)[0] * 255).astype(int)
        image = np.where(mask == 0, image, inference)
        ## Apply augmentation, transformation and normalisation
        # add some noise to stereo image
        noise = np.random.randn(self.crop_height, self.crop_width, 3) / 50
        image = np.clip(image / 255 + noise, 0, 1) * 255

        # add blurring
        if random.random() > 0.5:
            image = gaussian(image,
                             sigma=random.random(),
                             multichannel=True)

        image = np.clip(image, 0, 255)

        # color augmentation
        stereo_aug = transforms.ColorJitter.get_params(
            self.stereo_brightness, self.stereo_contrast, 
            self.stereo_saturation, self.stereo_hue)

        image = stereo_aug(Image.fromarray(image.astype(np.uint8)))
        image = (self.to_tensor(image) - 0.45) / 0.225    ## Do this one at training
        return image

def save_checkpoint(_current, _cumulative_loss, _net, _optimizer, _loss, _indx=0):
    _cumulative_loss /= OPTIONS.save_freq
    print('\nAverage Loss: ', _cumulative_loss)

    _loss = np.append(_loss, _cumulative_loss)#validate_loss
    save_loss = 0
    torch.save( {'epoch' : _current,
                'state_dict' : _net.state_dict(),
                'optimizer' : _optimizer.state_dict(),
                'loss' : _loss,
                'sampler_index' : _current},
                join(save_path, f"{_current:08d}" + '.vodka'))

                # 'data_index': train_data.index}, 
    print('Saved weight: ', join(save_path, f"{_current:08d}" + '.vodka'), '\n')
    return _loss

def train(_current, _epochs, _net, _optimizer, _criterion, _loss, train_data, processor=None):
    current = _current
    done_train = False
    if (_current >= OPTIONS.training_steps): done_train = True

    save_loss = 0
    while (not done_train):
        for batch_idx, inputs in enumerate(train_data):
            current += 1
            s = time.time()

            left, right, disp = inputs['left_image'], inputs['right_image'], inputs['disparity']

            if OPTIONS.fill_mode =='inpaint':
                masks = inputs['mask'].cpu().detach().numpy()
                images = right.cpu().detach().numpy()
                right = torch.zeros_like(left)
                for i in range(len(right)):
                    right[i] = processor.postprocess(images[i], masks[i])
            # testing 
            # test_inputs = { 'left_image' : left, 'right_image' : right, 'disparity' : disp, 'mask': inputs['mask'] }
            # plot_tuple(test_inputs)
            
            left = left.to(device)
            right= right.to(device)
            disp = disp.to(device)

            _optimizer.zero_grad()
            res = _net(left, right)
            loss = _criterion(res, disp, reduction='mean')#_criterion(disp, res)
            loss.backward()
            _optimizer.step()
            e = time.time() - s
            print('Post', f"{current:08d}" , '--' , f"{e:.5f}", 's', 'loss:', f"{loss.item():.2f}")
            save_loss += loss.item()
            

            if (current % OPTIONS.save_freq == 0):
                _loss = save_checkpoint(current, save_loss, 
                                        _net, _optimizer, 
                                        _loss, batch_idx)
                save_loss = 0

            if (current >= OPTIONS.training_steps):
                done_train = True
                break

    plt.scatter(np.arange( len(_loss) ), _loss)
    plt.show()

    return _net

def quick_check(_net, _train_set):
    inference(_net, 20, dimw=608, dimh=320, isPrint=True)
    inference(_net, 20, dimw=1280, dimh=720, isPrint=True)

    for _ in range(5):
        _train_set.index = np.random.randint(_train_set.length, size=1)[0]
        plot_sinlge(_net, _train_set)

    try: plot_sinlge_test(_net)
    except Exception: print('Din Don!')

def main():
    print('Version', VERSION)
    print('Using device:', device)
    print()     


    msnet = None
    if OPTIONS.network_variant == 'original':
        print('Using original network')
        msnet = MSNet(max_disp= OPTIONS.max_disparity, 
                    feature=OPTIONS.feature_size).to(device)
    elif OPTIONS.network_variant == 'new':
        print('Using new network')
        msnet = MSNetNew(max_disp= OPTIONS.max_disparity, 
                    feature=OPTIONS.feature_size).to(device)
    else:
        print('Network not found: ', OPTIONS.network_variant)
        exit()

    optimizer = optim.Adam(msnet.parameters(), lr = 0.001, betas=(0.9, 0.999))
    criterion = F.smooth_l1_loss 
    loss = np.array([])
    current_epoch = 0
    sampler_index = 0
    epochs = OPTIONS.training_steps
    print('Estimator, Max_disp', OPTIONS.max_disparity)
    print('Epoch Aim:', epochs)
    print('Save Freq:', OPTIONS.save_freq)
    print('Batch Size:', OPTIONS.batch_size)
    print('Feature Size:', OPTIONS.feature_size)
    print('Painting Method:', OPTIONS.fill_mode)

    ## Create the save folder
    global save_path
    save_path = save_path + '_' + OPTIONS.model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    points = [f for f in listdir(save_path) if isfile(join(save_path, f))]
    points.sort()

    ## load the last weights   
    if (len(points) > 0):
        print('loading', join(save_path, points[-1]))
        check_point = torch.load(join(save_path, points[-1]))
        
        current_epoch = check_point['epoch']
        print('trained', current_epoch, 'epoch')
        msnet.load_state_dict(check_point['state_dict'])
        optimizer.load_state_dict(check_point['optimizer'])
        loss = check_point['loss']
        sampler_index = check_point['sampler_index']

    painter = None
    postprocessor = None
    nworker = 4

    if OPTIONS.fill_mode == "inpaint":
        # nworker = 1
        painter = InpaintManager(OPTIONS.width, OPTIONS.height)
        postprocessor = StereoImagePostProcess(painter, OPTIONS.height, OPTIONS.width)

    ## Initialise datasets
    train_set = MSCOCO_dataset(data_paths[OPTIONS.training_dataset],
                               OPTIONS.max_disparity,
                               OPTIONS.height, OPTIONS.width,
                               background_mode = OPTIONS.fill_mode)
    
    train_sampler = MySampler(train_set, sampler_index, batch_size=OPTIONS.batch_size)

    TrainImgLoader = torch.utils.data.DataLoader(train_set, 
                                                 batch_size=OPTIONS.batch_size,
                                                 sampler=train_sampler,
                                                 num_workers=nworker, 
                                                 drop_last=False,
                                                 pin_memory=True)

    # Train it
    msnet.train()
    msnet = train(current_epoch, epochs, msnet, optimizer, criterion, loss,
                   TrainImgLoader, postprocessor)

    # quick_check(msnet, train_set)

def test():
    painter = InpaintManager(OPTIONS.width, OPTIONS.height)
    postprocessor = StereoImagePostProcess(painter, OPTIONS.height, OPTIONS.width)
      ## Initialise datasets
    train_set = MSCOCO_dataset(data_paths[OPTIONS.training_dataset],
                               OPTIONS.max_disparity,
                               OPTIONS.height, OPTIONS.width,
                               background_mode = OPTIONS.fill_mode,
                               is_training=False)
    
    TrainImgLoader = torch.utils.data.DataLoader(train_set, 
                                                 batch_size=OPTIONS.batch_size,
                                                 shuffle=False,
                                                 num_workers=0, drop_last=False,
                                                 pin_memory=True)
    
    limit = 10
    random.seed(110598)
    for ind, inputs in enumerate(TrainImgLoader):
        if OPTIONS.fill_mode=='inpaint':
            left, right, disp = inputs['left_image'], inputs['right_image'], inputs['disparity']

            if OPTIONS.fill_mode =='inpaint':
                masks = inputs['mask'].cpu().detach().numpy()
                images = right.cpu().detach().numpy()
                right = torch.zeros_like(left)

                for i in range(len(right)):
                    right[i] = postprocessor.postprocess(images[i], masks[i])

            ## testing 
            test_inputs = { 'left_image' : left, 'right_image' : right, 'disparity' : disp, 'mask': inputs['mask'] }
            plot_tuple(test_inputs)
        else:
            plot_tuple(inputs)
        # plt.imshow(tensor2img(inputs['right_image']) * .225 + 0.45)
        # plt.savefig(OPTIONS.fill_mode + ".png",bbox_inches='tight')
        # plt.imshow(inputs['mask'].cpu().detach()[0])
        # plt.savefig('disparity_mask.png',bbox_inches='tight') 
        random.seed(110598)
        limit -=1
        if limit <= 0: break

if __name__ == '__main__':
    # test()
    main()
