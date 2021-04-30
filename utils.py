import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import copy, time

from DatasetLoader.img_utils import pil_loader

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor2img(tensor):
    image = tensor.permute(0, 2, 3, 1)
    return image.cpu().detach().numpy()

def tensor2dsp(tensor):
    # disp  = tensor.permute(0, 1, 2)
    return tensor.cpu().detach()

def interpolate_disparity(disp_map):
    plt.figure()

    n_cmap = copy.copy(plt.get_cmap('viridis'))
    n_cmap.set_bad(color='black')

    plt.imshow(disp_map, interpolation='bilinear', cmap=n_cmap)
    plt.colorbar()
    plt.show()

def plot_tuple(inputs):
    ## Convert back to image format
    if 'mask' in inputs:
        for mask in inputs['mask']:
            interpolate_disparity(mask.cpu().detach())

    n_cmap = copy.copy(plt.get_cmap('viridis'))
    n_cmap.set_bad(color='black')

    left, right, disp = inputs['left_image'], inputs['right_image'], inputs['disparity']

    left  = tensor2img( left) * 0.225 + 0.45
    right = tensor2img(right) * 0.225 + 0.45
    disp  = tensor2dsp( disp)

    for i in range(len(left)):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        axes[0].imshow(left[i],  interpolation='bilinear', cmap=n_cmap)
        axes[1].imshow(right[i], interpolation='bilinear', cmap=n_cmap)
        axes[2].imshow(disp[i],  interpolation='bilinear', cmap=n_cmap)
        fig.tight_layout()
        plt.show()

def plot_stereo_pair(imgl, imgr, displ, dispr):  
    plt.figure(figsize=(6,6))  
    plt.subplot(2,2,1)
    plt.grid(False)
    plt.imshow(imgl, cmap=plt.cm.binary)

    plt.subplot(2,2,2)
    plt.grid(False)
    plt.imshow(imgr, cmap=plt.cm.binary)

    n_cmap = copy.copy(plt.get_cmap('viridis'))
    n_cmap.set_bad(color='black')

    plt.subplot(2,2,3)
    plt.grid(False)
    plt.imshow(displ, interpolation='bilinear', cmap=n_cmap)

    plt.subplot(2,2,4)
    plt.grid(False)
    plt.imshow(dispr, interpolation='bilinear', cmap=n_cmap)

    plt.show()

def plot_stereo_pair_tensor(imgl, imgr, displ, dispr):  
    imgl = imgl.permute(0, 2, 3, 1)
    imgr = imgr.permute(0, 2, 3, 1)

    displ = displ.permute( 1, 2, 0)
    dispr = dispr.permute( 1, 2, 0)

    imgl = imgl.cpu().detach().numpy()[0]
    imgr=  imgr.cpu().detach().numpy()[0]
    displ = displ.cpu().detach()
    dispr = dispr.cpu().detach()

    plt.figure(figsize=(6,6))  
    plt.subplot(2,2,1)
    plt.grid(False)
    plt.imshow(imgl, cmap=plt.cm.binary)

    plt.subplot(2,2,2)
    plt.grid(False)
    plt.imshow(imgr, cmap=plt.cm.binary)

    n_cmap = copy.copy(plt.get_cmap('viridis'))
    n_cmap.set_bad(color='black')

    plt.subplot(2,2,3)
    plt.grid(False)
    plt.imshow(displ, interpolation='bilinear', cmap=n_cmap)

    plt.subplot(2,2,4)
    plt.grid(False)
    plt.imshow(dispr, interpolation='bilinear', cmap=n_cmap)

    plt.show()

def inference(_net, count, dimw=1024, dimh=786, isPrint=False, _crit=None):
    ## Check inference speed
    fps = np.zeros(count)
    errors = 0
    for i in range(count):
        left, right, disp = get_training_tuple(dimw, dimh)

        s = time.time()
        res = _net(left, right)
        if isPrint:
            e = time.time()-s
            print('Post X', e, 's', 1/e, 'FPS')
            fps[i] = 1/e
        
        if _crit is not None:
            loss = _crit(res, disp)
            errors += loss.item()

    if isPrint:
        plt.scatter(np.arange(count), fps)
        plt.show()

    return errors / count

def get_training_tuple(dimw, dimh):
    left  = torch.zeros(1, 3, dimw, dimh).cuda()
    right = torch.zeros(1, 3, dimw, dimh).cuda()

    disp = torch.zeros(1024,1024).cuda()

    ## randomize a few cubes
    for i in range(10):
        colorR, colorG, colorB = np.random.uniform(0.2, 1), np.random.uniform(0.2, 1), np.random.uniform(0.2, 1)
        disparity = np.random.randint(20, 192)
        coordx, coordy = np.random.randint(0, dimw-20), np.random.randint(0, dimh-20)
        width, height  = np.random.randint(10, dimw-coordx), np.random.randint(10, dimh-coordy)
        
        left[:, 0, coordy:(coordy+height), coordx:(coordx + width)] = colorR
        left[:, 1, coordy:(coordy+height), coordx:(coordx + width)] = colorG
        left[:, 2, coordy:(coordy+height), coordx:(coordx + width)] = colorB
        disp[coordy:(coordy+height), coordx:(coordx + width)] = disparity

        coordx = max(0, coordx - disparity)

        right[:, 0, coordy:(coordy+height), coordx:(coordx + width)] = colorR
        right[:, 1, coordy:(coordy+height), coordx:(coordx + width)] = colorG
        right[:, 2, coordy:(coordy+height), coordx:(coordx + width)] = colorB
    return left, right, disp

def plot_sinlge(_net, _train_data):
    inputs = _train_data.getitem() #get_training_tuple(dimension)
        
    left, right, disp = inputs['left_image'], inputs['right_image'], inputs['disparity']

    left = left.unsqueeze(0).to(device)
    right= right.unsqueeze(0).to(device)
    disp = disp.unsqueeze(0).to(device)

    left_image = left.permute(0, 2, 3, 1)
    right_image = right.permute(0, 2, 3, 1)

    pred = _net(left, right)

    plot_stereo_pair(left_image.cpu().detach().numpy()[0],
                     right_image.cpu().detach().numpy()[0],
                     disp.cpu().detach().numpy()[0],
                     pred.cpu().detach().numpy()[0]) #pred

    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    for ax, img in zip(axes.flat, (disp, pred)): # pred
        
        n_cmap = copy.copy(plt.get_cmap('viridis'))
        n_cmap.set_bad(color='black')

        im = ax.imshow(img.cpu().detach().numpy()[0], interpolation='bilinear', cmap=n_cmap)
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

def plot_sinlge_test(_net):
    transformer = transforms.ToTensor()
    left, right = (transformer(pil_loader('C:/Users/jiehe/Desktop/left.png'))) ,\
                  (transformer(pil_loader('C:/Users/jiehe/Desktop/right.png')))   #- 0.45) / 0.225
                  #- 0.45) / 0.225 ,\

    left = left.unsqueeze(0).to(device)
    right= right.unsqueeze(0).to(device)

    left_image = left.permute(0, 2, 3, 1)
    right_image = right.permute(0, 2, 3, 1)

    pred = _net(left, right)

    plot_stereo_pair(left_image.cpu().detach().numpy()[0],
                     right_image.cpu().detach().numpy()[0],
                     pred.cpu().detach().numpy(),
                     pred.cpu().detach().numpy()) #pred

    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    for ax, img in zip(axes.flat, (pred, pred)): # pred
        
        n_cmap = copy.copy(plt.get_cmap('viridis'))
        n_cmap.set_bad(color='black')

        im = ax.imshow(img.cpu().detach().numpy(), interpolation='bilinear', cmap=n_cmap)
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
