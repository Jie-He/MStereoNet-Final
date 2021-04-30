print('Version', '1.2.5')
import time, copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from DatasetLoader.mscoco_dataset import MSCOCO_dataset 
from DatasetLoader.middlebury_dataset import MIDDLEBURY_dataset

from options import Options
from utils import interpolate_disparity, plot_tuple, \
                  plot_stereo_pair, plot_stereo_pair_tensor

import numpy as np
import os
from os import listdir
from os.path import isfile, join

from Network.MStereoNet import MSNet
from Network.MStereoNetNew import MSNetNew
from DatasetLoader.img_utils import pil_loader, write_pfm
from utils import tensor2dsp, tensor2img

OPTIONS = Options()
OPTIONS = OPTIONS.parse()

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print() 

### PATHS to read and write sample inferences
inference_path = 'stereo_images'
output_path    = 'stereo_images/out/'
save_path = 'models/weights'

## Read Image (CPU) -> (GPU)
def stereo_totensor(imgl_path, imgr_path):
    transformer = transforms.ToTensor() 
    imgl, imgr = (transformer(pil_loader(imgl_path)) - 0.45) / 0.225,\
                 (transformer(pil_loader(imgr_path)) - 0.45) / 0.225  

    imgl = imgl.unsqueeze(0).to(device)
    imgr = imgr.unsqueeze(0).to(device)
    return imgl, imgr

def save_depth_matplot(disparity, fname):
    n_cmap = copy.copy(plt.get_cmap('viridis'))
    n_cmap.set_bad(color='black')
    plt.imsave(fname, disparity, cmap=n_cmap)
    plt.close()

def run_inference(_net):
    l_path = join(inference_path, 'left' )
    r_path = join(inference_path, 'right')
    o_path = join(inference_path, 'out'  )

    for inx, fname in enumerate(listdir(l_path)):
        with torch.no_grad():
            left_name  = join( l_path, fname )
            right_name = join( r_path, fname )
            imgl, imgr = stereo_totensor(left_name, right_name)
            inf_time = time.time()
            disparity = _net(imgl, imgr)
            end_time = time.time() - inf_time
        del imgl
        del imgr
        torch.cuda.empty_cache()

        ## Could also calculate error
        disp_cpu = disparity.cpu().detach().numpy()[0]
        save_depth_matplot(disp_cpu, 
                           join(output_path, fname[:-3] + 'png'))
        ## Save the depth map as pfm
        np.save(join(output_path, fname[:-3] + 'npy'), disp_cpu.astype(np.float32))
        # print('done', fname, 'time:', end_time, 'fps:', 1/end_time)
        print('Post', f"{inx:08d}" , f'{fname: <25}' , '--' , f"time: {end_time:.5f}", 's', f"FPS: {1/end_time:.2f}")
   
def main():
    nfeature = OPTIONS.feature_size
    print('nfeature', nfeature)

    msnet = None
    if OPTIONS.network_variant == 'original':
        print('Using original network')
        msnet = MSNet(max_disp= OPTIONS.max_disparity, 
                    feature=nfeature).to(device)
    elif OPTIONS.network_variant == 'new':
        print('Using new network')
        msnet = MSNetNew(max_disp= OPTIONS.max_disparity, 
                    feature=nfeature).to(device)
    else:
        print('Network not found: ', OPTIONS.network_variant)
        exit()

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

    run_inference(msnet)

    print('Post: INF',)

if __name__ == '__main__':
    main()