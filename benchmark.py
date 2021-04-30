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
from DatasetLoader.Kitti_dataset import KITTI_dataset 
from DatasetLoader.middlebury_dataset import MIDDLEBURY_dataset

from options import Options
from utils import interpolate_disparity, plot_tuple, \
                  plot_stereo_pair, plot_stereo_pair_tensor

import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle

from Network.MStereoNet import MSNet
from Network.MStereoNetNew import MSNetNew
from DatasetLoader.img_utils import pil_loader, write_pfm
from utils import tensor2dsp, tensor2img

from dataset_paths import data_paths

import time as time

OPTIONS = Options()
OPTIONS = OPTIONS.parse()

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print() 

log_path    = 'models/logs'
save_path   = 'models/weights'

## Add model name to it as well
write_path = 'D:/StereoTrainingSet/EPE/'

write_error_image = OPTIONS.save_error_image

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

def compute_errors(prediction, ground_truth):
    mask = (ground_truth > 0) & (ground_truth <=192)

    abs_difference = np.abs(ground_truth[mask] - prediction[mask])
    EPE = abs_difference.mean()

    d1 = (abs_difference >= 1).sum() / mask.sum()
    d2 = (abs_difference >= 2).sum() / mask.sum()
    d3 = (abs_difference >= 3).sum() / mask.sum()

    error_image = None
    if write_error_image:
        error_image = np.zeros_like(ground_truth)
        error_image[mask] = abs_difference 

    return d1, d2, d3, EPE, mask, error_image

    # # unormalise
    # inputs['left_image']  = inputs['left_image']  * 0.255 + 0.45
    # inputs['right_image'] = inputs['right_image'] * 0.255 + 0.45

    # plot_stereo_pair_tensor(inputs['left_image'],
    #                         inputs['right_image'],
    #                         inputs['disparity'].to(device),
    #                         prediction)

def evaluate_from_loader(_net, _loader, printing, fname='default'):
    errors = {'d1':[], 'd2':[], 'd3':[], 'EPE':[]}
    image_write_to = os.path.join(write_path, fname + '-' + OPTIONS.model_name)
    print(image_write_to)
    if not os.path.exists(image_write_to):
        os.makedirs(image_write_to)

    inference_time = 0

    for inx, inputs in enumerate(_loader): ## Should be batch size 1
        left = inputs['left_image'].to(device)
        right= inputs['right_image'].to(device)

        stime = time.time()
        prediction = _net(left, right)
        inference_time += (time.time() - stime)

        prediction = prediction.cpu().detach().numpy()
        inputs['disparity'] = inputs['disparity'].cpu().detach().numpy()
        nd1, nd2, nd3, nEPE, mask, abs_error = compute_errors(prediction, inputs['disparity'])

        if write_error_image: ## if required
            iw2 = os.path.join(image_write_to, str(inx).zfill(3))
            save_depth_matplot(prediction[0], iw2+'-0.png')
            save_depth_matplot(abs_error[0], iw2+'-1.png')
            if fname == 'middle':
                mask = inputs['disparity'] > 192
                prediction[mask] = 0
                save_depth_matplot(prediction[0], iw2+'-2.png')

        errors['d1'].append(nd1)
        errors['d2'].append(nd2)
        errors['d3'].append(nd3)
        errors['EPE'].append(nEPE)
        if printing:
            print('Index: ', inx, ': ', nd1, nd2, nd3, nEPE)
            
    inference_time = inference_time / len(errors['d1'])
    errors['d1'] = [ np.mean(errors['d1']),  np.std(errors['d1']) ]
    errors['d2'] = [ np.mean(errors['d2']),  np.std(errors['d2']) ]
    errors['d3'] = [ np.mean(errors['d3']),  np.std(errors['d3']) ]
    errors['EPE']= [ np.mean(errors['EPE']), np.std(errors['EPE'])]

    print('AVERAGE: ', errors['d1'], errors['d2'], errors['d3'], errors['EPE'],
          'Inference Time:', inference_time, 'FPS:', 1.0/inference_time)
    return errors

def evaluate_kitti(_net, name, printing=True):
    is_12 = False
    if name == "kitti12" : is_12 = True
    kitti_dataset = KITTI_dataset(data_paths[name], is_12)
    kitti_loader  = torch.utils.data.DataLoader(kitti_dataset, 
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0, drop_last=False,
                                                pin_memory=True)

    return evaluate_from_loader(_net, kitti_loader, printing, fname=name)

def evaluate_middlebury(_net, printing=True):
    middle_dataset = MIDDLEBURY_dataset(data_paths['middlebury'])
    middle_loader  = torch.utils.data.DataLoader(middle_dataset, 
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0, drop_last=False,
                                                pin_memory=True)
    
    return evaluate_from_loader(_net, middle_loader, printing, fname='middle')

def eval(_net, printing=True):
    combine_errors = {}
    with torch.no_grad():
        combine_errors['2012'] = evaluate_kitti(_net, 'kitti15', printing)
        # input("Press Enter to continue...")
        combine_errors['2015'] = evaluate_kitti(_net, 'kitti12', printing)
        # input("Press Enter to continue...")
        combine_errors['MIDL'] = evaluate_middlebury(_net, printing)
    return combine_errors

def plot_errors(cberrors):
    c = len(cberrors)
    keys = ['d1', 'd2', 'd3', 'EPE']
    kitti12 = {'d1':[], 'd2':[], 'd3':[], 'EPE':[]}
    kitti15 = {'d1':[], 'd2':[], 'd3':[], 'EPE':[]}
    middleb = {'d1':[], 'd2':[], 'd3':[], 'EPE':[]}

    for cerror in cberrors:
        for k in keys:
            kitti12[k].append(cerror['2012'][k])
            kitti15[k].append(cerror['2015'][k])
            middleb[k].append(cerror['MIDL'][k])

    for k in keys:
        kitti12[k] = np.array(kitti12[k])
        kitti15[k] = np.array(kitti15[k])
        middleb[k] = np.array(middleb[k])

    xrange = (np.arange(c) + 1) * 5000
    for k in keys:
        ## Plot mean and std(as error bar, maybe not)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

        axes[0].plot(xrange, kitti12[k][:,0], '--yo', label='kitti12')
        axes[0].plot(xrange, kitti15[k][:,0], '--bo', label='kitti15')
        axes[1].plot(xrange, middleb[k][:,0], '--ro', label='middleb')
        # fig.tight_layout()

        axes[0].set_title("KITTI 2012 & 2015 " + k + " loss")
        axes[1].set_title("Middlebury "        + k + " loss")
        for a in axes:
            a.grid()
            a.legend()
            a.set_xlabel('steps trained')
            a.set_ylabel( k +' loss')
        plt.show()

def main():
    nfeature = OPTIONS.feature_size
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
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    points = [f for f in listdir(save_path) if isfile(join(save_path, f))]
    points.sort()

    if OPTIONS.quick_bench == 1:
        chkpoint = points[-1]
        check_point = torch.load(join(save_path, chkpoint))
        print('Loading', join(save_path, chkpoint))
        print('Trained', check_point['epoch'], 'epochs')
        msnet.load_state_dict(check_point['state_dict'])
        eval(msnet, printing=False)
        exit()

    epoch_errors = []

    ## Make the log path
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    error_log = join(log_path, (OPTIONS.error_log + '.pickle'))
    print("Error_log", error_log)

    print('')
    ## Load if the file exists
    if os.path.exists(error_log):
        with open(error_log, 'rb') as handle:
            epoch_errors = pickle.load(handle)
        print('Loading existing log with', len(epoch_errors), 'entires.')

    ## Append if any new weights
    print('Appending', len( points[len(epoch_errors) : ]), 'logs')
    print('Total of' , len( points ), 'weights')
    print('')

    for chkpoint in points[ len(epoch_errors): ]:
        check_point = torch.load(join(save_path, chkpoint))
        print('Loading', join(save_path, chkpoint))
        print('Trained', check_point['epoch'], 'epochs')
        msnet.load_state_dict(check_point['state_dict'])
        epoch_errors.append(eval(msnet, printing=False))

        ## save the log with pickle
        with open(error_log, 'wb') as handle:
            print('Saving...')
            pickle.dump(epoch_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_errors(epoch_errors)
    print('Post: INF',)

## Test code
def test():
    epoch_errors = [ {'2012': {'d1': 0.9, 'd2': 0.5, 'd3': 0.1, 'EPE':9},
                      '2015': {'d1': 0.8, 'd2': 0.3, 'd3': 0.1, 'EPE':8},
                      'MIDL': {'d1': 0.9, 'd2': 0.5, 'd3': 0.2, 'EPE':7},},

                      {'2012': {'d1': 0.4, 'd2': 0.2, 'd3': 0.1, 'EPE':6},
                       '2015': {'d1': 0.4, 'd2': 0.1, 'd3': 0.1, 'EPE':5},
                       'MIDL': {'d1': 0.4, 'd2': 0.2, 'd3': 0.2, 'EPE':4},},

                      {'2012': {'d1': 0.2, 'd2': 0.2, 'd3': 0.1, 'EPE':3},
                       '2015': {'d1': 0.1, 'd2': 0.1, 'd3': 0.1, 'EPE':2},
                       'MIDL': {'d1': 0.1, 'd2': 0.2, 'd3': 0.2, 'EPE':1},}]
    # plot_errors(epoch_errors)

        # if not os.path.exists(error_log):
    #     for chkpoint in points:
    #         check_point = torch.load(join(save_path, chkpoint))
    #         print('Loading', join(save_path, chkpoint))
    #         print('Trained', check_point['epoch'], 'epochs')
    #         msnet.load_state_dict(check_point['state_dict'])
    #         epoch_errors.append(eval(msnet, printing=False))

    #         #Save it with pickle
    #     with open(error_log, 'wb') as handle:
    #         pickle.dump(epoch_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     print('loading instead!')
    #     with open(error_log, 'rb') as handle:
    #         epoch_errors = pickle.load(handle)

if __name__ == '__main__':
    main()
    # test()