# Run Midas on a set folder and produce the 
# depth .pfm to a folder
import sys
import re
import os
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt
import urllib.request
import copy
import time
from PIL import Image

extension = 'pfm'

def read_image(path):
    img = cv2.imread(path)
 
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img
 
def write_depth(path, depth, bits=1):

    def write_pfm(path, image, scale=1):
 
        with open(path, "wb") as file:
            color = None
    
            if image.dtype.name != "float32":
                raise Exception("Image dtype must be float32.")
    
            image = np.flipud(image)
    
            if len(image.shape) == 3 and image.shape[2] == 3:  # color image
                color = True
            elif (
                len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
            ):  # greyscale
                color = False
            else:
                raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")
    
            file.write("PF\n" if color else "Pf\n".encode())
            file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))
    
            endian = image.dtype.byteorder
    
            if endian == "<" or endian == "=" and sys.byteorder == "little":
                scale = -scale
    
            file.write("%f\n".encode() % scale)
    
            image.tofile(file)
    
    def write_npy(path, image, scale=1):
        np.save(path, image)

    if extension == 'npy':  write_npy(path + ".npy", depth.astype(np.float32))
    else: write_pfm(path + ".pfm", depth.astype(np.float32))

def read_depth(path):
    def read_pfm(file):
        with open(file, 'rb') as fh:
            fh.readline()
            width, height = str(fh.readline().rstrip())[2:-1].split()
            fh.readline()
            disp = np.fromfile(fh, '<f')
            return np.flipud(disp.reshape(int(height), int(width)))

    def read_npy(path):
        disparity = np.load(path)
        return disparity
    # return read_npy(path)


    if extension == 'npy': return read_npy(path)
    return read_pfm(path)

## Load the midas network from torch
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
device = torch.device("cuda") if torch.cuda.is_available() \
                              else torch.device("cpu")
midas.to(device)
midas.eval()

## Use the normal size network pretrained weights
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.default_transform

image_path = '/mnt/seaweed/homes/jh2699/Project/MDataset/coco_img'
save_path  = '/mnt/seaweed/homes/jh2699/Project/MDataset/coco_depth'

## Check a folder and display a random pair of data (image and depth)
## Folder Size Check
def do_folder_check(folder_number):
    def interpolate_disparity(disp_map):
        plt.figure()

        n_cmap = copy.copy(plt.get_cmap('viridis'))
        n_cmap.set_bad(color='black')

        plt.imshow(disp_map, interpolation='bilinear', cmap=n_cmap)
        plt.colorbar()
        plt.show()

    dest = 'subfolder' + str(folder_number)

    full_path1= os.path.join(image_path,dest)
    full_path = os.path.join(save_path, dest)
    print('checking full_path', full_path)

    mlist = os.listdir(full_path) # dir is your directory path
    print( 'has files size:', len(mlist) )
    #random name

    avg = 0
    for c in mlist:
        print('checking', c)

        image     = read_image(os.path.join(full_path1, c[:-3] + 'jpg'))
        # read_pfm_time = time.time()
        disparity = read_depth(  os.path.join(full_path , c) )
        # difference = time.time() - read_pfm_time
        # print("Reading took", difference)
    
    # print("Average Reading Time:", avg/len(mlist), 's')
    interpolate_disparity(image)
    interpolate_disparity(disparity)

## Actually run midas on a folder of images and save the depth
def do_folder(folder_number):
  dest = 'subfolder' + str(folder_number)

  try:
    os.mkdir( os.path.join(save_path, dest) )
  except Exception: pass

  print('\n Doing Folder Number', folder_number, '\n')

  print('doing: ', os.path.join(image_path, dest) )
  print('save to:', os.path.join(save_path, dest) )
  ind = 0

  epoch_time = time.time()
  for fname in os.listdir( os.path.join(image_path, dest) ):
    source = os.path.join(image_path, dest, fname)
    print(folder_number, 'Progress', ind)
    ind += 1

    img = cv2.imread(source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    with torch.no_grad():
      prediction = midas(input_batch)

      prediction = (torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,).squeeze().cpu().numpy())
      
    output = prediction
    filename = os.path.join(save_path, dest, os.path.splitext(os.path.basename(source))[0])
    # print(filename)
    write_depth(filename, prediction, bits=2)

  print('took:', time.time() - epoch_time)

begin = 3
endin = 13

#for i in range(begin, endin):
#  do_folder(i)

for i in range(begin, endin):
  do_folder_check(i)

torch.cuda.empty_cache()

print('Done')
