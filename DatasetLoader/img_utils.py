import numpy as np
import sys
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def read_pfm(file):
    with open(file, 'rb') as fh:
        fh.readline()
        width, height = str(fh.readline().rstrip())[2:-1].split()
        fh.readline()
        disp = np.fromfile(fh, '<f')
        return np.flipud(disp.reshape(int(height), int(width)))

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