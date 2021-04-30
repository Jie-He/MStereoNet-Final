import time 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np

## From PSMnet
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, use_bn=True):
    if use_bn:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                             nn.BatchNorm2d(out_planes))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False)

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, use_bn=True):
    if use_bn:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                             nn.BatchNorm3d(out_planes))
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,
                         bias=False)

class BasicBlock(nn.Module):
    def __init__(self, channel_num, _stride=1, _dilation=1, _padding=1):
        super(BasicBlock, self).__init__()
    
        self.conv_block1 = nn.Sequential(  
            nn.Conv2d(channel_num, channel_num, 3, padding=_padding, stride=_stride, dilation=_dilation),
            nn.BatchNorm2d(channel_num))
    
        self.stride = _stride

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=_padding, dilation=_dilation),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        ## Use holder (more memory) if stride is not 1
        if self.stride != 1:
            x = self.conv_block1(x)
            xs = self.relu(x) 
            xs = x + self.conv_block2(xs)
            xs = self.relu(xs)
            return xs
        
        xs = self.conv_block1(x)
        xs = self.relu(xs) 
        xs = self.conv_block2(xs) + x
        xs = self.relu(xs)
        return xs

class DisparityEstimation(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimation, self).__init__()

        self.max_disp = max_disp
        self.match_similarity = match_similarity
        # self.candidate = None
        self.disp = torch.Tensor(np.reshape(np.array(range(max_disp)),[1, max_disp,1,1])).cuda()

    def forward(self, cost_volume):
        assert cost_volume.dim() == 4

        # Matching similarity or matching cost
        cost_volume = cost_volume if self.match_similarity else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]

        disp = torch.sum(prob_volume * self.disp.data, 1, keepdim=True)

        return disp

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask