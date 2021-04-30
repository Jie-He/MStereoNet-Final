import time 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from Network.StereoNetRefinement import StereoNetRefinement
from Network.submodules import convbn, convbn_3d, BasicBlock, DisparityEstimation

class MSNetNew(nn.Module):
    def __init__(self, max_disp=192, feature=32):
        super(MSNetNew, self).__init__()

        self.max_disp = max_disp
        self.feature = feature

        self.conv1 = convbn( 3, feature, 3, 3, 1, 1, True)
        self.conv2 = convbn(feature, feature, 3, 1, 1, 1, True)
        self.conv3 = convbn(feature, feature, 3, 1, 1, 1, True)

        ## ResNet Basic Block

        self.resblock1 = BasicBlock(feature)
        self.resblock2 = BasicBlock(feature)
        self.resblock3 = BasicBlock(feature)

        ## Pyramid Pooling,
        ## Stacking those after resizing to (downsampled)image size
        self.branch1 = nn.Sequential(nn.AvgPool2d((32,32), stride=(32,32)),
                                     convbn(feature, feature//2, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((16,16), stride=(16,16)),
                                     convbn(feature, feature//2, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((8,8), stride=(8,8)),
                                     convbn(feature, feature//2, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((4,4), stride=(4,4)),
                                     convbn(feature, feature//2, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        ##NOTE TO SELF 80 from feature//2 * 4 (4 pyramid pooling) + feature
        flast = feature//2 * 4 + feature

        self.lastConv = nn.Sequential(convbn(flast, feature * 2, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(feature * 2, feature, kernel_size=1, 
                                                padding=0, stride=1, 
                                                bias=False))

        self.disparity_estimate = DisparityEstimation(self.max_disp//4)

        self.refinement = StereoNetRefinement()

        print('Post 0')

    def forward(self, left_img, right_img):

        def feature_extraction(img):
            img = F.relu(self.conv1(img))
            img = F.relu(self.conv2(img))
            img = F.relu(self.conv3(img))

            img = self.resblock1(img)
            img = self.resblock2(img)
            img = self.resblock3(img)

            output_branch1 = self.branch1(img)
            output_branch1 = F.interpolate(output_branch1, (img.size()[2:]), 
                                           mode='bilinear', 
                                           align_corners=False)

            output_branch2 = self.branch2(img)
            output_branch2 = F.interpolate(output_branch2, (img.size()[2:]), 
                                           mode='bilinear', 
                                           align_corners=False)

            output_branch3 = self.branch3(img)
            output_branch3 = F.interpolate(output_branch3, (img.size()[2:]), 
                                           mode='bilinear', 
                                           align_corners=False)

            output_branch4 = self.branch4(img)
            output_branch4 = F.interpolate(output_branch4, (img.size()[2:]), 
                                           mode='bilinear', 
                                           align_corners=False)

            output_feature = torch.cat((img, output_branch4, output_branch3, 
                                        output_branch2, output_branch1), 1)
                                        
            output_feature = self.lastConv(output_feature)
            return output_feature

        left = feature_extraction(left_img)
        right= feature_extraction(right_img)
        ## Cost volume by correlation
        b, c, h, w = left.size()
        cost_volume = left.new_zeros(b, self.max_disp//4, h, w)

        cost_volume[:, 0, :, :] = (left * right).mean(dim=1)
        for i in range(1, self.max_disp//4):
            cost_volume[:, i, :, i:] = (left[:, :, :, i:] *
                                        right[:, :, :, :-i]).mean(dim=1)
        cost_volume = cost_volume.contiguous()

        disparity = self.disparity_estimate(cost_volume)
        disparity = torch.squeeze(disparity, 1)
        disparity = self.refinement(disparity, left_img, right_img)
    
        return disparity
