import torch
import torch.nn as nn
import torch.nn.functional as F

from Network.submodules import convbn, convbn_3d, BasicBlock

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))

### From StereoNet
class StereoNetRefinement(nn.Module):
    def __init__(self):
        super(StereoNetRefinement, self).__init__()

        # Original StereoNet: left, disp
        features = 32
        self.conv = conv2d(4, features)

        self.dilation_list = [1, 4, 8, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:           ## Padding = dilation to prevent change it resolution
            self.dilated_blocks.append(BasicBlock(features, _stride=1, _dilation=dilation, _padding=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(features, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img=None):
        """Upsample low resolution disparity prediction to
        corresponding resolution as image size
        Args:
            low_disp: [B, H, W]
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]
        """
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        ## originally resize disparity to image size
        ## uses more GPU memory when using residue blocks
        ## resize image to 1/4 to save GPU memory
        b, c, h, w = low_disp.size()
        low_left = F.interpolate(left_img, size=(h, w), 
                                 mode='bilinear', align_corners=False)

        concat = torch.cat((low_disp, low_left), dim=1)  # [B, 4, H, W]
        out = self.conv(concat)
        out = self.dilated_blocks(out)
        residual_disp = self.final_conv(out)

        disp = F.relu(low_disp + residual_disp, inplace=True)  # [B, 1, H, W]

        disp = F.interpolate(disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
        disp = disp * scale_factor  # scale correspondingly
        disp = disp.squeeze(1)  # [B, H, W]

        return disp