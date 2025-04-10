from torch import nn

import torch.nn.functional as F
import torch
from inter_modules.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from torch.nn.modules.pooling import AdaptiveMaxPool3d

#FIRST APPROACH

class ResidualBlock(nn.Module):
    """
    Res block, to facilitate feature fusion, perserves spatial resolution
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):       
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += x
        out = F.relu(out)
        return out

class MultiFramePooling(nn.Module):
    """
    Corresponding Feature across frames are maxpooled from multi frames
    """
    def __init__(self, target_size): 

        super(MultiFramePooling, self).__init__() 

        self.target_size = target_size
        self.adaptive_max_pool = AdaptiveMaxPool3d(output_size=self.target_size)

    def forward(self, views):  
        stacked_views = torch.stack(views, dim=2)
        pooled = self.adaptive_max_pool(stacked_views)       # pooled has shape = output_size
        pooled = pooled.squeeze(dim=2)                       # getting rid of the singelton dimension        
        return pooled  
