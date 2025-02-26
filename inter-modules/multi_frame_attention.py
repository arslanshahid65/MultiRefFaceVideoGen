from torch import nn

import torch.nn.functional as F
import torch

class ChannelNorm(nn.Module):
    '''
    Pixelwise Temporal Normamlization similar to ChannelNorm in: https://doi.org/10.48550/arXiv.2006.09965
    '''
    def __init__(self, eps=1e-5):
        super(ChannelNorm, self).__init__()
        # Learnable per channel (in this case per temporal dimension) scale (alpha) and shift (beta) parameters
        self.alpha = nn.Parameter(torch.ones(1,768,1,1))
        self.beta = nn.Parameter(torch.zeros(1,768,1,1))

        self.eps = eps
    def forward(self, x):

        # Compute mean and standard deviation across C*T dimension
        batch_size = x.size(0)
        num_channels= x.size(1)
        temporal = x.size(2)
        height = x.size(3)
        width = x.size(4)
        # (N, C*T, H, W)
        x = x.contiguous()
        x = x.view(batch_size, num_channels*temporal, height, width)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        # Pixel-wise normalization
        x_normalized = (x - mean) / (torch.sqrt(var + self.eps))   # std + self.eps?

        # Apply learnable scale and shift
        x_normalized = (x_normalized * self.alpha) + self.beta
        #x_normalized = x_normalized.contiguous()

        x_normalized = x_normalized.view(batch_size, num_channels, temporal, height, width)
        return x_normalized

class NLBlockND(nn.Module):
    """
        Implementation of Non-Local Block (https://github.com/tea1528/Non-Local-NN-Pytorch) with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
    """

    def __init__(self, in_channels, inter_channels=None,
                 dimension=2, bn_layer=True):
        """
            Non local block with embedded gaussian version to achieve cross frame attention from multi frame warpings
        """        
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(1, 1))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
            
    def forward(self, x, num_ref_frames):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x.size(0)
        num_channels= x.size(1)
        temporal = x.size(2)
        height = x.size(3)
        width = x.size(4)
        # (N, C, T, HW)
        
        # this reshaping and permutation is for the pixelwise non-local temporal cross attention as opposed to spacio-temporal function in original implementation
        x1 = x.permute(0,2,1,3,4).contiguous()
        x1 = x1.view(batch_size * temporal, num_channels, height, width)        
        g_x = self.g(x1)
        g_x = g_x.view(batch_size, temporal, self.inter_channels, height, width)
        g_x = g_x.permute(0, 2, 1, 3, 4)
        g_x = g_x.view(batch_size, self.inter_channels, num_ref_frames, -1)
        g_x = g_x.permute(0, 1, 3, 2)

        theta_x = self.theta(x1)
        theta_x = theta_x.view(batch_size, temporal, self.inter_channels, height, width)
        theta_x = theta_x.permute(0, 2, 1, 3, 4)
        theta_x = theta_x.view(batch_size, self.inter_channels, num_ref_frames, -1)


        phi_x = self.phi(x1)
        phi_x = phi_x.view(batch_size, temporal, self.inter_channels, height, width)
        phi_x = phi_x.permute(0, 2, 1, 3, 4)
        phi_x = phi_x.view(batch_size, self.inter_channels, num_ref_frames, -1)

        theta_x = theta_x.permute(0, 1, 3, 2)

        f = torch.matmul(theta_x, phi_x)
                
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 1, 3, 2).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])


        y1 = y.permute(0,2,1,3,4).contiguous()
        y1 = y1.view(batch_size * temporal, self.inter_channels, height, width)  
        W_y = self.W_z(y1)

        W_y = W_y.view(batch_size, temporal, self.in_channels, height, width)
        W_y = W_y.permute(0, 2, 1, 3, 4)
        # residual connection
        z = W_y + x
        return z