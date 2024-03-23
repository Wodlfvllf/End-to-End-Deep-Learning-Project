import torch
import torch.nn as nn
import torch
import math
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from utils_2 import *
from dataset_2 import *

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        
        # Number of input channels to image
        self.gate_channels = gate_channels
        
        #MLP layer
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

    def forward(self, x):
        
        #Avg_pool
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_avg = self.mlp( avg_pool )

        #max_pool
        max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_max = self.mlp( max_pool )

        #Element wise sum
        channel_att_sum = channel_att_max + channel_att_avg

        #scaling output of channel attention to match dimensions with input
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        
    def forward(self, x):
        
        # Applying Average Pooling and Maxpooling layer and concatenating
        x_compress = self.compress(x)
        
        # Applying Convolution operation on concatenated inputs
        x_out = self.spatial(x_compress)
        
        # Applying Sigmoid to attention mask
        scale = F.sigmoid(x_out) # broadcasting
        
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=1, no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.no_spatial=no_spatial
        
        if not no_spatial:
            self.SpatialGate = SpatialGate()
            
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class VGG_12(nn.Module):
    def __init__(self, BasicBlock, in_channels, out_channels, CBAM):
        super(VGG_12, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_1 = self.__make_layer(BasicBlock, change = 'yes')
        self.cbam_1 = self.channel_block_attention(CBAM)
        self.block_2 = self.__make_layer(BasicBlock, change = 'yes')
        self.cbam_2 = self.channel_block_attention(CBAM)
        self.block_3 = self.__make_layer(BasicBlock, change = 'yes')
        self.cbam_3 = self.channel_block_attention(CBAM)
        self.block_4 = self.__make_layer(BasicBlock, change = 'NO')
        self.cbam_4 = self.channel_block_attention(CBAM)
        self.block_5 = self.__make_layer(BasicBlock, change = 'NO')
        self.fc_1 = nn.Linear(in_features=128, out_features=64)  # Adjust in_features according to the output shape after blocks
#         self.fc_2 = nn.Linear(in_features=512, out_features=64)
        self.fc_3 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.2)
        self.avg_pool = nn.AvgPool2d(kernel_size = 3)
        
    def __make_layer(self, block, change):
        op =  block(in_channels=self.in_channels, out_channels=self.out_channels)
        self.in_channels = self.out_channels
        if change == 'yes':
            self.out_channels = self.out_channels*2
        return op
    
    def channel_block_attention(self, cbam):
        return cbam(self.in_channels)
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.cbam_1(x)
#         x = self.relu(x)
        
        x = self.block_2(x)
        x = self.cbam_2(x)
#         x = self.relu(x)
        
        x = self.block_3(x)
        x = self.cbam_3(x)
#         x = self.relu(x)
        
        x = self.block_4(x)
        x = self.cbam_4(x)
#         x = self.relu(x)
        
        x = self.block_5(x)
#         x = self.relu(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
#         x = self.dropout(x)
        x = self.relu(x)
        
        x = self.fc_3(x)
        x = self.sigmoid(x)
        return x