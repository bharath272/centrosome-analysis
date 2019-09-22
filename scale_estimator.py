import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class ScaleEstimator(nn.Module):
    def __init__(self, in_channels=3,ksize=5,hidden_channels=10, num_layers = 3):
        super(ScaleEstimator,self).__init__()
        conv_layers = []
        inc = in_channels
        for i in range(num_layers):
            conv_layers.append(nn.Conv2d(inc, hidden_channels, ksize, stride=2))
            conv_layers.append(nn.ReLU())
            inc = hidden_channels
        self.conv = nn.Sequential(*conv_layers)
        self.scalepredict = nn.Conv2d(hidden_channels, in_channels, 1)
        self.biaspredict = nn.Conv2d(hidden_channels, in_channels, 1)
        self.in_channels = in_channels


    def forward(self, x):
        out = self.conv(x)


        scale = self.scalepredict(out)
        scale = torch.log(1 + torch.exp(scale))
        bias = self.biaspredict(out)
        n,c,h,w = scale.shape
        scale = scale.view(n,c,-1)
        scale = torch.mean(scale, dim=2)
        bias = bias.view(n,c,-1)
        bias = torch.mean(bias, dim=2)

        scale = scale.view(scale.size(0), scale.size(1),1,1)
        bias = bias.view(scale.size(0), scale.size(1),1,1)
        
        return scale, bias
