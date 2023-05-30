# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 2048) 
        self.relu = nn.ReLU() 
        self.fc1 = nn.Linear(2048, 1024) 
        self.fc2 = nn.Linear(1024, 128)   
        self.layer_out = nn.Linear(128, dim_out)   
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1,2048)
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return self.softmax(x)


