import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import monotonic
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time
import logging
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # for softplus
from torch.nn import ReLU
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm

# Plotly imports for visualization
import plotly.graph_objects as go
from plotly_resampler import FigureResampler



# =======================================================
# Softplus Layer and MLP_Softplus network definition
# =======================================================
class SoftplusLayer(nn.Module):
    """
    Softplus layer: A linear transformation followed by the Softplus activation.
    Weight initialization is done with Kaiming uniform initialization, which is commonly used for ReLU-like activations.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Softplus()  # state-of-the-art smooth activation
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # Kaiming uniform initialization (nonlinearity can be 'relu' as Softplus has a similar behavior)
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            if self.linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
                bound = 1 / math.sqrt(fan_in)
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return self.activation(self.linear(x))


class MLP_Softplus(nn.Module):
    """
    Multi-Layer Perceptron using Softplus-based layers.
    """
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=50, num_layers=4,
                 dropout=0.0, final_activation=None):
        super().__init__()
        layers = []
        # First layer
        layers.append(SoftplusLayer(input_dim, hidden_dim))
        # Hidden layers with optional dropout
        for _ in range(num_layers - 2):
            layers.append(SoftplusLayer(hidden_dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        # Final linear layer (without activation by default)
        layers.append(nn.Linear(hidden_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




# =======================================================
# Softplus Layer and MLP_Softplus network definition
# =======================================================
class SoftplusLayer(nn.Module):
    """
    Softplus layer: A linear transformation followed by the Softplus activation.
    Weight initialization is done with Kaiming uniform initialization, which is commonly used for ReLU-like activations.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Softplus()  # state-of-the-art smooth activation
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # Kaiming uniform initialization (nonlinearity can be 'relu' as Softplus has a similar behavior)
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            if self.linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
                bound = 1 / math.sqrt(fan_in)
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return self.activation(self.linear(x))


class MLP_Softplus(nn.Module):
    """
    Multi-Layer Perceptron using Softplus-based layers.
    """

    def __init__(self, input_dim=2, output_dim=1, hidden_dim=50, num_layers=4,
                 dropout=0.0, final_activation=None):
        super().__init__()
        layers = []
        # First layer
        layers.append(SoftplusLayer(input_dim, hidden_dim))
        # Hidden layers with optional dropout
        for _ in range(num_layers - 2):
            layers.append(SoftplusLayer(hidden_dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        # Final linear layer (without activation by default)
        layers.append(nn.Linear(hidden_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

