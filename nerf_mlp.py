import torch
from torch import nn
import torch.nn.functional as F

import pdb

class NeRFConfig:
    def __init__(self):
        self.freq_encoding_position = 10
        self.in_features_position = 60
        self.layers_density = 8
        self.density_layer_channels = 256
        self.skip_connection_layers = [4]
        self.freq_encoding_direciton = 4
        self.in_features_direction = 24
        self.layers_color = 1
        self.color_layer_channels = 128

class NeRFMLP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else NeRFConfig()

        self.density_layers = []
        in_features = self.config.in_features_position
        for i in range(self.config.layers_density):
            if i in self.config.skip_connection_layers:
                in_features += self.config.in_features_position
            out_features = self.config.density_layer_channels
            if i == self.config.layers_density - 1:
                out_features += 1
            layer = nn.Linear(in_features=in_features, out_features=out_features)
            self.density_layers.append(layer)
            in_features = out_features

        self.color_layers = []
        in_features = self.config.density_layer_channels + self.config.in_features_direction
        for i in range(self.config.layers_color):
            out_features = self.config.color_layer_channels
            layer = nn.Linear(in_features=in_features, out_features=out_features)
            self.color_layers.append(layer)
            in_features = out_features
        layer = nn.Linear(in_features=in_features, out_features=3)
        self.color_layers.append(layer)

    @staticmethod
    def frequency_encoding(x, max_freq):
        fns = []
        for i in range(max_freq):
            fns.append(lambda k, x : torch.sin(pow(2.0, k) * x))
            fns.append(lambda k, x : torch.cos(pow(2.0, k) * x))
        return torch.cat([fns[i](i//2, x) for i in range(len(fns))], -1)

    def forward(self, xyz, dir):
        xyz_freq = self.frequency_encoding(xyz, self.config.freq_encoding_position)
        h = xyz_freq
        for i in range(len(self.density_layers)):
            if i in self.config.skip_connection_layers:
                h = torch.hstack((h, xyz_freq))
            h = F.relu(self.density_layers[i](h))

        density = h[:, 0]

        dir_freq = self.frequency_encoding(dir, self.config.freq_encoding_direciton)
        h = torch.hstack((h[:,1:], dir_freq))
        for i in range(len(self.color_layers)):
            h = self.color_layers[i](h)
            if i == len(self.color_layers) - 1:
                h = torch.sigmoid(h)
            else:
                h = F.relu(h)

        return density, h
    

if __name__ == '__main__':
    xyz = torch.rand(10, 3)
    print(xyz)
    print(xyz.shape)
    dir = torch.rand(10, 3)
    print(dir)
    print(dir.shape)
    mlp = NeRFMLP()
    density, color = mlp(xyz, dir)
    print(color)
    print(color.shape)