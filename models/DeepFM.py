# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time


class DeepFM(nn.Module):
    def __init__(self, features, embedding_size=5, hidden_dims=[32, 32], dropout=[0.5, 0.5]):
        super(DeepFM, self).__init__()
        self.field_size = 2  # according to the dataset, we know the field include user id, movie id,timestamp
        self.features = [features[:, 0], features[:, 1]]  # just consider user id and movie id
        self.feature_sizes = [len(np.unique(self.features[0])),
                             len(np.unique(self.features[1]))]  # constain data size due to the mini project
        self.hidden_dim = hidden_dims
        self.bias = torch.nn.Parameter(torch.randn(1))
        self.embedding_size = embedding_size

        self.embedding_layer = nn.ModuleList(
            [nn.Embedding(feature_size,self.embedding_size) for feature_size in self.feature_sizes]
        )

        # initial dnn model part
        self.dims = self.field_size * self.embedding_size
        self.fc_dim = 100 # fully connected layer dimension

        self.fc_layer1 = nn.Sequential(
            nn.Linear(self.dims, self.fc_dim), nn.BatchNorm2d(self.fc_dim)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim), nn.BatchNorm2d(self.fc_dim)
        )

        # initial fm model part

    def forward(self, *input):
        pass