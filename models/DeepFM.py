# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time


class DeepFM(nn.Module):

    def __init__(self, feature_sizes, embedding_size=5, hidden_dims=[32, 32], dropout=[0.5, 0.5]):
        super(DeepFM, self).__init__()
        self.field_size = len(
            feature_sizes)  # according to the dataset, we know the field include user id, movie id,timestamp
        self.feature_sizes = feature_sizes  # constain data size due to the mini project
        self.hidden_dim = hidden_dims
        self.bias = torch.nn.Parameter(torch.randn(1))
        self.embedding_size = embedding_size
        self.learning_rate = 0.2
        self.bias = self.bias = torch.nn.Parameter(torch.randn(1))
        self.device = torch.device('cpu')

        # initial fm model part
        self.first_order_feature = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes]
        )
        self.second_order_feature = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes]
        )

        # initial dnn model part
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dim
        self.fc_dim = 100  # fully connected layer dimension

        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i),
                    nn.Dropout(dropout[i - 1]))  # dropout is used to prevent overfitting



    def forward(self, Xi, Xv):
        """
        Xi: A tensor of input's index, shape of (N,field_size,1)
        Xv: A tensor of input's value, shape of (N,field_size,1)
        """
        # fm
        fm_first_order_emb_arr = []  # Xi*W
        for i, emb in enumerate(self.first_order_feature):
            Xi_temp = Xi[:, i, :].to(device=self.device, dtype=torch.long)
            fm_first_order_emb_arr.append((torch.sum(emb(Xi_temp), 1).t() * Xv[:, i]).t())

        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)

        fm_second_order_emb_arr = []  # Xi * Vi
        for i, emb in enumerate(self.second_order_feature):
            Xi_temp = Xi[:, i, :].to(device=self.device, dtype=torch.long)
            fm_second_order_emb_arr.append((torch.sum(emb(Xi_temp), 1).t() * Xv[:, i]).t())

        sum_Xi_Vi = sum(fm_second_order_emb_arr)
        Xi_2_Vi_2 = [item * item for item in fm_second_order_emb_arr]

        fm_second_order = 0.5 * (sum_Xi_Vi * sum_Xi_Vi - sum(Xi_2_Vi_2))

        # deep
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dim) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)

        # sum
        self.bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum

    def fit(self, loader_train,loader_val,epochs=1):
        model = self.train().to(device=self.device)
        loss_function = F.mse_loss
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        for _ in range(2000):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=torch.float)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)

                total = model(xi, xv)
                y.reshape(total.size())
                loss = loss_function(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if t % 10 ==0:
                    print('Iteration %d,loss = %.4f' % (t, loss.item()))
                    # self.check_accuracy(loader_val,model)

    def check_accuracy(self, loader, model):
        print('Checking accuracy on validation set')
        # num_correct = 0
        # num_samples = 0
        error=0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=torch.float)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)

                total = model(xi, xv)

                error += F.mse_loss(total,y)

                # num_correct += preds.sum()
                # print(preds)
                # num_samples += preds.size()[0]
            #                print("successful")
            # acc = float(num_correct) / num_samples
            # print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
            print("The summary of loss fuction error is %f" % error)