# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class DeepFM(nn.Module):

    def __init__(self, feature_sizes, embedding_size=5, hidden_dims=[32, 32], dropout=[0.5, 0.5]):
        super(DeepFM, self).__init__()
        self.field_size = len(
            feature_sizes)  # according to the dataset, we know the field include user id, movie id,timestamp
        self.feature_sizes = feature_sizes  # constain data size due to the mini project
        self.hidden_dim = hidden_dims
        self.bias = nn.Parameter(torch.randn(1))
        self.embedding_size = embedding_size
        self.learning_rate = 0.01
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

        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
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

    def fit(self, loader_train, epochs=1):
        model = self.train().to(device=self.device)
        loss_function = F.mse_loss
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_data = []
        loss_index = []

        with open("deepFM_training_data.csv","w") as f:
            for epoch in range(epochs):
                training_loss = 0.0
                for t, (xi, xv, y) in enumerate(loader_train):
                    xi = xi.to(device=self.device, dtype=torch.float)
                    xv = xv.to(device=self.device, dtype=torch.float)
                    y = y.to(device=self.device, dtype=torch.float)*2

                    total = model(xi, xv)
                    y = y.reshape(total.size())
                    loss = torch.sqrt(loss_function(total, y))
                    for i in range(len(total)):
                        error = torch.abs(y[i]-total[i])
                        content = str(int(xi[i][0]))+","+str(int(xi[i][1]))+","+str(int(y[i]))+","+str(int(total[i]))+","+str(int(error))+"\n"
                        f.writelines(content)


                    # print("predicted value is: %.4f, label is %f" % (total, y))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()

                    if t % 200 == 0:
                        with open("./deepfm_train.txt", "w") as f_2:
                            content_2 = 'Epoch %d,loss = %.4f' % (epoch + 1, loss.item())
                            content_2 += "\n"
                            f_2.writelines(content_2)
                            print(content_2)
                        # self.check_accuracy(loader_val, model)
                loss_index.append(epoch + 1)
                loss_data.append(training_loss)
        # f.()
        plt.title("The result of loss function optimization(DeepFm)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(loss_index, loss_data)
        plt.show()

    def validation(self, loader):
        print("Now, it's validation time!")
        model = self.eval()  # set model to evaluation mode
        num_correct = 0
        num_samples = 0

        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=torch.float)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=torch.float)

                # calculate predicted_y and get y value
                # then compare if the error between them is smaller than 0.5
                total = model(xi, xv)
                y = y.reshape(total.size())

                num_samples += len(total)
                for i in range(len(total)):
                    val_error = abs(y[i] - total[i])
                    if val_error < 0.5:
                        num_correct += 1

        accuracy = float(num_correct) / num_samples
        print("The accuracy of validation is %d / %d (%.2f%%)" % (num_correct,num_samples,100 * accuracy))

