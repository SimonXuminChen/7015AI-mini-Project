# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


class FM(nn.Module):
    def __init__(self, feature_sizes, k_size=5):
        super(FM, self).__init__()
        self.feature_sizes = feature_sizes
        self.total_dim = int(feature_sizes[0]) + int(feature_sizes[1])  # total dims
        self.k_size = k_size
        self.learning_rate = 0.00000000000000001
        self.device = "cpu"
        # self.linear = nn.Linear(100, 1)
        self.w = torch.randn((len(feature_sizes), 1),requires_grad=True)
        self.w0 = 0.
        self.v = nn.Parameter(torch.ones((len(feature_sizes),1),requires_grad=True))

    def forward(self, Xi):
        # linear_part = self.linear(xi)
        # # 矩阵相乘 (batch*p) * (p*k)
        # inter_part1 = torch.mm(xi, xv.t())  # out_size = (batch, k)
        # # 矩阵相乘 (batch*p)^2 * (p*k)^2
        # inter_part2 = torch.mm(torch.pow(xi, 2), torch.pow(xv, 2).t())  # out_size = (batch, k)
        # output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        # # 这里torch求和一定要用sum
        if Xi.size()[-1] == 1:
            Xi = Xi.view((Xi.size()[0], Xi.size()[1]))
        m, n = Xi.size()  # 矩阵的行列数，即样本数和特征数
        # 初始化参数
        # w = random.randn(n, 1)#其中n是特征的个数
        inter_1 = torch.mm(Xi,self.v.t())  # *表示矩阵的点乘
        inter_2 = torch.mm(torch.pow(Xi, 2), torch.pow(self.v, 2).t())  # 二阶交叉项的计算
        interaction = torch.sum(torch.pow(inter_1, 2) - inter_2) * 0.5  # 二阶交叉项计算完成

        result = self.w0 + torch.sum(torch.mm(Xi,self.w.t())) + interaction

        return result  # out_size = (batch, 1)

    def fit(self, loader, epochs=1):
        model = self.train().to(device=self.device)
        loss_function = F.mse_loss
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        loss_data = []
        loss_index = []

        for epoch in range(epochs):
            training_loss = 0.0
            for t, (xi, xv, y) in enumerate(loader):
                xi = xi.to(device=self.device, dtype=torch.float)
                _ = xv.to(device=self.device, dtype=torch.float)
                # print(y)
                y = y.to(device=self.device, dtype=torch.float)

                for i in range(len(xi)):

                    total = model(xi[i])
                    loss = torch.sqrt(loss_function(total, y[i]))

                    # print("predicted value is: %.4f, label is %f" % (total, y))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.item()

                if t % 200 == 0:
                    print('Epoch %d,loss = %.4f' % (epoch + 1, loss.item()))
                    # self.check_accuracy(loader_val, model)
            loss_index.append(epoch + 1)
            loss_data.append(training_loss)

        plt.title("The result of loss function optimization(FM)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(loss_index, loss_data)
        plt.show()
