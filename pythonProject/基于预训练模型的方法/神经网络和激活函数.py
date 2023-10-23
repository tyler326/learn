#!/user/bin/python
# -*- coding: UTF-8 -*-

"""
@version: python3.8
@author: 'zhuxun'
@contact: '821673485@qq.com'
@software: PyCharm
@file: 神经网络和激活函数.py
@time: 2023/10/20 15:23
"""
"--------------------------------神经网络和激活函数---------------------------------------------------"
import torch
from torch import nn
linear =nn.Linear(in_features=32,out_features=2)
inputs = torch.rand(3,32)
outputs = linear(inputs)
print(outputs)

"""激活函数"""
from torch.nn import functional as F
#Sigmoid
activation = F.sigmoid(inputs)
print("sigmod:\n{}".format(activation))
#Softmax,dim沿第n维计算
activation = F.softmax(outputs,dim=1)
print("softmax:\n{}".format(activation))
#ReLu
activation =F.relu(outputs)
print("ReLu:\n{}".format(activation))

"----------------------------------异或问题-----------------------------------------------------------------"
class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_class):
        super(MLP,self).__init__()
        #线性变换: 输入层->隐藏层
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        #使用激活函数ReLu
        self.activate = F.relu
        # 线性变换: 隐藏层->输出层
        self.linear2 = nn.Linear(hidden_dim,num_class)

    def forward(self,inputs):
        hidden =self.linear1(inputs)
        activation = self.activate(hidden)
        #activation=F.relu(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs,dim=1)
        return probs

mlp =MLP(input_dim=4,hidden_dim=8,num_class=2)
inputs = torch.rand(3,4)
print(inputs)
probs = mlp(inputs)
print(probs)