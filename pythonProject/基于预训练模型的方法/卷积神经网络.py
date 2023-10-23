#!/user/bin/python
# -*- coding: UTF-8 -*-

"""
@version: python3.8
@author: 'zhuxun'
@contact: '821673485@qq.com'
@software: PyCharm
@file: 卷积神经网络.py
@time: 2023/10/20 16:40
"""
import torch
from torch.nn import Conv1d

conv1 =Conv1d(5,2,4)
conv2 =Conv1d(5,2,3)
inputs = torch.rand(2,5,6)
outputs1 = conv1(inputs)
outputs2 = conv2(inputs)
print(outputs1)
print(outputs2)

"通过torch.nn.functional中的池化函数"
# import torch.nn.functional as F
# outputs1 =F.max_pool1d(outputs1,kernel_size=outputs1.shape[2])
# outputs2 =F.max_pool1d(outputs2,kernel_size=outputs2.shape[2])
from torch.nn import MaxPool1d
pool1 = MaxPool1d(3)
pool2 = MaxPool1d(4)

outputs_pool1 =pool1(outputs1)
outputs_pool2 =pool1(outputs2)
print(outputs_pool1)
print(outputs_pool2)


"""
    由于outputs_pool1和outputs_pool2是两个独立的张量，为了进行下一步操作，
    还需要调用 torch.cat 函数将它们拼接起来。在此之前，还需要调用squeeze函数将最后一个为1的维度删除，
    即将2行1列的矩阵变为1个向量。
"""

outputs_pool_squeeze1 =outputs_pool1.squeeze(dim=2)
outputs_pool_squeeze2 =outputs_pool2.squeeze(dim=2)
print(outputs_pool_squeeze1)
print(outputs_pool_squeeze2)
outputs_pool =torch.cat([outputs_pool_squeeze1,outputs_pool_squeeze2],dim=1)
print(outputs_pool)

from torch.nn import Linear
linear = Linear(4,2)
outputs_linear =linear(outputs_pool)
print(outputs_linear)