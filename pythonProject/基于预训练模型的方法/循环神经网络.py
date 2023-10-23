#!/user/bin/python
# -*- coding: UTF-8 -*-

"""
@version: python3.8
@author: 'zhuxun'
@contact: '821673485@qq.com'
@software: PyCharm
@file: 循环神经网络.py
@time: 2023/10/20 19:58
"""
"----------------------------------RNN----------------------------------------"
import torch
from torch.nn import RNN
rnn = RNN(input_size=4,hidden_size=5,batch_first=True)
inputs = torch.rand(2,3,4)
outputs,hn=rnn(inputs)
print(outputs)
print(hn)
print(outputs.shape,hn.shape)

# "----------------------------------LSTM--------------------------------------"
from torch.nn import LSTM
lstm = LSTM(input_size=4,hidden_size=5,batch_first=True)
inputs = torch.rand(2,3,4)
outputs,(hn,cn)=lstm(inputs)
print(outputs)
print(hn)
print(cn)
print(outputs.shape,hn.shape,cn.shape)
"---------------------------------- Bi-RNN----------------------------------------"
bi_rnn =RNN(input_size=4,hidden_size=5,bidirectional=True,batch_first=True)
inputs = torch.rand(2,3,4)
print(inputs)
outputs,hn=bi_rnn(inputs)
print(outputs)
print(hn)
print(outputs.shape,hn.shape)

"----------------------------------Bi-LSTM-----------------------------------------"
bi_lstm = LSTM(input_size=4,hidden_size=5,batch_first=True,bidirectional=True)
inputs = torch.rand(2,3,4)
outputs,(hn,cn)=lstm(inputs)
print(outputs)
print(hn)
print(cn)
print(outputs.shape,hn.shape,cn.shape)