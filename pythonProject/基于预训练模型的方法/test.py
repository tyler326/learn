#!/user/bin/python
# -*- coding: UTF-8 -*-

"""
@version: python3.8
@author: 'zhuxun'
@contact: '821673485@qq.com'
@software: PyCharm
@file: test.py
@time: 2023/10/22 0:42
"""
from collections import defaultdict
# NLTK提供的句子倾向性分析数据 sentence_polarity
from nltk.corpus import sentence_polarity
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
# tqdm是一个Python模块，能以进度条的方式显示迭代的进度
from tqdm.auto import tqdm

# 实现词表映射的类：Vocab
class Vocab:
    def __init__(self,tokens = None):
        # 使用列表存储所有的标记，从而可根据索引值获取相应的标记
        self.idx_to_token = list()
        # 使用字典实现标记到索引值的映射
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

            self.unk = self.token_to_idx["<unk>"]

    # 创建词表，text包含若干句子，每个句子由若干标记构成
    @classmethod
    def build(cls,text,min_freq = 1 , reserved_tokens = None):
        # 存储标记及其出现次数的映射字典
        token_freqs = defaultdict(int)
        # 无重复地进行标记
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        # 用户自定义的预留标记
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token,freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]

        return cls(uniq_tokens)

    # 返回词表的大小，即词表中有多少个互不相同的标记
    def __len__(self):
        return len(self.idx_to_token)

    # 查找输入标记对应的索引值
    def __getitem__(self, token):
        return self.token_to_idx.get(token,self.unk)

    # 查找一系列输入标记对应的索引值
    def convert_tokens_to_ids(self,tokens):
        return [self[token] for token in tokens]

    # 查找一系列索引值对应的标记
    def convert_ids_to_tokens(self,indices):
        return [self.idx_to_token[index] for index in indices]

# 保存词表
def save_vocab(vocab,path):
    with open(path,"w") as writer:
        writer.write("\n".join(vocab.idx_to_token))

# 读取词表
def read_vocab(path):
    with open(path,"r") as f:
        tokens = f.read().split("\n")
    return Vocab(tokens)

# 创建存储数据的BowDataset的子类 → 词袋
class BowDataset(Dataset):
    def __init__(self,data):
        # data 为原始的数据
        self.data = data
    def __len__(self):#
        # 返回数据集中样例的数目
        return len(self.data)
    def __getitem__(self, i):
        # 获取下标为1的样例
        return self.data[i]

# 对一个批次的样本进行整理，从独立样本集合中构建各批次的输入输出
def collate_fn(examples):
    # 输入inputs定义为一个张量的列表，其中每个张量为原始句子中标记序列对应的索引值序列
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # 输出的目标targets为该批次中全部样例输出结果
    targets = torch.tensor([ex[1] for ex in examples],dtype=torch.long)
    # 获取一个批次中每个样例的序列长度
    offsets = [0] + [i.shape[0] for i in inputs]
    # 根据序列的长度，转换为每个序列起始位置的偏移量(offsets)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim = 0)

    # 将inputs列表中的张量拼接成一个大的张量
    inputs = torch.cat(inputs)
    return inputs , offsets , targets

# 融入词向量层的多层感知器模型
class MLP(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_class):
        super(MLP, self).__init__()
        # 词向量层
        self.embedding = nn.EmbeddingBag(vocab_size,embedding_dim)
        # 线性变换： 词向量层 → 隐含层
        self.linear1 = nn.Linear(embedding_dim,hidden_dim)
        #  使用relu激活函数
        self.activate = torch.relu
        # 线性变换： 激活层 → 输出层
        self.linear2 = nn.Linear(hidden_dim,num_class)

    def forward(self,inputs,offsets):
        embedding = self.embedding(inputs,offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = torch.log_softmax(outputs,dim = 1)
        return log_probs

# 设置超参数
embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5

# 数据加载

def load_sentence_polarity():
    # 使用全部句子集合创建词表
    vocab = Vocab.build(sentence_polarity.sents())
    # 褒贬各4000句作为训练数据
    train_data = [(vocab.convert_tokens_to_ids(sentence),0) for sentence in sentence_polarity.sents(categories="pos")[:4000]] + [
        (vocab.convert_tokens_to_ids(sentence),1) for sentence in sentence_polarity.sents(categories = "neg")[:4000]
    ]

    # 剩下数据作为测试数据
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in
                  sentence_polarity.sents(categories="pos")[4000:]] + [
                     (vocab.convert_tokens_to_ids(sentence), 1) for sentence in
                     sentence_polarity.sents(categories="neg")[4000:]
                 ]
    return train_data,test_data,vocab

train_data ,test_data ,vocab= load_sentence_polarity()
train_dataset = BowDataset(train_data)
test_dataset = BowDataset(test_data)

train_data_loader = DataLoader(train_dataset,batch_size = batch_size,collate_fn=collate_fn,shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size = 1,collate_fn= collate_fn,shuffle=False)

# 模型加载
device = torch.device("cpu")
model = MLP(len(vocab),embedding_dim,hidden_dim,num_class)
# 将模型加载到CPU设备
model.to(device)

# 训练过程
nll_loss = nn.NLLLoss()
# 使用Adam优化器
optimizer = optim.Adam(model.parameters(),lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader,desc=f"Training Epoch {epoch}"):
        inputs,offsets,targets = [x.to(device) for x in batch]
        log_probs = model(inputs,offsets)
        # 计算损失
        loss = nll_loss(log_probs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Loss:{total_loss:.2f}")

# 测试过程
acc = 0
for batch in tqdm(test_data_loader,desc="Testing"):
    inputs , offsets , targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs,offsets)
        acc += (output.argmax(dim = 1) == targets).sum().item()

# 输出在测试集上的准确率
print(f"Acc:{acc / len(test_data_loader):.2f}")

