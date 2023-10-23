#!/user/bin/python
# -*- coding: UTF-8 -*-

"""
@version: python3.8
@author: 'zhuxun'
@contact: '821673485@qq.com'
@software: PyCharm
@file: 情感分类实战.py
@time: 2023/10/21 21:30
"""


"------------------------------------词表映射----------------------------------------------"
from collections import defaultdict
from torch import nn, optim


class Vocab:
    def __init__(self,tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                    tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx["<unk>"]

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token,self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


# embeding = nn.Embedding(8, 3)
# input = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
# output = embeding(input)
# print(output)
# print(output.shape)

import torch
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self,inputs,offsets):
        embeddings = self.embedding(inputs)
        embedding = embeddings.mean(dim=1)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        probs = F.log_softmax(outputs, dim=1)
        return probs


# mlp = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
# inputs = torch.tensor([[0, 1, 2, 1], [4, 6, 6, 7]], dtype=torch.long)
# outputs = mlp(inputs)
# print(outputs)

"-----------------------------------------数据处理--------------------------------"

def load_sentence_polarity():
    from nltk.corpus import sentence_polarity
    text=sentence_polarity.sents()
    print(type(text))
    vocab = Vocab.build(text)
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
                 + [(vocab.convert_tokens_to_ids(sentence), 1)
                    for sentence in sentence_polarity.sents(categories='neg')[:4000]]
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
                + [(vocab.convert_tokens_to_ids(sentence), 1)
                   for sentence in sentence_polarity.sents(categories='neg')[4000:]]
    return train_data, test_data, vocab


"""通过以上函数加载的数据不太方便直接给PyTorch使用，因此PyTorch提供了
DataLoader类（在torch.utils.data包中）。通过创建和调用该类的对象，可以在训练和
测试模型时方便地实现数据的采样、转换和处理等功能。例如，使用下列语句创建一个
DataLoader对象。
"""

from torch.utils.data import DataLoader, Dataset, dataset


def collate_fn(examples):
    input = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    #offsets = [0] + [i.shape[0] for i in input]

    #offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs = torch.cat(input)
    return inputs, offsets, targets


# data_loader = DataLoader(
#     dataset,
#     batch_size=64,
#     collate_fn=collate_fn,
#     shuffle=True
# )


class BowDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


"-----------------------------------训练测试-------------------------------"
from tqdm.auto import tqdm

embedding_dim = 128
hidden_dim = 256
num_class = 2
batch_size = 32
num_epoch = 5

train_data, test_data, vocab = load_sentence_polarity()
train_data = BowDataset(train_data)
test_data = BowDataset(test_data)
train_data_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=1, collate_fn=collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device)

nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, offsets, targets = [x.to(device) for x in batch]
        log_probs = model(inputs,offsets)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, offsets, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, offsets)
        acc += (output.argmax(dim=1) == targets).sum().item()
print(f"ACC: {acc / len(test_data_loader):.2f}")
