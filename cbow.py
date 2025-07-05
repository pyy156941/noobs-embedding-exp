#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# In[ ]:


class CbowEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, winlen):
        super().__init__()
        self.embedding_weights = nn.Parameter(torch.randn(vocab_size, embedding_dim, dtype = torch.float32, device = device) * (1.0 / embedding_dim ** 0.5))

    def forward(self, context, center):
        context_vecs = self.embedding_weights[context] # batch_size * (winlen - 1) * embedding
        avg_vecs = context_vecs.mean(dim = 1)
        return avg_vecs

    def getEmbedding(self, id):
        return self.embedding_weights[id]


# In[13]:


def preprocessing(text):
    _ = re.findall(r"[A-Za-z]+", text)
    words = []
    for word in _:
        words.append(word.lower())
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    word2id = {w : i for i, w in enumerate(set(words))}
    id2word = {i : w for _, (w, i) in enumerate(word2id.items())}
    return words, word2id, id2word

def generateData(words, word2id, winlen): # winlen must be odd
    vocab_size = len(word2id)
    word_size = len(words)
    batch_size = word_size - winlen + 1
    context_train = np.zeros((batch_size, winlen - 1))
    center_train = np.zeros((batch_size))
    for _ in range(winlen // 2, word_size - winlen // 2):
        fr = _ - winlen // 2
        center_train[fr] = word2id[words[_]]
        for __ in range(_ - winlen // 2, _):
            context_train[fr][__ - (_ - winlen // 2)] = word2id[words[__]]
        for __ in range(_ + 1, _ + winlen // 2 + 1):
            context_train[fr][__ - (_ - winlen // 2) - 1] = word2id[words[__]]
    return torch.tensor(context_train).int(), torch.tensor(center_train).int(), vocab_size, word_size
        
with open("wiki-2.train.tokens", 'r') as f:
    text = f.read()

words, word2id, id2word = preprocessing(text)
context_train, center_train, vocab_size, word_size = generateData(words, word2id, 5)
print(context_train.shape, center_train.shape, vocab_size)


# In[14]:


def train(model, optimizer, context_train, center_train):
    criterion = nn.CosineEmbeddingLoss()
    flag = torch.ones(context_train.shape[0], device = device)
    num_epoches = 10000

    best_loss = float('inf')
    for epoch in range(num_epoches):
        model.train()
        optimizer.zero_grad()
        output = model(context_train, center_train)
        target = model.embedding_weights[center_train].detach().to(device)
        loss = criterion(output, target, flag)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
        elif epoch > 100 and loss.item() > best_loss * 1.05: 
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model, optimizer


# In[15]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CbowEmbedding(vocab_size, 128, 5).to(device)
model.embedding_weights.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[ ]:


current_model = torch.load("model-cbow-128.pth", map_location=device)
model.load_state_dict(current_model["state_dict"])
optimizer.load_state_dict(current_model["optimizer"])


# In[ ]:


context_train = context_train.to(device)
center_train = center_train.to(device)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} | Device: {param.device}")
#model, optimizer = train(model, optimizer, context_train, center_train)


# In[17]:


word2embed = {}
for (word, id) in word2id.items():
    embedding = model.getEmbedding(id).detach().to("cpu")
    embedding = embedding / (torch.sqrt(embedding.pow(2).sum()) + 1e-8)
    word2embed[word] = embedding

with open("output-cbow.txt", 'w') as f:
    for (word, embed) in word2embed.items():
        f.write(word)
        f.write(str(list(embed.numpy())))
        f.write('\n')


# In[18]:


def getClose(target):
    sims = []

    cos = nn.CosineSimilarity(dim=0)
    for (word, embed) in word2embed.items():
        sim = cos(embed, target).item()
        sims.append((word, sim))

    res = sorted(sims, key=lambda x : x[1], reverse = True)
    return res[:5]


# In[20]:


import random

correctCount = 0
totalCount = 0
with open("questions-words.txt", 'r') as f:
    qs = f.read()

qs_s = qs.split('\n')
random.shuffle(qs_s)

for q in qs_s[:100]:
    words = q.split()
    try:
        ans = getClose(word2embed[words[0].lower()] + word2embed[words[1].lower()] - word2embed[words[2].lower()])[0][0]
        if ans == words[3].lower():
            correctCount += 1
        else:
            print(words[0].lower(), words[1].lower(), words[2].lower(), ans)
        totalCount += 1
    except KeyError:
        pass

print(correctCount, totalCount)


# In[12]:


state = {
    'state_dict': model.state_dict(),  # model parameters
    'optimizer': optimizer.state_dict(),  # optimizer state
}
torch.save(state, 'model-cbow-128.pth')

completer = WordCompleter(list(word2embed.keys()))

while True:
    word = prompt("Word: ", completer = completer)
    try:
        embed = word2embed[word]
        print(embed, getClose(embed))
    except KeyError:
        print("Nonexistent word.")
