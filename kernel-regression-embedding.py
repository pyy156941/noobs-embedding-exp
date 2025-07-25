#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from collections import Counter

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter


# In[16]:


class KREmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sigma=1.0):
        super().__init__()
        self.embedding_weights = nn.Parameter(torch.randn(vocab_size, embedding_dim, dtype = torch.float32, device = device) * (1.0 / embedding_dim ** 0.5))
        self.sigma = sigma

    def forward(self, context, center):
        context_vecs = self.embedding_weights[context] # batch_size * (winlen - 1) * embedding
        center_vec = self.embedding_weights[center] # batch_size * embedding
        diff = context_vecs - center_vec.unsqueeze(1)  # batch_size * (winlen - 1) * embedding
        dist_sq = torch.sum(diff ** 2, dim=2)  # batch_size * (winlen - 1)
        weights = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # batch_size * (winlen - 1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # batch_size * (winlen - 1)
        weighted_context = (weights.unsqueeze(2) * context_vecs).sum(dim=1)  # batch_size * embedding

        similarity_matrix = torch.mm(weighted_context, self.embedding_weights.t())
        return similarity_matrix # compare to embeddings and output logits
    
    def getEmbedding(self, id):
        return self.embedding_weights[id]
# still stemming (use better tool), negative sampling

# In[8]:


class CorpusDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# In[9]:


def preprocessing(text, min_count = 15, threshold = 1e-5):
    _ = re.findall(r"[A-Za-z]+", text)
    words = []
    for word in _:
        words.append(word.lower())
    word_counts = Counter(words)
    words = [word for word in words if word_counts[word] >= min_count]

    sub_words = []
    for word in words:
        freq = word_counts[word] / len(words)
        discard_prob = 1.0 - np.sqrt(threshold / freq)

        if np.random.rand() > discard_prob:
            sub_words.append(word)

    word2id = {w : i for i, w in enumerate(set(sub_words))}
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
        
with open("wiki-103.train.tokens", 'r') as f:
    text = f.read()

words, word2id, id2word = preprocessing(text)
context_train, center_train, vocab_size, word_size = generateData(words, word2id, 7)
print(context_train.shape, center_train.shape, vocab_size)


# In[ ]:


def train(model, optimizer, dataloader):
    criterion = nn.CrossEntropyLoss()
    num_epoches = 100

    best_loss = float('inf')
    for epoch in range(num_epoches):
        model.train()
        epoch_loss = 0.0
        
        for batch_context, batch_center in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epoches}', leave=False):
            # Move data to device
            batch_context = batch_context.to(device)
            batch_center = batch_center.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_context, batch_center)
            target = batch_center.long()
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_context)
        
        epoch_loss /= len(context_train)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
        elif epoch > 20 and loss > best_loss * 1.05: 
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model, optimizer


# In[31]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = KREmbedding(vocab_size, 256, 7).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[ ]:


dataloader = DataLoader(
    dataset = CorpusDataset(context_train, center_train), 
    batch_size = 4096, 
    shuffle = True, 
    num_workers = 4,
    pin_memory = True if torch.cuda.is_available() else False
)
print(len(dataloader))

for name, param in model.named_parameters():
    print(f"{name}: {param.shape} | Device: {param.device}")
model, optimizer = train(model, optimizer, dataloader)


# In[ ]:


model.eval()
word2embed = {}
for (word, id) in word2id.items():
    embedding = model.getEmbedding(id).detach().to("cpu")
    embedding = embedding / torch.norm(embedding)
    word2embed[word] = embedding

with open("output-cbow.txt", 'w') as f:
    for (word, embed) in word2embed.items():
        f.write(word)
        f.write(str(list(embed.numpy())))
        f.write('\n')


# In[ ]:


def getClose(target, k = 5):
    sims = []

    cos = nn.CosineSimilarity(dim=0)
    for (word, embed) in word2embed.items():
        sim = cos(embed, target).item()
        sims.append((word, sim))

    res = sorted(sims, key=lambda x : x[1], reverse = True)
    return [_[0] for _ in res[:k]]


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
        target = word2embed[words[2].lower()] + word2embed[words[1].lower()] - word2embed[words[0].lower()]
        target = target / torch.norm(target)
        ans = getClose(target, 10)
        if words[3].lower() in ans:
            correctCount += 1
        print(words[0].lower(), words[1].lower(), words[2].lower(), ans)
        totalCount += 1
    except KeyError:
        pass

print(correctCount, totalCount)

state = {
    "state_dict" : model.state_dict(),  # model parameters
    "optimizer" : optimizer.state_dict(),  # optimizer state
    "word2id" : word2id, 
    "id2word" : id2word
}
torch.save(state, 'model-k103.pth')


# In[ ]:


completer = WordCompleter(list(word2embed.keys()))

while True:
    word = prompt("Word: ", completer = completer)
    cc = int(input("Input closest count: "))
    try:
        embed = word2embed[word]
        print(embed, *getClose(embed, cc))
    except KeyError:
        print("Nonexistent word.")

