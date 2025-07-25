#!/usr/bin/env python
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

class KREmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sigma=1.0):
        super().__init__()
        self.embedding_weights = nn.Parameter(torch.randn(vocab_size, embedding_dim, dtype = torch.float32))
        self.sigma = sigma

    def forward(self, context, center):
        context_vecs = self.embedding_weights[context] # batch_size * (winlen - 1) * embedding
        center_vec = self.embedding_weights[center] # batch_size * embedding
        diff = context_vecs - center_vec.unsqueeze(1)  # batch_size * (winlen - 1) * embedding
        dist_sq = torch.sum(diff ** 2, dim=2)  # batch_size * (winlen - 1)
        weights = torch.exp(-dist_sq / (2 * self.sigma ** 2))  # batch_size * (winlen - 1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # batch_size * (winlen - 1)
        weighted_context = (weights.unsqueeze(2) * context_vecs).sum(dim=1)  # batch_size * embedding

        return weighted_context
    
    def getEmbedding(self, id):
        return self.embedding_weights[id]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
current_model = torch.load("model-k103.pth", map_location=device)
word2id = current_model["word2id"]
id2word = current_model["id2word"]
vocab_size = len(word2id)

print(device)
model = KREmbedding(vocab_size, 256, 7).to(device)
model.embedding_weights.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.load_state_dict(current_model["state_dict"])
optimizer.load_state_dict(current_model["optimizer"])

model.eval()
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} | Device: {param.device}")
# model, optimizer = train(model, optimizer, context_train, center_train)

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

import faiss

def getClose(target, k = 5):
    target = target.cpu()
    words = list(word2embed.keys())
    embeddings = torch.stack([word2embed[w].cpu() for w in words])
    embedding_dim = target.shape[0]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.float().numpy())
    sim, indices = index.search(target.float().numpy().reshape(1, -1), k)
    return [words[i] for i in indices[0]]

import random

correctCount = 0
totalCount = 0
with open("questions-words.txt", 'r') as f:
    qs = f.read()

qs_s = qs.split('\n')
random.shuffle(qs_s)

for q in tqdm(qs_s):
    words = q.split()
    if len(words) != 4:
        continue
    try:
        target = word2embed[words[2].lower()] + word2embed[words[1].lower()] - word2embed[words[0].lower()]
        target = target / torch.norm(target)
        ans = getClose(target, 10)
        if words[3].lower() in ans:
            correctCount += 1
#        print(words[0].lower(), words[1].lower(), words[2].lower(), ans)
        totalCount += 1
    except KeyError:
        pass

print(correctCount, totalCount) # 7191 17776 7422 17776 10268 17776

def tc(word):
    if word.lower() in word2embed:
        return getClose(word2embed[word.lower()], 2)[1]
    else:
        return word.lower()

with open("a.txt", 'r') as f:
    text = f.read()

pattern = re.compile(r'([a-zA-Z]+)')
modified_essay = pattern.sub(lambda match: tc(match.group(0)), text)

with open("a_modified.txt", "w") as f:
    f.write(modified_essay)

completer = WordCompleter(list(word2embed.keys()))

while True:
    try:
        word = prompt("Word: ", completer = completer)
        cc = int(input("Input closest count: "))
        embed = word2embed[word]
        print(*getClose(embed, cc))
    except KeyError:
        print("Nonexistent word.")
    except ValueError:
        print("Invalid input, please retry.")

