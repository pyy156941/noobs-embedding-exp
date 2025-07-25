#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from collections import Counter

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from gensim.utils import simple_preprocess
import spacy

torch.backends.cudnn.benchmark = True
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
nlp.max_length = 100_000_0000


# In[ ]:


class KREmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sigma=1.0, num_negatives=5):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        nn.init.normal_(self.center_embeddings.weight, mean=0, std=1 / (embedding_dim ** 2))
        nn.init.normal_(self.context_embeddings.weight, mean=0, std=1 / (embedding_dim ** 2))
        
        self.sigma = sigma
        self.num_negatives = num_negatives
        self.vocab_size = vocab_size

    def forward(self, context, center, neg_samples=None):
        context_vecs = self.center_embeddings(context)  # [batch_size, winlen-1, embedding_dim]
        center_vec = self.context_embeddings(center)  # [batch_size, embedding_dim]
        neg_vecs = self.context_embeddings(neg_samples)  # [batch_size, num_negatives, embedding_dim]
        diff = context_vecs - center_vec.unsqueeze(1)  # [batch_size, winlen-1, embedding_dim]
        dist_sq = torch.sum(diff**2, dim=2)  # [batch_size, winlen-1]
        weights = torch.exp(-dist_sq / (2 * self.sigma**2))  # [batch_size, winlen-1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize
        weighted_context = (weights.unsqueeze(2) * context_vecs).sum(dim=1)  # [batch_size, embedding_dim]
        return weighted_context, center_vec, neg_vecs

    def getEmbedding(self, word_id):
        word_tensor = torch.tensor(word_id, device=self.center_embeddings.weight.device)
        return self.center_embeddings(word_tensor)


# In[4]:


class CorpusDataset(Dataset):
    def __init__(self, contexts, centers, neg_samples):
        self.contexts = contexts
        self.centers = centers
        self.neg_samples = neg_samples  # New: store negative samples
        
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.centers[idx], self.neg_samples[idx]


# In[ ]:


def preprocessing(text, min_count=5, threshold=1e-5, chunk_size=500000):
    # Stage 1: Fast Gensim cleaning (memory-safe)
    words = simple_preprocess(text, deacc=True, min_len=2)
    
    # Stage 2: Chunked SpaCy lemmatization
    def process_chunk(chunk):
        doc = nlp(" ".join(chunk))
        return [
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_punct and len(token) > 1
        ]
    
    # Split words into chunks to avoid SpaCy memory issues
    lemmatized = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        lemmatized.extend(process_chunk(chunk))
    
    # Stage 3: Subsampling (Word2Vec style)
    word_counts = Counter(lemmatized)
    total_words = len(lemmatized)
    subsampled = [
        word for word in lemmatized
        if word_counts[word] >= min_count and 
            np.random.rand() > (1 - np.sqrt(threshold / (word_counts[word]/total_words)))
    ]
    
    # Build vocab
    vocab = list(set(subsampled))
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    
    return subsampled, word2id, id2word

def generateData(words, word2id, winlen, num_negatives=15):
    vocab_size = len(word2id)
    word_size = len(words)
    batch_size = word_size - winlen + 1
    word_counts = Counter(words)
    word_freq = np.array([word_counts[id2word[i]] for i in range(vocab_size)])
    noise_dist = word_freq ** 0.75
    noise_dist /= noise_dist.sum()
    
    # Pre-allocate numpy arrays
    context_train = np.zeros((batch_size, winlen - 1), dtype=np.int32)
    center_train = np.zeros((batch_size), dtype=np.int32)
    neg_samples = np.zeros((batch_size, num_negatives), dtype=np.int32)
    
    rng = np.random.default_rng()
    neg_samples = rng.choice(
        vocab_size, 
        size=(batch_size, num_negatives), 
        replace=True, 
        p=noise_dist
    )
    for i in range(winlen // 2, len(words) - winlen // 2):
        idx = i - winlen // 2
        
        context = words[i - winlen // 2 : i] + words[i + 1 : i + winlen // 2 + 1]
        context_train[idx] = [word2id[w] for w in context]
        center_train[idx] = word2id[words[i]]
    
    return (
        torch.tensor(context_train),
        torch.tensor(center_train),
        torch.tensor(neg_samples),
        vocab_size,
        len(words)
    )
        
with open("wiki-103.train.tokens", 'r') as f:
    text = f.read()

words, word2id, id2word = preprocessing(text)
context_train, center_train, neg_train, vocab_size, word_size = generateData(words, word2id, 7)
print(context_train.shape, center_train.shape, neg_train.shape, vocab_size, context_train.dtype)

# with open("words.txt", 'w') as f:
#     f.write(str(words))


# In[30]:


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, weighted_context, center_vec, neg_vecs):
        # Positive score (kernel-weighted context vs center)
        pos_score = torch.sum(weighted_context * center_vec, dim=1)
        pos_loss = F.logsigmoid(pos_score)
        
        # Negative scores (kernel-weighted context vs negative samples)
        neg_scores = torch.bmm(neg_vecs, weighted_context.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)
        
        # Combine losses
        loss = -(pos_loss + neg_loss).mean()
        
        return loss


# In[35]:


def train(model, optimizer, dataloader):
    criterion = NegativeSamplingLoss() if hasattr(model, 'num_negatives') else nn.CrossEntropyLoss()
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
            batch_context, batch_center, batch_negs = batch
            batch_context, batch_center, batch_negs = batch_context.to(device), batch_center.to(device), batch_negs.to(device)
            
            optimizer.zero_grad()
            
            context_vec, center_vec, neg_vecs = model(batch_context, batch_center, batch_negs)
            loss = criterion(context_vec, center_vec, neg_vecs)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss/len(dataloader)}")
    
    return model, optimizer


# In[40]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model = KREmbedding(vocab_size, 256, 7).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


# In[ ]:


dataloader = DataLoader(
    dataset = CorpusDataset(context_train, center_train, neg_train), 
    batch_size = 131072, 
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
torch.save(state, 'model-kn103.pth')


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

