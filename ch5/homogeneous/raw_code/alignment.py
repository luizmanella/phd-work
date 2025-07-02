import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm

import numpy as np

import ot

rng = np.random.RandomState(42)


import torch
import torchvision
import torchvision.transforms as transforms

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def compute_barycenter(n, dists, reg):
    A = np.vstack(dists).T
    M = ot.utils.dist0(n)
    M /= M.max()
    weights = np.ones(len(dists))/len(dists)
    WB = ot.bregman.barycenter(A, M, reg, weights)
    return WB

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 1

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

n = 250
wb_batch_size = 100
r_dist, g_dist, b_dist = [], [], []
r,g,b = [],[],[]
for i, t in tqdm(enumerate(trainset)):
    img = t[0].flatten(1)
    idx = rng.randint(img.shape[1], size=(n,))
    _r = img[0, idx]
    _g = img[1, idx]
    _b = img[2, idx]
    _r /= _r.sum()
    _g /= _g.sum()
    _b /= _b.sum()
    r.append(_r.numpy())
    g.append(_g.numpy())
    b.append(_b.numpy())
    if i != 0:
        if i % (wb_batch_size - 1) == 0 or i == (len(trainset) - 1):
            r_dist.append(r)
            g_dist.append(g)
            b_dist.append(b)
            r,g,b = [], [], []

reg = 1e-2
rwb = []
for wb in tqdm(r_dist):
    rwb.append(compute_barycenter(n, wb, reg))
rwb = compute_barycenter(n, rwb, reg)

gwb = []
for wb in tqdm(g_dist):
    gwb.append(compute_barycenter(n, wb, reg))
gwb = compute_barycenter(n, gwb, reg)

bwb = []
for wb in tqdm(b_dist):
    bwb.append(compute_barycenter(n, wb, reg))
bwb = compute_barycenter(n, bwb, reg)

global_wb = np.array([rwb, gwb, bwb]).T

new_trainset = []
for i, t in tqdm(enumerate(trainset)):
    img = t[0].flatten(1).T
    idx = rng.randint(img.shape[1], size=(n,))
    img = img[idx, :].numpy()
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=img, Xt=global_wb)
    transported = torch.from_numpy(ot_sinkhorn.transform(t[0].flatten(1).T.numpy())).reshape(32,32,3).permute(2,0,1)
    new_trainset.append((transported, t[1]))


store_json_train = []
for i in tqdm(new_trainset):
    store_json_train.append((i[0].tolist(),i[1]))

# store the new trainset - which is projected onto the WB
with open('./new_trainset.json','w') as f:
    json.dump(store_json_train, f)

# stores the global barycenters
store_wbs = global_wb.tolist()
with open('./global_wb.json', 'w') as f:
    json.dump(store_wbs, f)

new_testset = []
for i, t in tqdm(enumerate(testset)):
    img = t[0].flatten(1).T
    idx = rng.randint(img.shape[1], size=(n,))
    img = img[idx, :].numpy()
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=img, Xt=global_wb)
    transported = torch.from_numpy(ot_sinkhorn.transform(t[0].flatten(1).T.numpy())).reshape(32,32,3).permute(2,0,1)
    new_testset.append((transported, t[1]))

store_json_test = []
for i in tqdm(new_testset):
    store_json_test.append((i[0].tolist(),i[1]))

# store the new testset - which is projected onto the WB as well
with open('./new_testset.json','w') as f:
    json.dump(store_json_test, f)