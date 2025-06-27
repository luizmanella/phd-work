import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import ot
from tqdm import tqdm

# Helper: compute Wasserstein barycenter
def compute_barycenter(n, dists, reg=1e-1):
    A = np.vstack(dists).T
    M = ot.utils.dist0(n)
    M /= M.max()
    weights = np.ones(len(dists)) / len(dists)
    WB = ot.bregman.barycenter(A, M, reg, weights)
    return WB

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, agent_id, train_dataset, test_dataset, device, train_batch_size=64, test_batch_size=128, reg=1e-1, n=1000):
        self.id = agent_id
        self.device = device
        self.model = CNN().to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.reg = reg
        self.n = n

    def compute_local_barycenters(self, n_samples=1000):
        channels = {0: [], 1: [], 2: []}
        rng = np.random.default_rng()
        for img, _ in self.train_dataset:
            flat = img.view(3, -1).numpy()
            for c in range(3):
                idx = rng.choice(flat.shape[1], size=n_samples, replace=False)
                channels[c].append(flat[c, idx])
        self.local_bary = {}
        for c in range(3):
            self.local_bary[c] = compute_barycenter(n_samples, channels[c], self.reg)
        return self.local_bary

    def project_to_global(self, global_bary):
        new_train, new_test = [], []
        rng = np.random.default_rng()
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=self.reg)
        def project(dataset, holder):
            for i, t in tqdm(enumerate(dataset), desc=f"Agent {self.id} projecting", leave=False):
                img, label = t
                flat = img.flatten(1).T
                idx = rng.integers(flat.shape[0], size=self.n)
                sample = flat[idx, :].numpy()
                global_wb = np.stack([global_bary[c] for c in range(3)], axis=1)
                ot_sinkhorn.fit(Xs=sample, Xt=global_wb)
                transported = ot_sinkhorn.transform(flat.numpy())
                img_t = torch.from_numpy(transported).reshape(32,32,3).permute(2,0,1)
                holder.append((img_t, label))
        project(self.train_dataset, new_train)
        project(self.test_dataset, new_test)
        self.train_loader = DataLoader(new_train, batch_size=self.train_loader.batch_size, shuffle=True)
        self.test_loader = DataLoader(new_test, batch_size=self.test_loader.batch_size, shuffle=False)

    def local_train(self, epochs, lr=0.01):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Agent {self.id} Epoch {epoch+1}/{epochs} complete")

    def local_test(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred = torch.max(outputs,1)
                total += labels.size(0)
                correct += (pred==labels).sum().item()
        acc = 100 * correct / total
        print(f"Agent {self.id} Local Accuracy: {acc:.2f}%")
        return acc

class Server:
    def __init__(self, N, device):
        self.N = N
        self.device = device
        self.global_model = CNN().to(device)

    def aggregate(self, participants):
        gdict = self.global_model.state_dict()
        for k in gdict: gdict[k] = torch.zeros_like(gdict[k])
        for a in participants:
            ldict = a.model.state_dict()
            for k in gdict: gdict[k] += ldict[k]
        for k in gdict: gdict[k] /= len(participants)
        self.global_model.load_state_dict(gdict)
        for a in participants: a.model.load_state_dict(gdict)

    def compute_and_broadcast_global_bary(self, local_barys, reg=1e-1):
        global_bary = {}
        for c in range(3):
            dists = [lb[c] for lb in local_barys]
            global_bary[c] = compute_barycenter(len(dists[0]), dists, reg)
        return global_bary

    def test_global(self, test_loader):
        self.global_model.eval()
        correct, total = 0,0
        with torch.no_grad():
            for inputs,labels in test_loader:
                inputs,labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                _, pred = torch.max(outputs,1)
                total += labels.size(0)
                correct += (pred==labels).sum().item()
        print(f"Global Model Accuracy: {100*correct/total:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("OT-Preprocessing algorithm.")
    parser.add_argument("--N", type=int, default=5, help="Total number of agents in the network")
    parser.add_argument("--P", type=int, default=3, help="Number of participating agents per round")
    parser.add_argument("--rounds", type=int, default=10, help="Total federated training rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("-n", type=int, default=1000, help="Number of pixels to subsample")
    parser.add_argument('--reg', type=float, default=1e-1, help='Entropy regularisation for local/global barycenters')
    args = parser.parse_args()

    N, P = args.N, args.P
    reg, n = args.reg, args.n

    rounds, epochs = args.rounds, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    train_idx = np.random.permutation(len(train_set))
    test_idx  = np.random.permutation(len(test_set))
    train_split = np.array_split(train_idx, N)
    test_split  = np.array_split(test_idx, N)

    agents = []
    for i in range(N):
        tr = Subset(train_set, train_split[i])
        te = Subset(test_set, test_split[i])
        agents.append(Agent(i, tr, te, device, red=reg, n=n))
    server = Server(N, device)
    global_test_loader = DataLoader(test_set, batch_size=128)

    # OT alignment
    local_barys = [a.compute_local_barycenters() for a in agents]
    global_bary = server.compute_and_broadcast_global_bary(local_barys)
    for a in agents:
        a.project_to_global(global_bary)

    # Federated training
    for r in range(rounds):
        parts = random.sample(agents, P)
        for a in parts:
            a.local_train(epochs)
        server.aggregate(parts)
        server.test_global(global_test_loader)

