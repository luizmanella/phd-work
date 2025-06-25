import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import ot

def compute_wasserstein_barycenter(images, reg):
    if isinstance(images, list):
        images = np.stack(images, axis=0)
    n_images, h, w = images.shape
    distributions = []
    for img in images:
        distributions.append((img / img.sum()).flatten())
    distributions = np.array(distributions).T
    xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    C = ot.utils.dist(coords, coords)
    C /= C.max()
    bary = ot.bregman.barycenter(distributions, C, reg)
    return bary.reshape((h, w))

def compute_wasserstein_distance(img, bary, reg):
    x = img.squeeze().cpu().numpy().astype(np.float64)
    y = bary if isinstance(bary, np.ndarray) else bary.squeeze().cpu().numpy().astype(np.float64)
    h, w = x.shape
    a = x.flatten(); a /= a.sum()
    b = y.flatten(); b /= b.sum()
    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    C = ot.utils.dist(coords, coords)
    C /= C.max()
    cost = ot.sinkhorn2(a, b, C, reg)[0]
    return float(np.sqrt(cost))

class Agent:
    def __init__(self, dataset):
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
        self.local_barycenters = {}

    def group_by_class(self):
        class_dict = {i: [] for i in range(10)}
        images, labels = next(iter(self.train_loader))
        for img, label in zip(images, labels):
            arr = img.squeeze().cpu().numpy()
            class_dict[int(label)].append(arr)
        return class_dict

    def compute_local_barycenters(self, bary_reg):
        grouped = self.group_by_class()
        for label, imgs in grouped.items():
            self.local_barycenters[label] = compute_wasserstein_barycenter(imgs, reg=bary_reg) if imgs else None
        return self.local_barycenters

    def infer_and_evaluate(self, global_barycenters, dist_reg):
        images, labels = next(iter(self.test_loader))
        preds = []
        for img in images:
            dists = []
            for lbl, bary in global_barycenters.items():
                dists.append(compute_wasserstein_distance(img, bary, reg=dist_reg))
            preds.append(int(np.argmin(dists)))
        preds = torch.tensor(preds)
        return (preds == labels).float().mean().item()

class Server:
    def __init__(self, num_agents, data_root, bary_reg, dist_reg):
        self.num_agents = num_agents
        self.bary_reg = bary_reg
        self.dist_reg = dist_reg
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=self.transform)
        self.agents = []
        self.global_barycenters = {}

    def setup_agents(self):
        lengths = [len(self.dataset) // self.num_agents] * self.num_agents
        for i in range(len(self.dataset) - sum(lengths)): lengths[i] += 1
        subsets = random_split(self.dataset, lengths)
        self.agents = [Agent(sub) for sub in subsets]

    def collect_and_aggregate(self):
        collected = {i: [] for i in range(10)}
        with ThreadPoolExecutor(max_workers=self.num_agents) as exe:
            futures = [exe.submit(agent.compute_local_barycenters, self.bary_reg) for agent in self.agents]
            for f in as_completed(futures):
                for lbl, bary in f.result().items():
                    if bary is not None: collected[lbl].append(bary)
        for lbl, blist in collected.items():
            self.global_barycenters[lbl] = compute_wasserstein_barycenter(blist, reg=self.bary_reg) if blist else None

    def evaluate_global(self):
        accuracies = {}
        with ThreadPoolExecutor(max_workers=self.num_agents) as exe:
            futures = {exe.submit(agent.infer_and_evaluate, self.global_barycenters, self.dist_reg): idx
                       for idx, agent in enumerate(self.agents)}
            for f in as_completed(futures): accuracies[futures[f]] = f.result()
        avg = sum(accuracies.values()) / len(accuracies)
        return accuracies, avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Wasserstein MNIST")
    parser.add_argument("--num_agents", type=int, help="Number of federated agents")
    parser.add_argument("--data_root", type=str, default='./data', help="MNIST dataset root")
    parser.add_argument("--bary_reg", type=float, help="Regularization for barycenter")
    parser.add_argument("--dist_reg", type=float, help="Regularization for distance")
    args = parser.parse_args()

    server = Server(args.num_agents, args.data_root, args.bary_reg, args.dist_reg)
    server.setup_agents()
    server.collect_and_aggregate()
    accuracies, avg_accuracy = server.evaluate_global()
    for idx, acc in accuracies.items(): print(f"Agent {idx}: accuracy = {acc:.4f}")
    print(f"Average accuracy: {avg_accuracy:.4f}")