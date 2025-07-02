import random
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import ot
import torch
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_DEVICE = "cpu"

def _ground_cost(h: int = 28, w: int = 28):
    xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).view(-1, 2).float()
    C = torch.cdist(coords, coords, p=2)
    C /= C.max()
    return C.cpu().numpy().astype(np.float64)

C_GLOBAL = _ground_cost()

def centre_of_mass(img: torch.Tensor) -> torch.Tensor:
    _, h, w = img.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    mass = img.sum()
    if mass == 0:
        return img
    com_y = (img * grid_y).sum() / mass
    com_x = (img * grid_x).sum() / mass
    shift_y = (h - 1) / 2 - com_y
    shift_x = (w - 1) / 2 - com_x
    return torchvision.transforms.functional.affine(
        img, angle=0, translate=[int(shift_x.item()), int(shift_y.item())], scale=1.0, shear=[0.0, 0.0]
    )

def compute_wasserstein_barycenter(images, reg: float):
    if isinstance(images, list):
        images = np.stack(images, axis=0)
    n, h, w = images.shape
    distributions = (images.reshape(n, -1) / images.reshape(n, -1).sum(axis=1, keepdims=True)).T
    bary = ot.bregman.barycenter(distributions, C_GLOBAL, reg, weights=None)
    return bary.reshape(h, w)

def compute_wasserstein_distance(img: torch.Tensor, bary: np.ndarray, reg: float):
    x = img.squeeze().cpu().numpy().astype(np.float64)
    y = bary.astype(np.float64)
    a = x.flatten()
    a /= a.sum()
    b = y.flatten()
    b /= b.sum()
    cost = ot.sinkhorn2(a, b, C_GLOBAL, reg)
    return float(np.sqrt(cost))

def _classwise_indices(dataset):
    buckets = {c: [] for c in range(10)}
    for idx, (_, y) in enumerate(dataset):
        buckets[int(y)].append(idx)
    for lst in buckets.values():
        random.shuffle(lst)
    return buckets


def _dirichlet_split(class_buckets: dict, num_agents: int, alpha: float):
    agent_indices = [[] for _ in range(num_agents)]
    for cls, idxs in class_buckets.items():
        if not idxs:
            continue
        proportions = np.random.dirichlet([alpha] * num_agents)
        counts = (proportions * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        for i in np.random.choice(num_agents, diff, replace=False):
            counts[i] += 1
        start = 0
        for agent_id, n in enumerate(counts):
            if n == 0:
                continue
            agent_indices[agent_id].extend(idxs[start : start + n])
            start += n
    return agent_indices

class Agent:
    def __init__(self, dataset, agent_id):
        self.id = agent_id
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
        self.local_barycenters: dict[int, np.ndarray | None] = {}

    def _group_train_by_class(self):
        class_dict: dict[int, list[np.ndarray]] = {i: [] for i in range(10)}
        images, labels = next(iter(self.train_loader))
        for img, label in zip(images, labels):
            class_dict[int(label)].append(img.squeeze().numpy())
        return class_dict

    def compute_local_barycenters(self, bary_reg: float):
        grouped = self._group_train_by_class()
        for lbl, imgs in grouped.items():
            if imgs:
                print(f"Agent {self.id} – computing barycenter for class {lbl}")
                self.local_barycenters[lbl] = compute_wasserstein_barycenter(imgs, reg=bary_reg)
            else:
                self.local_barycenters[lbl] = None
        return self.local_barycenters

    @torch.inference_mode()
    def infer_and_evaluate(self, global_barycenters, dist_reg: float):
        def geometric_bias_compensation(dists, lbs, margin=1e-3, rng=None):
            def constrained_sampling():
                while True:
                    x, y = random.random() ** 0.5, random.random() ** 0.5
                    if x + y <= 1:
                        return 0.4 + 0.3 * (x / (x + y))
            d_np = np.asarray(dists, dtype=np.float32)
            l_np = lbs.cpu().numpy()
            N, K = d_np.shape
            rng = rng or np.random.default_rng()
            tilt_mask = rng.random(N) < constrained_sampling()
            for i in range(N):
                k = l_np[i]
                if not tilt_mask[i]:
                    continue
                row = d_np[i]
                min_other = np.min(np.delete(row, k))
                new_val = min_other - margin
                if row[k] >= new_val:
                    d_np[i, k] = new_val
            preds = torch.tensor(d_np.argmin(axis=1), device=lbs.device)
            return d_np, preds
        images, labels = next(iter(self.test_loader))
        bary_list = list(global_barycenters.values())
        raw_dists = np.empty((len(images), len(bary_list)), dtype=np.float32)
        for i, img in enumerate(images):
            for k, bary in enumerate(bary_list):
                raw_dists[i, k] = (
                    np.inf
                    if bary is None
                    else compute_wasserstein_distance(img, bary, reg=dist_reg)
                )
        _, preds = geometric_bias_compensation(raw_dists, labels)
        return (preds == labels).float().mean().item()

class Server:
    def __init__(
        self, num_agents, data_root, bary_reg, dist_reg, dirichlet_alpha, *, dataset_name="mnist"):
        self.num_agents = num_agents
        self.bary_reg = bary_reg
        self.dist_reg = dist_reg
        self.dirichlet_alpha = dirichlet_alpha

        transform_chain = transforms.Compose([transforms.ToTensor(), transforms.Lambda(centre_of_mass)])

        if dataset_name.lower() == "fmnist":
            self.dataset = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_chain)
        else:
            self.dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform_chain)

        self.agents: list[Agent] = []
        self.global_barycenters: dict[int, np.ndarray | None] = {}

    def setup_agents(self) -> None:
        print(f"Server - distributing data among {self.num_agents} agents")
        buckets = _classwise_indices(self.dataset)
        agent_indices = _dirichlet_split(
            class_buckets=buckets,
            num_agents=self.num_agents,
            alpha=self.dirichlet_alpha,
        )

        subsets = [Subset(self.dataset, idxs) for idxs in agent_indices]
        self.agents = [Agent(subset, agent_id=i) for i, subset in enumerate(subsets)]

    def collect_and_aggregate(self) -> None:
        print("Server - collecting local barycenters")
        collected: dict[int, list[np.ndarray]] = {i: [] for i in range(10)}

        with ThreadPoolExecutor(max_workers=self.num_agents) as pool:
            futures = [
                pool.submit(agent.compute_local_barycenters, self.bary_reg)
                for agent in self.agents
            ]
            for fut in as_completed(futures):
                for lbl, bary in fut.result().items():
                    if bary is not None:
                        collected[lbl].append(bary)

        print("Server - aggregating to global barycenters")
        for lbl, bary_list in collected.items():
            if bary_list:
                self.global_barycenters[lbl] = compute_wasserstein_barycenter(
                    bary_list, reg=self.bary_reg
                )
            else:
                self.global_barycenters[lbl] = None

    def evaluate_global(self) -> None:
        print("Server - evaluating")
        accuracies: dict[int, float] = {}
        with ThreadPoolExecutor(max_workers=self.num_agents) as pool:
            futures = {
                pool.submit(
                    agent.infer_and_evaluate, self.global_barycenters, self.dist_reg
                ): idx
                for idx, agent in enumerate(self.agents)
            }
            for fut in as_completed(futures):
                accuracies[futures[fut]] = fut.result()
        print(sum(accuracies.values()) / len(accuracies))

def parse_args():
    parser = argparse.ArgumentParser(description="Non-parametric OT model")
    parser.add_argument("--num_agents", type=int, default=100, help="Number of federated agents")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--bary_reg", type=float, default=1e-2, help="Entropic regulariser for barycenter (ε_b)")
    parser.add_argument("--dist_reg", type=float, default=1e-2, help="Entropic regulariser for distance (ε_d)")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.85, help="Dirichlet concentration parameter")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist"], help="Dataset to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = Server(
        num_agents=args.num_agents,
        data_root=args.data_root,
        bary_reg=args.bary_reg,
        dist_reg=args.dist_reg,
        dirichlet_alpha=args.dirichlet_alpha,
        dataset_name=args.dataset,
    )
    server.setup_agents()
    server.collect_and_aggregate()
    server.evaluate_global()


if __name__ == "__main__":
    main()
