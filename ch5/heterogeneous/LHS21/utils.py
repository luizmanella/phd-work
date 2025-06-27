import os
import logging
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from model import *  # noqa: F401,F403  (repo-local models)
from datasets import (                # repo-local dataset wrappers
    CIFAR10_truncated,
    CIFAR100_truncated,
    ImageFolder_custom,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
#  Helpers                                                                   #
# ---------------------------------------------------------------------------#
def mkdirs(dirpath: str) -> None:
    """Create `dirpath` if it doesn’t exist (no error if it does)."""
    os.makedirs(dirpath, exist_ok=True)


# ---------------------------------------------------------------------------#
#  Dataset loading helpers                                                   #
# ---------------------------------------------------------------------------#
def _basic_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def load_cifar10_data(datadir: str):
    t = _basic_transform()
    train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=t)
    test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=t)

    return train_ds.data, train_ds.target, test_ds.data, test_ds.target


def load_cifar100_data(datadir: str):
    t = _basic_transform()
    train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=t)
    test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=t)

    return train_ds.data, train_ds.target, test_ds.data, test_ds.target


def load_tinyimagenet_data(datadir: str):
    t = _basic_transform()
    train_ds = ImageFolder_custom(os.path.join(datadir, "train"), transform=t)
    test_ds = ImageFolder_custom(os.path.join(datadir, "val"), transform=t)

    X_train = np.array([s[0] for s in train_ds.samples])
    y_train = np.array([int(s[1]) for s in train_ds.samples])
    X_test = np.array([s[0] for s in test_ds.samples])
    y_test = np.array([int(s[1]) for s in test_ds.samples])
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------#
#  Data-partition helper                                                     #
# ---------------------------------------------------------------------------#
def record_net_data_stats(y_train: np.ndarray,
                          net_dataidx_map: Dict[int, np.ndarray]) -> Dict[int, Dict[int, int]]:
    net_cls_counts: Dict[int, Dict[int, int]] = {}
    for net_i, idxs in net_dataidx_map.items():
        labels, counts = np.unique(y_train[idxs], return_counts=True)
        net_cls_counts[net_i] = {int(l): int(c) for l, c in zip(labels, counts)}

    sizes = [sum(cls_cnt.values()) for cls_cnt in net_cls_counts.values()]
    logger.info("Client data sizes — mean: %.2f  std: %.2f", np.mean(sizes), np.std(sizes))
    logger.info("Per-client class counts: %s", net_cls_counts)
    return net_cls_counts


def partition_data(dataset: str,
                   datadir: str,
                   partition: str,
                   n_parties: int,
                   beta: float = 0.4):
    if dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        K = 10
    elif dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
        K = 100
    elif dataset == "tinyimagenet":
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
        K = 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    n_train = len(y_train)

    # IID / homogeneous
    if partition in {"homo", "iid"}:
        idxs = np.random.permutation(n_train)
        splits = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: split for i, split in enumerate(splits)}

    # Dirichlet label partition (non-IID)
    elif partition in {"noniid", "noniid-labeldir"}:
        min_size = 0
        min_required = 10
        N = y_train.shape[0]

        while min_size < min_required:
            idx_batch: List[List[int]] = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties)
                                        for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                for idx_j, idx in zip(idx_batch, np.split(idx_k, cuts)):
                    idx_j += idx.tolist()
                min_size = min(len(idx_j) for idx_j in idx_batch)

        net_dataidx_map = {j: np.array(idxs) for j, idxs in enumerate(idx_batch)}

    else:
        raise ValueError(f"Unknown partition: {partition}")

    stats = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, stats


# ---------------------------------------------------------------------------#
#  Federated-learning helpers                                                #
# ---------------------------------------------------------------------------#
def get_trainable_parameters(net: nn.Module, device: str = "cpu") -> torch.Tensor:
    trainable = [p for p in net.parameters() if p.requires_grad]
    total_elems = sum(p.numel() for p in trainable)
    flat = torch.empty(total_elems, dtype=torch.float64, device=device)
    offset = 0
    for p in trainable:
        n = p.numel()
        flat[offset: offset + n].copy_(p.data.view(-1).double())
        offset += n
    return flat


def put_trainable_parameters(net: nn.Module, flat: torch.Tensor):
    trainable = [p for p in net.parameters() if p.requires_grad]
    offset = 0
    for p in trainable:
        n = p.numel()
        p.data.copy_(flat[offset: offset + n].view_as(p))
        offset += n


# ---------------------------------------------------------------------------#
#  Accuracy / loss utilities                                                 #
# ---------------------------------------------------------------------------#
@torch.no_grad()
def compute_accuracy(model: nn.Module,
                     dataloaders,
                     device: torch.device = torch.device("cpu"),
                     get_confusion_matrix: bool = False):
    single_loader = not isinstance(dataloaders, (list, tuple))
    if single_loader:
        dataloaders = [dataloaders]

    was_training = model.training
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    total, correct, losses = 0, 0, []
    true_all, pred_all = [], []

    for loader in dataloaders:
        for x, target in loader:
            x, target = x.to(device), target.long().to(device)
            _, _, out = model(x)
            loss = criterion(out, target)
            losses.append(loss.item())

            _, pred = out.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

            true_all.extend(target.cpu().numpy())
            pred_all.extend(pred.cpu().numpy())

    if was_training:
        model.train()

    avg_loss = float(np.mean(losses))
    acc = correct / float(total)
    if get_confusion_matrix:
        cm = confusion_matrix(true_all, pred_all)
        return acc, cm, avg_loss
    return acc, avg_loss


@torch.no_grad()
def compute_loss(model: nn.Module,
                 dataloader,
                 device: torch.device = torch.device("cpu")) -> float:
    was_training = model.training
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)
    losses = []
    for x, target in dataloader:
        x, target = x.to(device), target.long().to(device)
        _, _, out = model(x)
        losses.append(criterion(out, target).item())

    if was_training:
        model.train()

    return float(np.mean(losses))


# ---------------------------------------------------------------------------#
#  Model save / load                                                         #
# ---------------------------------------------------------------------------#
def save_model(model: nn.Module, model_index: int, modeldir: str):
    path = os.path.join(modeldir, f"trained_local_model{model_index}")
    torch.save(model.state_dict(), path)
    logger.info("Saved model to %s", path)


def load_model(model: nn.Module, model_index: int, modeldir: str,
               device: torch.device = torch.device("cpu")) -> nn.Module:
    path = os.path.join(modeldir, f"trained_local_model{model_index}")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


# ---------------------------------------------------------------------------#
#  DataLoader creator (with optional OT projector)                           #
# ---------------------------------------------------------------------------#
def _insert_projector(trans: transforms.Compose, projector):
    if projector is not None:
        # insert right before the final normalisation (assume last transform)
        for i, t in enumerate(trans.transforms[::-1], 1):
            if isinstance(t, transforms.Normalize):
                trans.transforms.insert(len(trans.transforms) - i, projector)
                break


def get_dataloader(dataset: str,
                   datadir: str,
                   train_bs: int,
                   test_bs: int,
                   dataidxs=None,
                   noise_level: float = 0.0,
                   projector=None):
    if dataset not in {"cifar10", "cifar100", "tinyimagenet"}:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if dataset == "cifar10":
        dl_cls = CIFAR10_truncated
        mean = [x / 255.0 for x in (125.3, 123.0, 113.9)]
        std = [x / 255.0 for x in (63.0, 62.1, 66.7)]
        normalize = transforms.Normalize(mean, std)

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze()
            ),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    elif dataset == "cifar100":
        dl_cls = CIFAR100_truncated
        normalize = transforms.Normalize(
            mean=[0.50707516, 0.48654887, 0.44091784],
            std=[0.26733429, 0.25643846, 0.27615047],
        )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    else:  # tinyimagenet
        dl_cls = ImageFolder_custom
        normalize = transforms.Normalize((0.5,)*3, (0.5,)*3)
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    # insert OT projector if provided
    _insert_projector(transform_train, projector)
    _insert_projector(transform_test, projector)

    # build datasets / loaders
    if dataset == "tinyimagenet":
        train_path = os.path.join(datadir, "train")
        val_path = os.path.join(datadir, "val")
        train_ds = dl_cls(train_path, dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_cls(val_path, transform=transform_test)
    else:
        train_ds = dl_cls(datadir, dataidxs=dataidxs, train=True,
                          transform=transform_train, download=True)
        test_ds = dl_cls(datadir, train=False,
                         transform=transform_test, download=True)

    train_dl = data.DataLoader(train_ds, batch_size=train_bs,
                               shuffle=True, drop_last=True)
    test_dl = data.DataLoader(test_ds, batch_size=test_bs, shuffle=False)
    return train_dl, test_dl, train_ds, test_ds
