import argparse
import networkx as nx
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import random
from gudhi.wasserstein import wasserstein_distance
from wegl import wasserstein_embedding
import torch
from scipy.stats import wasserstein_distance as ws1d

def pairwise_euclidean(embeddings):
    e = torch.tensor(embeddings)
    diff = e.unsqueeze(1) - e.unsqueeze(0)
    return torch.norm(diff, dim=2).numpy()

def pairwise_wasserstein(embeddings: list):
    E = np.array(embeddings).astype(float)
    N, D = E.shape
    positions = np.arange(D)
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        raw_p = E[i]
        p = np.clip(raw_p, 0, None)
        s = p.sum()
        if s > 0:
            p /= s
        else:
            p = np.ones(D) / D
        for j in range(i, N):
            raw_q = E[j]
            q = np.clip(raw_q, 0, None)
            t = q.sum()
            if t > 0:
                q /= t
            else:
                q = np.ones(D) / D
            wd = ws1d(positions, positions, u_weights=p, v_weights=q)
            W[i, j] = wd
            W[j, i] = wd
    return W

def compute_persistence(distance_matrix, homology_dim=0):
    finite_max = np.max(distance_matrix[np.isfinite(distance_matrix)])
    st = (gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=finite_max).create_simplex_tree(max_dimension=2))
    st.persistence()
    return np.array(st.persistence_intervals_in_dimension(homology_dim))

parser = argparse.ArgumentParser(description="Topological resilience of a weighted complete graph.")
parser.add_argument("--n_nodes",  default=10, type=int, help="Number of nodes in the complete graph")
parser.add_argument("--mean",     default=1,  type=float, help="Mean of the normal distribution for edge weights")
parser.add_argument("--std_dev",  default=10, type=float, help="Standard deviation of the normal distribution for edge weights")
args = parser.parse_args()
n_nodes  = args.n_nodes
mean     = args.mean
std_dev  = args.std_dev
G_complete = nx.complete_graph(n_nodes)

embedding = wasserstein_embedding(G_complete)
baseline_dist_matrix = pairwise_euclidean(embedding) # replace with W-dist
# baseline_dist_matrix = pairwise_wasserstein(embedding)
baseline_persistence = compute_persistence(baseline_dist_matrix)
G_progressive   = G_complete.copy()
edges_remaining = list(G_progressive.edges())
random.shuffle(edges_remaining)
best_distance_so_far = 0.0
resilience_results   = []

for i in range(int(n_nodes * (n_nodes - 1) / 2)):
    if not edges_remaining:
        print("No more edges to remove.")
        break
    edge_to_remove = edges_remaining.pop()
    G_progressive.remove_edge(*edge_to_remove)
    if not nx.is_connected(G_progressive):
        resilience = 0
        print(f"[{i+1}] Removed edge {edge_to_remove} but formed island")
    else:
        try:
            embs = wasserstein_embedding(G_progressive)
            degraded_dist_matrix = pairwise_euclidean(embs)
            # degraded_dist_matrix = pairwise_wasserstein(embs)
            degraded_persistence = compute_persistence(degraded_dist_matrix)
            distance = wasserstein_distance(
                baseline_persistence, degraded_persistence, order=1.0, internal_p=2.0
            )
            best_distance_so_far = max(best_distance_so_far, distance)
            distance_m = best_distance_so_far
            distance_m = distance
            resilience = 1 / (1 + distance_m)
            print(f"[{i+1}] Removed edge {edge_to_remove}")
        except Exception as e:
            print(f"Error during edge removal {edge_to_remove}: {e}")
    resilience_results.append(resilience)
plt.figure(figsize=(10, 5))
plt.plot([r for  r in resilience_results], marker="x")
plt.ylabel("Resilience (%)")
plt.title("Resilience Drop-off per Edge Removal")
plt.tight_layout()
plt.show()
