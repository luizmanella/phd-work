import argparse
import networkx as nx
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import random
from gudhi.wasserstein import wasserstein_distance

def get_distance_matrix(graph):
    return np.asarray(nx.floyd_warshall_numpy(graph, weight="weight"), dtype=float)

def compute_persistence(distance_matrix, homology_dim=1):
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

for u, v in G_complete.edges():
    G_complete[u][v]["weight"] = abs(np.random.normal(mean, std_dev))

baseline_dist_matrix = get_distance_matrix(G_complete)
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
    try:
        degraded_dist_matrix   = get_distance_matrix(G_progressive)
        degraded_persistence   = compute_persistence(degraded_dist_matrix)

        distance = wasserstein_distance(
            baseline_persistence, degraded_persistence, order=2.0, internal_p=2.0
        )
        best_distance_so_far = max(best_distance_so_far, distance)
        distance_m = best_distance_so_far
        resilience = 1 / (1 + distance_m)
        resilience_results.append((edge_to_remove, distance_m, resilience))
        print(f"[{i+1}] Removed edge {edge_to_remove}")
    except Exception as e:
        print(f"Error during edge removal {edge_to_remove}: {e}")
resilience_vals = [1] + [r for _, _, r in resilience_results]
plt.figure(figsize=(10, 5))
plt.plot(resilience_vals, marker="x")
plt.ylabel("Resilience (%)")
plt.title("Resilience Drop-off per Edge Removal")
plt.tight_layout()
plt.show()
