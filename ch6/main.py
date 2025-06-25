import argparse
import networkx as nx
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import random
from gudhi.wasserstein import wasserstein_distance

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser(description="Topological resilience of a weighted complete graph.")
parser.add_argument("--n_nodes", type=int, help="Number of nodes in the complete graph")
parser.add_argument("--mean", type=float, help="Mean of the normal distribution for edge weights")
parser.add_argument("--std_dev", type=float, help="Standard deviation of the normal distribution for edge weights")
args = parser.parse_args()

n_nodes = args.n_nodes
mean = args.mean
std_dev = args.std_dev

# -------------------------
# Initialize Complete Graph
# -------------------------

G_complete = nx.complete_graph(n_nodes)
for u, v in G_complete.edges():
    G_complete[u][v]['weight'] = abs(np.random.normal(mean, std_dev))

# -------------------------
# Compute Baseline Persistence Diagram
# -------------------------
def get_distance_matrix(graph):
    return np.asarray(nx.to_numpy_matrix(graph, weight='weight'))

def compute_persistence(distance_matrix):
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=1.5)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree.persistence()

baseline_dist_matrix = get_distance_matrix(G_complete)
baseline_persistence = compute_persistence(baseline_dist_matrix)

# -------------------------
# Progressive Edge Removal (No Replacement)
# -------------------------
G_progressive = G_complete.copy()
edges_remaining = list(G_progressive.edges())
random.shuffle(edges_remaining)

resilience_results = []

for i in range(20):
    if not edges_remaining:
        print("No more edges to remove.")
        break

    edge_to_remove = edges_remaining.pop()
    G_progressive.remove_edge(*edge_to_remove)

    try:
        degraded_dist_matrix = get_distance_matrix(G_progressive)
        degraded_persistence = compute_persistence(degraded_dist_matrix)

        distance = wasserstein_distance(baseline_persistence, degraded_persistence, order=2., internal_p=2.)
        resilience = (1 / distance) * 100 if distance > 1e-6 else float('inf')

        resilience_results.append((edge_to_remove, distance, resilience))
        print(f"[{i+1}] Removed edge {edge_to_remove}, Distance: {distance:.4f}, Resilience: {resilience:.2f}%")

    except Exception as e:
        print(f"Error during edge removal {edge_to_remove}: {e}")

# -------------------------
# Plot Resilience Over Removals
# -------------------------
edges_removed = [f"{u}-{v}" for (u, v, _) in resilience_results]
resilience_vals = [r for _, _, r in resilience_results]

plt.figure(figsize=(10, 5))
plt.bar(edges_removed, resilience_vals)
plt.xticks(rotation=45)
plt.ylabel("Resilience (%)")
plt.title("Resilience After Progressive Edge Removal")
plt.tight_layout()
plt.show()
