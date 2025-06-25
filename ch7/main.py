import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gudhi import RipsComplex
from gudhi.wasserstein import wasserstein_distance
import ot

# ----------------------------
# Graph Utilities
# ----------------------------
def create_weighted_complete_graph(n):
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.rand()
    return G

def remove_edges(G, k):
    G_sub = G.copy()
    edges = list(G_sub.edges())
    np.random.shuffle(edges)
    G_sub.remove_edges_from(edges[:k])
    return G_sub

def build_weighted_adjacency_matrix(G):
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        A[u, v] = d.get('weight', 1.0)
        A[v, u] = d.get('weight', 1.0)
    np.fill_diagonal(A, 1.0)
    return A

# ----------------------------
# WEGL Embedding Process
# ----------------------------
def diffusion_node_embedding(adj, x0, num_layers=2):
    deg = np.sqrt(adj.sum(axis=1, keepdims=True))
    D = adj / (deg @ deg.T + 1e-8)
    Xs = [x0]
    X = x0.copy()
    for _ in range(num_layers):
        X = D @ X
        Xs.append(X)
    H = np.concatenate(Xs, axis=1)
    return H

def compute_lot_embedding(Z_ref, Z, reg=1e-3):
    M = ot.dist(Z_ref, Z, metric='euclidean')**2
    P = ot.emd(np.ones(Z_ref.shape[0]) / Z_ref.shape[0],
               np.ones(Z.shape[0]) / Z.shape[0],
               M)
    F = P @ Z
    id_ref = Z_ref.copy()
    F_phi = (F - id_ref) / np.sqrt(Z_ref.shape[0])
    return F_phi.flatten()

# ----------------------------
# TDA and Visualization
# ----------------------------
def compute_persistence_diagram(points, max_dim=1):
    rips = RipsComplex(points=points)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    return st.persistence(), st.persistence_intervals_in_dimension(0), st.persistence_intervals_in_dimension(1)

def plot_diagram(intervals, title):
    plt.figure()
    for birth, death in intervals:
        plt.plot([birth], [death], 'bo')
    plt.title(title)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.grid(True)
    plt.show()

# ----------------------------
# Main Function with CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="WEGL-based Network Resilience Quantification")
    parser.add_argument('--nodes', type=int, help='Number of nodes in the complete graph')
    parser.add_argument('--edges-removed', type=int, help='Number of edges to remove')
    parser.add_argument('--embedding-dim', type=int, help='Dimension of initial node features')
    parser.add_argument('--layers', type=int, help='Number of diffusion layers')
    parser.add_argument('--plot', action='store_true', help='Plot persistence diagrams')
    
    args = parser.parse_args()

    print("[INFO] Generating graphs...")
    G_complete = create_weighted_complete_graph(args.nodes)
    G_sub = remove_edges(G_complete, args.edges_removed)

    print("[INFO] Building adjacency matrices...")
    adj_complete = build_weighted_adjacency_matrix(G_complete)
    adj_sub = build_weighted_adjacency_matrix(G_sub)

    print("[INFO] Initializing features...")
    x0 = np.random.rand(args.nodes, args.embedding_dim)

    print("[INFO] Computing WEGL embeddings...")
    Z_complete = diffusion_node_embedding(adj_complete, x0, args.layers)
    Z_sub = diffusion_node_embedding(adj_sub, x0, args.layers)

    print("[INFO] Computing LOT embeddings...")
    embedding_complete = compute_lot_embedding(Z_complete, Z_complete)
    embedding_sub = compute_lot_embedding(Z_complete, Z_sub)

    print("[INFO] Computing persistence diagrams...")
    _, h0_complete, _ = compute_persistence_diagram(Z_complete)
    _, h0_sub, _ = compute_persistence_diagram(Z_sub)

    if args.plot:
        plot_diagram(h0_complete, "Complete Graph - H0")
        plot_diagram(h0_sub, "Subgraph - H0")

    print("[INFO] Calculating Wasserstein distance...")
    dist = wasserstein_distance(h0_complete, h0_sub, order=2, internal_p=2)
    print(f"[RESULT] Wasserstein Distance (Resilience Metric): {dist:.4f}")

if __name__ == "__main__":
    main()
