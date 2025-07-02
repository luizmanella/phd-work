import networkx as nx
import numpy as np
from sklearn.manifold import MDS
import ot
import warnings


def wasserstein_embedding(G, dim=2, weight_attr='weight', eps=1e-8):
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected.")
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    dist = np.zeros((n, n))
    use_weighted = False
    if weight_attr:
        try:
            first_edge = next(iter(G.edges(data=True)))
            use_weighted = weight_attr in first_edge[2]
        except StopIteration:
            use_weighted = False

    if use_weighted:
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight_attr))
    else:
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # Fill distance matrix
    for u, lengths in path_lengths.items():
        i = idx[u]
        for v, d in lengths.items():
            j = idx[v]
            dist[i, j] = d
    sorted_dists = np.sort(dist, axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            u = np.ones(n) / n
            v = np.ones(n) / n
            w = ot.wasserstein_1d(sorted_dists[i], sorted_dists[j])
            W[i, j] = W[j, i] = w

    W = W + np.eye(n) * eps

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(W)

    # Map nodes to embeddings
    embedding = {node: coords[idx[node]] for node in nodes}
    return [v.tolist() for v in embedding.values()]
