import numpy as np
import networkx as nx
from node2vec import Node2Vec

adj = np.load("synthetic_data/1000ba_adj_matrix.npy")
G = nx.from_numpy_array(adj)
node2vec = Node2Vec(G, dimensions=128, walk_length=20, num_walks=100, workers=1,seed=42)
model = node2vec.fit(window=10, min_count=1)
embeddings = np.array([model.wv[str(i)] for i in range(G.number_of_nodes())])
np.save("synthetic_data/1000ba_xs.npy", embeddings)
