import numpy as np
import pandas as pd
from sir import sir_calculate_node_scores


content_path = "attribute networks/cora/cora.content"
cites_path = "attribute networks/cora/cora.cites"

content = pd.read_csv(content_path, sep='\t', header=None)
paper_ids = content.iloc[:, 0].values
features = content.iloc[:, 1:-1].astype(float).values
id_map = {id_: i for i, id_ in enumerate(paper_ids)}
edges = pd.read_csv(cites_path, sep='\t', header=None)
src = edges.iloc[:, 0].values
dst = edges.iloc[:, 1].values

N = len(paper_ids)
adj_matrix = np.zeros((N, N), dtype=int)
missing_count = 0
for cited, citing in zip(src, dst):
    if cited in id_map and citing in id_map:
        i, j = id_map[cited], id_map[citing]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    else:
        missing_count += 1
np.fill_diagonal(adj_matrix, 0)
sir_labels = sir_calculate_node_scores(adj_matrix)
np.save("attribute networks/cora/adj_matrix.npy", adj_matrix)
np.save("attribute networks/cora/features.npy", features)
np.save("attribute networks/cora/sir_labels.npy", sir_labels)

