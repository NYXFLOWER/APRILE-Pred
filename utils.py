import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw(edges, weights, nnn):
    """ edges: ndarray, size=(2, num_edges) """
    G = nx.Graph()
    
    # build graph
    mask = np.where(weights>nnn)[-1]
    edges = edges[:, mask]
    edges = zip(edges[0], edges[1], weights[mask])
    G.add_weighted_edges_from(edges)
    
    # draw graph
    plt.figure(figsize=(10, 8))
    nx.draw_spring(G, with_labels=True, font_weight='bold', )
    plt.savefig("./TIPExplainer/plot.png")

