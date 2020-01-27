import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle


np.random.seed(0)
n_node = 50
n_edge = 200


def draw(edges=np.random.randint(n_node, size=(2, n_edge)),
         weights=np.random.random(size=n_edge),
         nnn=0.5):
    """ edges: ndarray, size=(2, num_edges) """
    G = nx.Graph()

    # build graph
    mask = np.where(weights>nnn)[-1]
    edges = edges[:, mask]
    edges = zip(edges[0], edges[1], weights[mask])
    G.add_weighted_edges_from(edges)

    # draw graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)

    plt.show()
    # nx.draw_spring(G, with_labels=True, font_weight='bold')
    # plt.savefig("./plot/plot.png")


def visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight,
                    protein_name_dict=None, drug_name_dict=None):
    '''
    :param pp_idx: integer tensor of the shape (2, n_pp_edges)
    :param pp_weight: float tensor of the shape (1, n_pp_edges), values within (0,1)
    :param pd_idx: integer tensor of the shape (2, n_pd_edges)
    :param pd_weight: float tensor of the shape (1, n_pd_edges), values within (0,1)
    :param protein_name_dict: store elements {protein_index -> protein name}
    :param drug_name_dict: store elements {drug_index -> drug name}

    1. use different color for pp and pd edges
    2. annotate the weight of each edge near the edge (or annotate with the tranparentness of edges for each edge)
    3. annotate the name of each node near the node, if name_dict=None, then annotate with node's index
    '''
    G = nx.Graph()
    pp_edge, pd_edge = [], []
    p_node, d_node = set(), set()

    if not protein_name_dict:
        tmp = set(pp_idx.flatten()) | set(pd_idx[0])
        protein_name_dict = {i: 'p-'+str(i) for i in tmp}
    if not drug_name_dict:
        drug_name_dict = {i: 'd-'+str(i) for i in set(pd_idx[1])}

        # add pp edges
    for e in zip(pp_idx.T, pp_weight.T):
        t1, t2 = protein_name_dict[e[0][0]], protein_name_dict[e[0][1]]
        G.add_edge(t1, t2, weights=e[1][0])
        pp_edge.append((t1, t2))
        p_node.update([t1, t2])

    # add pd edges
    for e in zip(pd_idx.T, pd_weight.T):
        t1, t2 = protein_name_dict[e[0][0]], drug_name_dict[e[0][1]]
        G.add_edge(t1, t2, weights=e[1][0])
        pd_edge.append((t1, t2))
        p_node.add(t1)
        d_node.add(t2)

    # draw figure
    plt.figure(figsize=(10, 10))

    # draw nodes
    pos = nx.spring_layout(G)
    for p in d_node:  # raise drug nodes positions
        pos[p][1] += 1
    nx.draw_networkx_nodes(G, pos, nodelist=p_node, node_size=500, node_color='y')
    nx.draw_networkx_nodes(G, pos, nodelist=d_node, node_size=500, node_color='blue')

    # draw edges and edge labels
    nx.draw_networkx_edges(G, pos, edgelist=pp_edge, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=pd_edge, width=2, edge_color='g')
    nx.draw_networkx_edge_labels(G, pos, font_size=10,
                                 edge_labels={(u, v): str(d['weights'])[:4] for
                                              u, v, d in G.edges(data=True)})

    # draw node labels
    for p in pos:  # raise text positions
        pos[p][1] += 0.05
    nx.draw_networkx_labels(G, pos, font_size=14)

    plt.savefig()


n_p_node = 10
n_d_node = 10
n_pp_edges = 10
n_pd_edges = 5

with open("./data/index_map/drug-map.pkl", 'rb') as f:
    drug_name_dict = pickle.load(f)
drug_name_dict = {v: 'CID' + str(k) for k, v in drug_name_dict.items()}

with open("./data/index_map/protein-map.pkl", 'rb') as f:
    protein_name_dict = pickle.load(f)
protein_name_dict = {v: 'Gene' + str(k) for k, v in protein_name_dict.items()}

pp_idx = np.random.randint(n_p_node, size=(2, n_pp_edges), )
pp_weight = np.random.random(size=(1, n_pp_edges))
pd_idx = np.random.randint(n_d_node, size=(2, n_pd_edges))
pd_weight = np.random.random(size=(1, n_pd_edges))

# dict-case
visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight, protein_name_dict,
                drug_name_dict)
# non-dict-case
visualize_graph(pp_idx, pp_weight, pd_idx, pd_weight)
