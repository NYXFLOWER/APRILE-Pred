from .. import utils
import numpy as np

np.random.seed(0)
n_node = 50
n_edge = 200

edges = np.random.randint(n_node, size=(2, n_edge))
weights = np.random.random(size=n_edge)
nnn = 0.5

utils.draw(edges, weights, nnn)