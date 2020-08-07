from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

v = []
v.append(Vertex('0',['causes'], ['1']))
v.append(Vertex('1',['prevents'], ['2']))
v.append(Vertex('2',['causes'], ['3']))
v.append(Vertex('3',['prevents'], ['0']))


g = Networkx(vertices=v)
g.draw()

wedges = False

embedding = g.deep_walk(walk_length=20, window=7, walks_per_node=160, embedding_dim=128, with_edges=wedges)
clusters = g.k_means(n_clusters=2)

evecs = np.array([v for v in embedding.values()])
pca = PCA(n_components=2)
pca.fit(evecs)
evecs = pca.transform(evecs)
fig, axs = plt.subplots(1,2, figsize=(20,10))
axs[0].scatter(x=evecs[:,0], y=evecs[:,1])
for i in g.get_vertices():
    axs[0].annotate(g.get_vertex(i), (evecs[i][0], evecs[i][1]))

nx.draw_networkx(g.g, ax=axs[1], labels={i:g.get_vertex(i) for i in g.get_vertices()}, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
plt.show()
