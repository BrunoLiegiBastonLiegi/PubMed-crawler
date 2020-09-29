from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import sys, random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

G = nx.karate_club_graph()
nx.draw(G, with_labels=True)
pos = nx.drawing.layout.spring_layout(G)
<<<<<<< HEAD
=======
print(pos)
>>>>>>> 191434fbaf47827975d943eddf20dffe2aa69b1e
plt.show()

g = Networkx()
for n in G.nodes:
    [g.add_edge('COEXIST_WITH',n,i) for i in G.neighbors(n)]
nx.draw_networkx(g.g, pos=pos, with_labels=True)
plt.show()


<<<<<<< HEAD
embedding = g.deep_walk(walk_length=80, window=5, walks_per_node=10, embedding_dim=128, with_edges=True)
clusters = g.k_means(n_clusters=2)
colors = [c[1] for c in sorted(clusters.items())]

=======
embedding = g.deep_walk(walk_length=80, window=20, walks_per_node=10, embedding_dim=128)
clusters = g.k_means(n_clusters=2)
print(sorted(clusters.items()))
colors = [c[1] for c in sorted(clusters.items())]
>>>>>>> 191434fbaf47827975d943eddf20dffe2aa69b1e
embedding = np.array([v for v in embedding.values()])
pca = PCA(n_components=2)
pca.fit(embedding)
embedding = pca.transform(embedding)
#embedding = TSNE(n_components=2).fit_transform(embedding)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x=embedding[:,0], y=embedding[:,1])
<<<<<<< HEAD
for i in range(len(embedding)):
    ax.annotate(i, (embedding[i][0], embedding[i][1]))
=======
>>>>>>> 191434fbaf47827975d943eddf20dffe2aa69b1e
plt.show()
    
nx.draw_networkx(g.g, pos=pos, node_color=colors, cmap=plt.cm.tab20, with_labels=True)
plt.show()
