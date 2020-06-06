from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import sys, random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

G = nx.karate_club_graph()
nx.draw(G, with_labels=True)
plt.show()

vertices = []
for e in G.edges():
    #dir = random.choice(['straight', 'inverted', 'bi'])
    dir = 'bi'
    #label = random.choice(['hates', 'likes'])
    label = ' '
    if dir == 'inverted':
        vertices.append(Vertex(str(e[1]), [label], [str(e[0])]))
    else:
        vertices.append(Vertex(str(e[0]), [label], [str(e[1])]))
        if dir == 'bi':
            vertices.append(Vertex(str(e[1]), [label], [str(e[0])]))

#g = Graph_tool(vertices=vertices)
g = Networkx(vertices=vertices)
#g.draw()

embedding = g.deep_walk(window=3, walks_per_node=80)
clusters = g.k_means(elbow_range=(2,20))
#colors = []
embedding = np.array([v for v in embedding.values()])
#pca = PCA(n_components=50)
#pca.fit(embedding)
#embedding = pca.transform(embedding)
embedding = TSNE(n_components=2).fit_transform(embedding)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x=embedding[:,0], y=embedding[:,1])
plt.show()

for k,v in clusters.items():
    print(k,':',v)
    #colors.append(v)
    
nx.draw_networkx(g.g, node_color=[c for c in clusters.values()], cmap=plt.cm.tab20, with_labels=True)
plt.show()
