from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

g = Networkx(vertices=[Vertex(str(i),[],[]) for i in range(12)])

v = []
v.append(Vertex('0',['CAUSES','CAUSES'],['1','3']))
v.append(Vertex('1',['CAUSES','CAUSES','COEXIST_WITH'],['0','2','4']))
v.append(Vertex('2',['CAUSES','CAUSES','COEXIST_WITH','CAUSES'],['1','3','5','6']))
v.append(Vertex('3',['CAUSES','CAUSES','CAUSES'],['0','2','7']))
v.append(Vertex('4',['COEXIST_WITH','COEXIST_WITH','COEXIST_WITH'],['1','5','8']))
v.append(Vertex('5',['COEXIST_WITH','COEXIST_WITH','COEXIST_WITH','PREVENTS'],['4','9','2','6']))
v.append(Vertex('6',['COEXIST_WITH','CAUSES','PREVENTS','PREVENTS'],['5','2','7','10']))
v.append(Vertex('7',['CAUSES','PREVENTS','PREVENTS'],['3','11','6']))
v.append(Vertex('8',['COEXIST_WITH','COEXIST_WITH'],['4','9']))
v.append(Vertex('9',['COEXIST_WITH','COEXIST_WITH','PREVENTS'],['8','5','10']))
v.append(Vertex('10',['PREVENTS','PREVENTS','PREVENTS'],['9','6','11']))
v.append(Vertex('11',['PREVENTS','PREVENTS'],['10','7']))

[g.add_vertex(i) for i in v]

nx.draw_networkx(g.g, with_labels=True)
plt.show()

embedding = g.deep_walk(walk_length=10, window=5, walks_per_node=10, embedding_dim=128, with_edges=True)
#embedding = g.deep_walk(walk_length=10, window=5, walks_per_node=10, embedding_dim=128, with_edges=False)
clusters = g.k_means(n_clusters=3)


embedding = np.array([v for v in embedding.values()])
pca = PCA(n_components=2)
pca.fit(embedding)
embedding = pca.transform(embedding)
fig, axs = plt.subplots(1,2, figsize=(20,10))
axs[0].scatter(x=embedding[:,0], y=embedding[:,1])
for i in range(len(embedding)):
    axs[0].annotate(i, (embedding[i][0], embedding[i][1]))

nx.draw_networkx(g.g, ax=axs[1], node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
plt.show()
