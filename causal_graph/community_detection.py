from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


v = []
'''
v.append(Vertex('1',['facebook','email','telephone'],['2','2','3']))
v.append(Vertex('2',['facebook','facebook','facebook','facebook','email','email','telephone'],['1','3','4','5','3','1','5']))
v.append(Vertex('3',['facebook','email','telephone','telephone'],['2','2','1','4']))
v.append(Vertex('4',['facebook','email','email','telephone'],['2','5','6','3']))
v.append(Vertex('5',['facebook','facebook','facebook','email','email','email','telephone'],['6','7','2','6','7','4','2']))
v.append(Vertex('6',['facebook','facebook','email','email','email','telephone'],['5','8','5','8','4','8']))
v.append(Vertex('7',['facebook','facebook','email','email','telephone'],['5','8','5','8','8']))
v.append(Vertex('8',['facebook','facebook','email','email','telephone','telephone'],['7','6','7','6','7','6']))
'''


v.append(Vertex('0',['facebook' for i in range(4)]+['mail' for i in range(3)],['1','2','3','4']+['1','2','3']))
v.append(Vertex('1',['facebook' for i in range(4)]+['mail' for i in range(3)],['0','2','3','5']+['0','2','3']))
v.append(Vertex('2',['facebook' for i in range(4)]+['mail' for i in range(3)],['0','1','3','6']+['1','0','3']))
v.append(Vertex('3',['facebook' for i in range(4)]+['mail' for i in range(3)],['1','2','0','7']+['1','2','0']))
v.append(Vertex('4',['facebook']+['mail' for i in range(4)],['0']+['5','6','7','8']))
v.append(Vertex('5',['facebook']+['mail' for i in range(4)],['1']+['4','6','7','9']))
v.append(Vertex('6',['facebook']+['mail' for i in range(4)],['2']+['5','4','7','10']))
v.append(Vertex('7',['facebook']+['mail' for i in range(4)],['3']+['5','6','4','11']))
v.append(Vertex('8',['facebook' for i in range(3)]+['mail'],['9','10','11']+['4']))
v.append(Vertex('9',['facebook' for i in range(3)]+['mail'],['8','10','11']+['5']))
v.append(Vertex('10',['facebook' for i in range(3)]+['mail'],['9','8','11']+['6']))
v.append(Vertex('11',['facebook' for i in range(3)]+['mail'],['9','10','8']+['7']))



g = Networkx(vertices=v)
g.draw(edge_label='type')


wedges = True

embedding = g.deep_walk(walk_length=80, window=8, walks_per_node=10, embedding_dim=128, with_edges=wedges)
clusters = g.k_means(n_clusters=3)

evecs = np.array([v for v in embedding.values()])
pca = PCA(n_components=2)
pca.fit(evecs)
evecs = pca.transform(evecs)
fig, axs = plt.subplots(1,2, figsize=(20,10))
axs[0].scatter(x=evecs[:,0], y=evecs[:,1])
i = 0
for k in embedding.keys():
    axs[0].annotate(k, (evecs[i][0], evecs[i][1]))
    i += 1

nx.draw_networkx(g.g, ax=axs[1], labels={i:g.get_vertex(i) for i in g.get_vertices()}, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
plt.show()
