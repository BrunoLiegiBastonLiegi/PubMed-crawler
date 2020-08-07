from graph import Graph, Vertex, Networkx, Graph_tool, MultilayerGraph, Modularity
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import requests as req
from bs4 import BeautifulSoup

r = req.get('http://www.casos.cs.cmu.edu/computational_tools/datasets/external/padgett/padgett.xml')

soup = BeautifulSoup(r.text, "xml")

edges = soup.find_all('network')

marriage = edges[0].find_all('link', value='1.0000')
business = edges[1].find_all('link', value='1.0000')

g = MultilayerGraph(['marriage','business'])
g.l = 0.3

v = []

for e in marriage:
    src = e['source'] if list(e['source'])[-1] == 'I' else e['source']+'I'
    trg = e['target'] if list(e['target'])[-1] == 'I' else e['target']+'I'
    g.add_edge('marriage', src, trg)
    v.append(Vertex(src, ['marriage'], [trg]))

for e in business:
    src = e['source'] if list(e['source'])[-1] == 'I' else e['source']+'I'
    trg = e['target'] if list(e['target'])[-1] == 'I' else e['target']+'I'
    g.add_edge('business', src, trg)
    v.append(Vertex(src, ['business'], [trg]))

g = Networkx(vertices=v)

#pos = nx.spring_layout(g.g)
#pos = {0: np.array([ 0.2920545 , -0.90126149]), 1: np.array([-0.04862197, -0.41444639]), 2: np.array([ 0.02620048, -0.13971469]), 3: np.array([-0.39191525, -0.27251725]), 4: np.array([0.36279274, 0.3233072 ]), 5: np.array([-0.28078161,  0.07785087]), 6: np.array([-0.28976652,  0.49279886]), 7: np.array([0.28151077, 0.71675766]), 8: np.array([-0.09113253,  0.5945853 ]), 9: np.array([-0.03083951,  0.76460052]), 10: np.array([0.17588439, 0.56412292]), 11: np.array([ 0.31584407, -0.17766321]), 12: np.array([0.10644024, 0.13752079]), 13: np.array([-0.24509931, -0.76594106]), 14: np.array([-0.1825705, -1.       ])}


#for e in g.get_edges():
    #print(g.get_vertex(e[0]), ' ', g.get_vertex(e[1]), ' ', e[2])

g.draw()

wedges = True

embedding = g.deep_walk(walk_length=80, window=5, walks_per_node=80, embedding_dim=128, with_edges=wedges)
#embedding = g.deep_walk(walk_length=80, window=4, walks_per_node=80, embedding_dim=128)
clusters = g.k_means(n_clusters=4)
for i in range(4):
    g.draw_cluster([i])

g.draw_embedding()
plt.show()

#evecs = np.array([v for v in embedding.values()])
#pca = PCA(n_components=2)
#pca.fit(evecs)
#evecs = pca.transform(evecs)
#fig, axs = plt.subplots(1,2, figsize=(20,10))
#axs[0].scatter(x=evecs[:,0], y=evecs[:,1])
#i = 0
#for k in embedding.keys():
    #axs[0].annotate(k, (evecs[i][0], evecs[i][1]))
    #i += 1
    
#nx.draw_networkx(g.g, ax=axs[1], pos=pos, labels={i:g.get_vertex(i) for i in g.get_vertices()}, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
#nx.draw_networkx(g.G['marriage'], ax=axs[1], node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)

#g.draw()
#plt.show()



#m = Modularity(A=g.adjacency_matrix(), M=2, clusters={g.vocab2index[k]:v for k,v in clusters.items()})
#print(m.Q())
#print(g.adjacency_matrix())
#print(m.ERC)
