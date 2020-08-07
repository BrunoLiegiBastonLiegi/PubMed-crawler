from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

'''
v = []
v.append(Vertex('M1',['family','family','family','friend'],['F1','S1','D1','M2']))
v.append(Vertex('F1',['family','family','family','friend'],['M1','S1','D1','F2']))
v.append(Vertex('M2',['family','family','family','friend'],['F2','S2','D2','M1']))
v.append(Vertex('F2',['family','family','family','friend'],['M2','S2','D2','F1']))
v.append(Vertex('S2',['family','family','family','friend'],['M2','F2','D2','S1']))
v.append(Vertex('S1',['family','family','family','friend'],['M1','F1','D1','S2']))
v.append(Vertex('D1',['family','family','family','friend'],['M1','F1','S1','D2']))
v.append(Vertex('D2',['family','family','family','friend'],['M2','F2','S2','D1']))

v.append(Vertex('M1',['friend'],['F2']))
v.append(Vertex('M2',['friend'],['F1']))
v.append(Vertex('F1',['friend'],['M2']))
v.append(Vertex('F2',['friend'],['M1']))
v.append(Vertex('S1',['friend'],['D2']))
v.append(Vertex('D1',['friend'],['S2']))
v.append(Vertex('S2',['friend'],['D1']))
v.append(Vertex('D2',['friend'],['S1']))

g = Networkx(vertices=v)
g.draw()
'''

'''
v = []
v.append(Vertex('M1',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('M2',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('F1',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('F2',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('S1',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('S2',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('D1',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('D2',['' for i in range(4)],['family','family','family','friend']))
v.append(Vertex('family',['' for i in range(24)],['F1','S1','D1','M1','S1','D1','M1','F1','D1','M1','F1','S1','F2','S2','D2','M2','S2','D2','M2','F2','D2','M2','F2','S2']))
v.append(Vertex('friend',['' for i in range(8)],['M2','F2','S2','D2','M1','F1','S1','D1']))

f = Networkx(vertices=v)
f.draw(edge_label='weight')
'''


v = []
v.append(Vertex('0',['friend' for i in range(3)], ['1','2','3']))
v.append(Vertex('1',['friend' for i in range(3)], ['0','2','3']))
v.append(Vertex('2',['friend' for i in range(4)], ['1','3','0','7']))
v.append(Vertex('3',['friend' for i in range(3)]+['family' for i in range(3)], ['1','2','0', '4','5','6']))
v.append(Vertex('4',['family' for i in range(3)], ['3','5','6']))
v.append(Vertex('5',['family' for i in range(3)], ['4','6','3']))
v.append(Vertex('6',['family' for i in range(3)], ['4','5','3']))

v.append(Vertex('7',['friend' for i in range(2)], ['2','8']))

v.append(Vertex('10',['friend' for i in range(3)], ['11','8','9']))
v.append(Vertex('9',['friend' for i in range(3)], ['10','8','11']))
v.append(Vertex('8',['friend' for i in range(4)], ['10','9','11','7']))
v.append(Vertex('11',['friend' for i in range(3)]+['family' for i in range(3)], ['10','9','8', '14','12','13']))
v.append(Vertex('12',['family' for i in range(3)], ['11','13','14']))
v.append(Vertex('13',['family' for i in range(3)], ['14','11','12']))
v.append(Vertex('14',['family' for i in range(3)], ['12','11','13']))

g = Networkx(vertices=v)

pos = {
    g.get_vertex('0') : np.array([0,0]),
    g.get_vertex('1') : np.array([1,1]),
    g.get_vertex('2') : np.array([2,0]),
    g.get_vertex('3') : np.array([1,-1]),
    g.get_vertex('4') : np.array([0,-2]),
    g.get_vertex('5') : np.array([1,-3]),
    g.get_vertex('6') : np.array([2,-2]),
    g.get_vertex('7') : np.array([3,0]),
    g.get_vertex('8') : np.array([4,0]),
    g.get_vertex('9') : np.array([5,1]),
    g.get_vertex('10') : np.array([6,0]),
    g.get_vertex('11') : np.array([5,-1]),
    g.get_vertex('12') : np.array([6,-2]),
    g.get_vertex('13') : np.array([5,-3]),
    g.get_vertex('14') : np.array([4,-2]),
}

g.draw(pos=pos)

wedges = True

embedding = g.deep_walk(walk_length=80, window=10, walks_per_node=10, embedding_dim=128, with_edges=wedges)
clusters = g.k_means(n_clusters=2)

evecs = np.array([v for v in embedding.values()])
pca = PCA(n_components=2)
pca.fit(evecs)
evecs = pca.transform(evecs)
fig, axs = plt.subplots(1,2, figsize=(20,10))
axs[0].scatter(x=evecs[:,0], y=evecs[:,1])
for i in g.get_vertices():
    axs[0].annotate(g.get_vertex(i), (evecs[i][0], evecs[i][1]))
if wedges:
    axs[0].annotate('friend', (evecs[-1][0], evecs[-1][1]))
    axs[0].annotate('family', (evecs[-2][0], evecs[-2][1]))

nx.draw_networkx(g.g, ax=axs[1], pos=pos, labels={i:g.get_vertex(i) for i in g.get_vertices()}, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
plt.show()
