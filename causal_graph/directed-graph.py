from graph import Graph, Vertex, Networkx, Graph_tool
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import sys


inpt = sys.argv[1]


if inpt == '0':

    g = Networkx(vertices=[Vertex(str(i),[],[]) for i in range(6)])
    
    v = []
    v.append(Vertex('0',['',''],['1','2']))
    v.append(Vertex('1',['',''],['0','3']))
    v.append(Vertex('2',['',''],['3','4']))
    v.append(Vertex('3',['',''],['2','5']))
    v.append(Vertex('4',['',''],['2','5']))
    v.append(Vertex('5',['',''],['4','3']))


    [g.add_vertex(i) for i in v]

    pos = {
        0:np.array([0,1]),
        1:np.array([0,0]),
        2:np.array([1,1]),
        3:np.array([1,0]),
        4:np.array([2,1]),
        5:np.array([2,0])
    }
    
    nx.draw_networkx(g.g, pos=pos, with_labels=True)
    plt.show()
    
    wedges = False
    embedding = g.deep_walk(walk_length=10, window=4, walks_per_node=80, embedding_dim=128, with_edges=wedges)
    clusters = g.k_means(n_clusters=2)


    embedding = np.array([v for v in embedding.values()])
    pca = PCA(n_components=2)
    pca.fit(embedding)
    embedding = pca.transform(embedding)
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].scatter(x=embedding[:,0], y=embedding[:,1])
    for i in range(len(g.get_vertices())):
        axs[0].annotate(i, (embedding[i][0], embedding[i][1]))

    nx.draw_networkx(g.g, ax=axs[1], pos=pos, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
    plt.show()


elif inpt == '1':

    f = Networkx(vertices=[Vertex(str(i),[],[]) for i in range(16)])

    v = []
    for i in [0,4,8,12]:
        for j in range(3):
            v.append(Vertex(str(i+j),[''],[str(i+j+1)]))
        v.append(Vertex(str(i+3),[''],[str(i)]))
    
    v.append(Vertex('0',[''],['7']))
    v.append(Vertex('4',[''],['11']))
    v.append(Vertex('8',[''],['15']))
    v.append(Vertex('12',[''],['3']))
    v.append(Vertex('1',[''],['6']))
    v.append(Vertex('5',[''],['10']))
    v.append(Vertex('9',[''],['14']))
    v.append(Vertex('13',[''],['2']))

    [f.add_vertex(i) for i in v]


    pos = {
        0: np.array([0,1]),
        1: np.array([1,2]),
        2: np.array([2,1]),
        3: np.array([1,0]),
        4: np.array([1,5]),
        5: np.array([2,4]),
        6: np.array([1,3]),
        7: np.array([0,4]),
        8: np.array([5,4]),
        9: np.array([4,3]),
        10: np.array([3,4]),
        11: np.array([4,5]),
        12: np.array([4,0]),
        13: np.array([3,1]),
        14: np.array([4,2]),
        15: np.array([5,1]),
    }

    nx.draw_networkx(f.g, pos=pos, with_labels=True)
    plt.show()
    
    wedges = False
    embedding = f.deep_walk(walk_length=20, window=5, walks_per_node=80, embedding_dim=128, with_edges=wedges)
    clusters = f.k_means(n_clusters=4)


    embedding = np.array([v for v in embedding.values()])
    pca = PCA(n_components=2)
    pca.fit(embedding)
    embedding = pca.transform(embedding)
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].scatter(x=embedding[:,0], y=embedding[:,1])
    for i in range(len(f.get_vertices())):
        axs[0].annotate(i, (embedding[i][0], embedding[i][1]))

    nx.draw_networkx(f.g, ax=axs[1], pos=pos, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
    plt.show()



elif inpt == '2':

    h = Networkx(vertices=[Vertex(str(i),[],[]) for i in range(8)])

    v = []
    for i in range(3):
        v.append(Vertex(str(i), [''], ['3']))
        v.append(Vertex(str(i), [''], ['4']))

    v.append(Vertex('3', ['','',''], ['5','6','7']))
    v.append(Vertex('4', ['','',''], ['5','6','7']))

    [h.add_vertex(i) for i in v]

    pos = {
        0:np.array([0,2]),
        1:np.array([0,1]),
        2:np.array([0,0]),
        3:np.array([1,1.5]),
        4:np.array([1,0.5]),
        5:np.array([2,2]),
        6:np.array([2,1]),
        7:np.array([2,0]),
    }
    
    nx.draw_networkx(h.g, pos=pos, with_labels=True)
    plt.show()

    wedges = False
    embedding = h.deep_walk(walk_length=3, window=3, walks_per_node=80, embedding_dim=128, with_edges=wedges)
    clusters = h.k_means(n_clusters=3)


    embedding = np.array([v for v in embedding.values()])
    pca = PCA(n_components=2)
    pca.fit(embedding)
    embedding = pca.transform(embedding)
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    axs[0].scatter(x=embedding[:,0], y=embedding[:,1])
    for i in range(len(h.get_vertices())):
        axs[0].annotate(i, (embedding[i][0], embedding[i][1]))

    nx.draw_networkx(h.g, ax=axs[1], pos=pos, node_color=[c for c in clusters.values()], cmap=plt.cm.get_cmap('cool'), with_labels=True)
    plt.show()
