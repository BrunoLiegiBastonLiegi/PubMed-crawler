from graph import MultilayerGraph
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


m = MultilayerGraph(['family','friend'])

m.add_edge('family', 'M1' , 'F1', dir='bi')
m.add_edge('family', 'M1' , 'S1', dir='bi')
m.add_edge('family', 'M1' , 'D1', dir='bi')
m.add_edge('family', 'F1' , 'S1', dir='bi')
m.add_edge('family', 'F1' , 'D1', dir='bi')
m.add_edge('family', 'S1' , 'D1', dir='bi')

m.add_edge('family', 'M2' , 'F2', dir='bi')
m.add_edge('family', 'M2' , 'S2', dir='bi')
m.add_edge('family', 'M2' , 'D2', dir='bi')
m.add_edge('family', 'F2' , 'S2', dir='bi')
m.add_edge('family', 'F2' , 'D2', dir='bi')
m.add_edge('family', 'S2' , 'D2', dir='bi')

m.add_edge('friend', 'M1' , 'M2', dir='bi')
m.add_edge('friend', 'F1' , 'F2', dir='bi')
m.add_edge('friend', 'S1' , 'S2', dir='bi')
m.add_edge('friend', 'D1' , 'D2', dir='bi')

'''
m.add_edge('friend', 'M1' , 'F2', dir='bi')
m.add_edge('friend', 'F1' , 'M2', dir='bi')
m.add_edge('friend', 'S1' , 'D2', dir='bi')
m.add_edge('friend', 'D1' , 'S2', dir='bi')
'''


#print('VERTICES:\n', m.vertices())
#print('EDGES:\n', m.edges())
#m.draw()

m.l = 0.3
embedding = m.deep_walk(walk_length=80, window=5, walks_per_node=80, embedding_dim=128)
#print(embedding)  # embedding vectors are in different order!!!! this is the problem
clusters = m.k_means(n_clusters=2)

evecs = np.array([v for v in embedding.values()])
pca = PCA(n_components=2)
pca.fit(evecs)
evecs = pca.transform(evecs)
fig, axs = plt.subplots(1,2, figsize=(20,10))
axs[0].scatter(x=evecs[:,0], y=evecs[:,1])
verts = m.vertices()
for i in range(len(verts)):
    axs[0].annotate(verts[i], (evecs[i][0], evecs[i][1]))

plt.show()
m.draw()
