from bs4 import BeautifulSoup
from graph import Vertex, Graph, Graph_tool, Networkx, MultilayerGraph
import numpy as np
import sys
import networkx as nx
import re, json

from graph import Vertex, Graph, Graph_tool, Networkx
import numpy as np
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    lines = f.readlines()

v = []

for l in lines:
    tmp = re.split('\s{2,}', l.replace('\n',''))
    try:
        v.append(Vertex(tmp[0],[tmp[2]],[tmp[1]]))
    except:
        print('WARNING: expected three words:')
        print(tmp)

#g = Graph_tool(vertices=preds)
f = Networkx(vertices=preds)

#g.causal()
f.causal()
#g.clustering()
#g.filter_by('co-occurrence', threshold=0.01)
f.filter_by(method='redundancy', k=3)
#f.to_unoriented()
#f.filter_by(method='degree')
#g.filter_by(method='redundancy', k=6)

#pos = nx.drawing.layout.spring_layout(f.g)
#pos = {0: np.array([ 0.005177  , -0.01378308]), 1: np.array([0.08327484, 0.01442597]), 2: np.array([-0.24544246, -0.04149584]), 3: np.array([0.0183189 , 0.07946743]), 4: np.array([-0.04167114, -0.04631544]), 5: np.array([ 0.22179903, -0.08233053]), 6: np.array([ 0.11229854, -0.08666977]), 7: np.array([0.13185139, 0.03685468]), 8: np.array([-0.28334697, -0.05864635]), 9: np.array([-0.16833371, -0.00570331]), 10: np.array([0.00645827, 0.1140777 ]), 11: np.array([-0.08926613,  0.06505432]), 12: np.array([0.08381458, 0.05888981]), 13: np.array([0.00639632, 0.08147242]), 14: np.array([0.0280978 , 0.11547187]), 15: np.array([ 0.04868724, -0.0020816 ]), 16: np.array([-0.15370671,  0.06533819]), 17: np.array([ 0.06783998, -0.04316682]), 18: np.array([0.05254816, 0.09761442]), 19: np.array([ 0.06967138, -0.02456179]), 20: np.array([ 0.01459776, -0.04759082]), 21: np.array([0.04642875, 0.11307282]), 22: np.array([-0.01271988,  0.13150242]), 23: np.array([-0.27265078,  0.76176284]), 24: np.array([ 0.05601203, -0.0207886 ]), 25: np.array([-0.02969678,  0.06804893]), 26: np.array([-0.00803031,  0.09999429]), 27: np.array([ 0.07896369, -0.05830509]), 28: np.array([ 0.00808347, -0.19997771]), 29: np.array([0.00346288, 0.1402495 ]), 30: np.array([0.02189014, 0.14096652]), 31: np.array([0.04302224, 0.13574455]), 32: np.array([0.03031942, 0.03939797]), 33: np.array([0.19822748, 0.03460848]), 34: np.array([-0.05834297,  0.15271569]), 35: np.array([ 0.14559157, -0.09213088]), 36: np.array([0.55851283, 0.36859964]), 37: np.array([ 0.34557479, -0.71020279]), 38: np.array([ 0.25629947, -0.12165649]), 39: np.array([ 0.26415418, -0.09386016]), 40: np.array([ 0.06637083, -0.14807578]), 41: np.array([-0.14663127,  0.04903381]), 42: np.array([ 0.11414366, -0.12657848]), 43: np.array([0.05448691, 0.01057209]), 44: np.array([0.4469181, 0.5045464]), 45: np.array([0.16381289, 0.08525051]), 46: np.array([0.18093175, 0.0524541 ]), 47: np.array([ 0.10539647, -0.03177878]), 48: np.array([ 0.08645199, -0.07133407]), 49: np.array([0.15424783, 0.13443951]), 50: np.array([ 0.08666284, -0.26367477]), 51: np.array([-0.2324429 , -0.81880477]), 52: np.array([ 0.09572893, -0.12351976]), 53: np.array([0.49948753, 0.28537847]), 54: np.array([-0.08811068,  0.20266171]), 55: np.array([-0.10586945,  0.18894225]), 56: np.array([ 0.02667821, -0.20235512]), 57: np.array([0.17747862, 0.12979788]), 58: np.array([-0.10415483,  0.02011054]), 59: np.array([-0.05207181,  0.04457359]), 60: np.array([ 0.09844576, -0.46280494]), 61: np.array([ 0.05123891, -0.08287735]), 62: np.array([-0.07268903,  0.03793238]), 63: np.array([-0.02770273, -0.0850404 ]), 64: np.array([-0.28089655,  0.12209451]), 65: np.array([0.13313945, 0.68964945]), 66: np.array([0.31582352, 0.59838874]), 67: np.array([-0.87062533, -0.48114309]), 68: np.array([-0.92291609, -0.51532254]), 69: np.array([-0.47298694,  0.00409422]), 70: np.array([-1.        , -0.00201489]), 71: np.array([ 0.08370344, -0.77264709]), 72: np.array([0.14746448, 0.21133665]), 73: np.array([0.05810269, 0.05513623]), 74: np.array([0.15199074, 0.14088412]), 75: np.array([-0.05568139,  0.13566963]), 76: np.array([-0.02583272,  0.11246171]), 77: np.array([0.17061703, 0.10942884]), 78: np.array([-0.04502356,  0.10768657]), 79: np.array([-0.06887011,  0.09714189]), 80: np.array([-0.1187595 ,  0.11042179]), 81: np.array([0.08655432, 0.23008602]), 82: np.array([-0.05171484,  0.76037387]), 83: np.array([ 0.1032379, -0.0625086]), 84: np.array([ 0.20585854, -0.62378157]), 85: np.array([ 0.10125042, -0.14415439]), 86: np.array([ 0.23990576, -0.11315221]), 87: np.array([ 0.60781499, -0.27909008]), 88: np.array([ 0.21485401, -0.51992883]), 89: np.array([0.32567127, 0.38797789]), 90: np.array([0.17683614, 0.0274616 ]), 91: np.array([ 0.11119225, -0.25047807]), 92: np.array([-0.51939739, -0.2011651 ]), 93: np.array([-0.95655629, -0.32668347]), 94: np.array([-0.01264491, -0.09669111]), 95: np.array([ 0.13041841, -0.19474561]), 96: np.array([-0.01224661,  0.06368744]), 97: np.array([-0.12330612,  0.24643356]), 98: np.array([-0.01496335, -0.24827693]), 99: np.array([-0.09165136,  0.13008105]), 100: np.array([ 0.12320406, -0.15848016]), 101: np.array([-0.25897893,  0.05036838]), 102: np.array([-0.9140028 ,  0.11996084]), 103: np.array([-0.00453831,  0.04538325]), 104: np.array([ 0.14349975, -0.1480782 ]), 105: np.array([-0.03252094,  0.0872213 ])}
#f.draw()
#g.draw()
#g.json('g.json')
#f.json('graph.json')


#embedding = g.deep_walk(walks_per_node=60)

wedges = True

embedding = f.deep_walk(walk_length=80, window=10, walks_per_node=20, embedding_dim=2, with_edges=wedges)
#embedding = g.deep_walk(walk_length=80, window=10, walks_per_node=10, embedding_dim=300)
clusters = f.k_means(elbow_range=(2,30))
#clusters = f.k_means(6)
#clusters = g.k_means(elbow_range=(2,30))

ann = []
for k in embedding.keys():
    if re.match('^[A-Z][A-Z0-9]+', k) != None:
        ann.append(k)
f.draw_embedding(ann)
if wedges:
    plt.savefig('embedding_with_edges.png')
else:
    plt.savefig('embedding_no_edges.png')
    
#for i in range(max(list(clusters.values())) + 1):
 #   f.draw_cluster([i])
if wedges: 
    with open('clusters-with-edges.txt', 'w') as out:
        out.write(json.dumps(f.get_clusters(), indent=4))
else:
    with open('clusters-no-edges.txt', 'w') as out:
        out.write(json.dumps(f.get_clusters(), indent=4))
'''
evecs = np.array([v for v in embedding.values()])
pca = PCA(n_components=50)
pca.fit(evecs)
evecs = pca.transform(evecs)
evecs = TSNE(n_components=2).fit_transform(evecs)
i = 0
twod_emb = {}
for k in embedding.keys():
    twod_emb[k] = evecs[i]
    i += 1
'''
#fig, axs = plt.subplots(1,2, figsize=(20,10))
#c = list(clusters.values())



    
#axs[0].scatter(x=evecs[:,0], y=evecs[:,1], c=c, cmap=plt.cm.get_cmap('tab10'), label=list(f.get_clusters().keys()))
'''
i = 0
for k in embedding.keys():
    if re.match('^[A-Z][A-Z0-9]+', k) != None:
        axs[0].annotate(k, (evecs[i][0], evecs[i][1]), fontsize=8)
    i += 1
'''
'''
axs[1].scatter(x=evecs[:,0], y=evecs[:,1], c=c, cmap=plt.cm.get_cmap('tab10'))
i = 0
for k in embedding.keys():
    if re.match('^[A-Z][A-Z0-9]+', k) != None:
        axs[1].annotate(k, (evecs[i][0], evecs[i][1]), fontsize=8)
    i += 1
'''
#nx.draw_networkx(f.g, ax=axs[1], labels={i:f.get_vertex(i) for i in f.get_vertices()}, node_color=list(clusters.values()), cmap=plt.cm.get_cmap('hsv'), with_labels=True, edge_color='grey', font_size=8)
#plt.show()


#f.filter_by('co-occurrence', threshold=0.05)
#clusters = g.k_means(elbow_range=(2,30))
#for k,v in clusters.items():
#    print(k,':',v)

#f.json('graph.json')
#g.json('graph.json')


#for n in g.get_neighbors(g.get_vertex('Virus Diseases')):
#    print(g.get_vertex(int(n)))
#for n in f.get_neighbors(f.get_vertex('Virus Diseases')):
#    print(f.get_vertex(n))

#embedding = g.deep_walk(walks_per_node=60)
embedding = f.deep_walk(walks_per_node=20)
embedding = np.array([v for v in embedding.values()])
pca = PCA(n_components=50)
pca.fit(embedding)
embedding = pca.transform(embedding)
embedding = TSNE(n_components=2).fit_transform(embedding)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x=embedding[:,0], y=embedding[:,1])
plt.show()
#f.filter_by('co-occurrence', threshold=0.05)
#clusters = g.k_means(elbow_range=(2,30))
clusters = f.k_means(elbow_range=(2,30))
#for k,v in clusters.items():
#    print(k,':',v)

f.json('graph.json')
#g.json('graph.json')


def cos(v1,v2):
    n1 = np.sqrt(np.dot(v1,v1))
    n2 = np.sqrt(np.dot(v2,v2))
    return np.dot(v1,v2)/(n1*n2)

#v1 = embedding[g.get_vertex('Virus Diseases')]
#v1 = embedding[f.get_vertex('Virus Diseases')]
#v2 = embedding[g.get_vertex('Pharmaceutical Preparations')]
#v2 = embedding[g.get_vertex('Antiviral Therapy')]

#for key, value in embedding.items():
    #print(g.get_vertex(key),'-->',cos(v1,value))
    #print(f.get_vertex(key),'-->',cos(v1,value))

