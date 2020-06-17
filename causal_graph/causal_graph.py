from bs4 import BeautifulSoup
from graph import Vertex, Graph, Graph_tool, Networkx
#from gensim.models import Word2Vec
import numpy as np
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

causePredicates = ['CAUSES','PREVENTS','DISRUPTS','INHIBITS',
                    'PREDISPOSES','PRODUCES']
twoDirectionalPredicates = ['COEXISTS_WITH','ASSOCIATED_WITH']

with open(sys.argv[1]) as f:
    
    soup = BeautifulSoup(f.read(), "xml").find_all('Utterance')
    
    preds = []
    i = 0

    for sent in soup:
        predication = sent.Predication
        if predication != None:
            pred = predication.Predicate['type']
            if True:#pred in causePredicates or pred in twoDirectionalPredicates:
                subj = sent.find(id=predication.Subject['entityID'])
                obj = sent.find(id=predication.Object['entityID'])
                dict = {}
                try:
                    s = subj['name'].replace(':','')
                    #s = subj['text'].replace(':','')
                except:
                    s = subj['entrezName'].replace(':','')
                    #s = subj['text'].replace(':','')
                try:
                    o = [obj['name'].replace(':','')]
                    #o = [obj['text'].replace(':','')]
                except:
                    o = [obj['entrezName'].replace(':','')]
                    #o = [obj['text'].replace(':','')]
                preds.append(Vertex(s, [pred], o))
        i += 1
        print(i, '/', len(soup), '\r', end='')

#print('\n', len(preds), ' predications found')


#g = Graph_tool(vertices=preds)
f = Networkx(vertices=preds)

#g.causal()
f.causal()
#g.clustering()
#g.filter_by('co-occurrence', threshold=0.01)
#g.filter_by(method='redundancy', k=6)
#g.merge_vertices(g.v_mapping['Disease'], g.v_mapping['Virus Diseases'])
#g.draw()
#f.degree(d=3)
#g.degree(d=3)
#f.filter_by('degree', d=3)
#f.draw()
#g.draw()
#g.json('g.json')
#f.json('graph.json')


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

