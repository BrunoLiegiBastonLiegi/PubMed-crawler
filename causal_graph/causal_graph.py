from bs4 import BeautifulSoup
from graph import Vertex, Graph, Graph_tool, Networkx
#from gensim.models import Word2Vec
import numpy as np


causePredicates = ['CAUSES','PREVENTS','DISRUPTS','INHIBITS',
                    'PREDISPOSES','PRODUCES']
twoDirectionalPredicates = ['COEXISTS_WITH','ASSOCIATED_WITH']

with open('predicates.xml') as f:
    
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
                except:
                    s = subj['entrezName'].replace(':','')
                try:
                    o = [obj['name'].replace(':','')]
                except:
                    o = [obj['entrezName'].replace(':','')]
                preds.append(Vertex(s, [pred], o))
        i += 1
        #print(i, '/', len(soup), '\r', end='')

#print('\n', len(preds), ' predications found')

    
g = Graph_tool(vertices=preds)
f = Networkx(vertices=preds)

print('~~~~~~~~~~~~~~~ GRAPHTOOL')
g.causal()
print('~~~~~~~~~~~~~~~ NETWORKX')
f.causal()
#g.clustering()
#g.filter_by('co-occurrence', threshold=0.01)
#g.filter_by(method='redundancy', k=6)
#g.merge_vertices(g.v_mapping['Disease'], g.v_mapping['Virus Diseases'])
g.draw()
f.draw()
#g.json()

print(f.get_vertices())
print(g.get_vertices())

#embedding = g.deep_walk()

def cos(v1,v2):
    n1 = np.sqrt(np.dot(v1,v1))
    n2 = np.sqrt(np.dot(v2,v2))
    return np.dot(v1,v2)/(n1*n2)

#v1 = embedding[g.label2vertex['Procedures']]
#v2 = embedding[g.v_mapping['Pharmaceutical Preparations']]
#v2 = embedding[g.v_mapping['Antiviral Therapy']]

#print('Cosine similarity between node \'Procedures\' and:')
#for key, value in embedding.items():
    #print(g.verts_text[key],'-->',cos(v1,value))

