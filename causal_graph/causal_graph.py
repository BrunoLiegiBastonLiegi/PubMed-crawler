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

g.causal()
f.causal()
#g.clustering()
#g.filter_by('co-occurrence', threshold=0.01)
#g.filter_by(method='redundancy', k=6)
#g.merge_vertices(g.v_mapping['Disease'], g.v_mapping['Virus Diseases'])
#g.draw()
#f.draw()
g.json('g.json')
f.json('f.json')

#dg = g.get_edges()
#df = f.get_edges()

dg = g.get_vertices()
df = f.get_vertices()

tmp = []

'''
for i in df:
    tmp.append(str(f.get_vertex(i[0]))+' - '+str(f.get_vertex(i[1]))+' - '+i[2]+' - '+str(i[3]))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
for i in dg:
    print(str(g.get_vertex(i[0]))+' - '+str(g.get_vertex(i[1]))+' - '+i[2]+' - '+str(i[3]) in tmp)


for i in df:
    tmp.append(str(f.get_vertex(i)))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
for i in dg:
    if str(g.get_vertex(i)) not in tmp:
        print(g.get_vertex(i))
'''


c=0
for i in g.get_edges('Virus Diseases', dir='all'):
    print(g.get_vertex(i[0]),' - ', g.get_vertex(i[1]), ' - ', i[2], ' - ', i[3])
    c+=1
print(c)
print('\n')
c=0
for i in f.get_edges('Virus Diseases', dir='all'):
    print(f.get_vertex(i[0]),' - ', f.get_vertex(i[1]), ' - ', i[2], ' - ', i[3])
    c+=1
print(c)
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

