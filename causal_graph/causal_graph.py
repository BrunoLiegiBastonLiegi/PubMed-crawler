from bs4 import BeautifulSoup
from graph import Vertex, Graph
from gensim.models import Word2Vec


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
        print(i, '/', len(soup), '\r', end='')

print('\n', len(preds), ' predications found')

    
g = Graph(vertices=preds)
g.causal()
#g.redundancy_filter(k=2)
g.co_occurrence(0.1)
#g.merge_vertices(g.v_mapping['Disease'], g.v_mapping['Virus Diseases'])
#model = Word2Vec.load('../word-embedding/models/word2vec_window=6_size=300_min_count=7_iter=20.model')
#g.word_embedding_filter(model, 'virus')
g.draw()


