from bs4 import BeautifulSoup
import sys

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
                    o = obj['name'].replace(':','')
                    #o = [obj['text'].replace(':','')]
                except:
                    o = obj['entrezName'].replace(':','')
                    #o = [obj['text'].replace(':','')]
                preds.append([s, o, pred])
        i += 1
        print(i, '/', len(soup), '\r', end='')

with open(sys.argv[1].replace('predicates.xml' ,'edges.txt'), 'w') as f:
    for p in preds:
        f.write('{0:100}{1:100}{2}\n'.format(str(p[0]),str(p[1]),str(p[2])))
