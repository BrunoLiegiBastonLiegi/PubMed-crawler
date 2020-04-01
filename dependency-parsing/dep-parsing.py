import spacy, re, random
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from spacy.symbols import nsubj

#import tensorflow_hub as hub
#import tensorflow as tf


query = 'stroke-risk-factors'

with open('sentences/'+query+'_sentences.txt', 'r') as f:
    sents = f.readlines()


for i in range(len(sents)):
    doc = nlp(sents[random.randrange(len(sents))])
    print(doc)
    #for tok in doc:
        #print(tok,'  ',tok.dep_,'\n')
    subjs = []
    roots = []
    objs = []
    for tok in doc:
        
        if tok.dep_ == 'nsubj' or tok.dep_ == 'nsubjpass':
            if tok.head.dep_ == 'ROOT':
                subjs.append((tok,tok.head))
            else:
                print(tok, '  ', tok.head, '--> non root head for nsubj\n\n\n\n')
                
        if tok.dep_ == 'pobj':
            head = tok.head
            while True:
                if head.dep_ == 'nsubj':
                    break
                if head.dep_ == 'dobj':
                    break
                if head.dep_ == 'pobj':
                    break
                if head.dep_ == 'nsubjpass':
                    break
                if head.dep_ == 'ROOT':
                    break
                head = head.head
            objs.append((tok,head))
            
        if tok.dep_ == 'dobj':
            if tok.head.dep_ == 'ROOT':
                objs.append((tok,tok.head))
            else:
                print(tok, '  ', tok.head, '--> non root head for dobj\n\n\n\n')
                
        if tok.dep_ == 'ROOT':
            roots.append(tok)

    print(subjs, '\n', roots, '\n', objs, '\n')
    displacy.serve(doc, style='dep')












        
