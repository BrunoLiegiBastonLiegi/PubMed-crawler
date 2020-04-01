import requests as req
import re, os, spacy

from spacy.lang.en import English
#nlp = English()
nlp = spacy.load("en_core_web_md")

import tensorflow_hub as hub
import tensorflow as tf





index = 'stroke-risk-factors'
f = open('../json/' + index + '/' + index + '.json', 'r')

lines = f.readlines()
f.close()
ids = []
for i in range(int(len(lines)/2)):
    ids.append(re.findall("\"[0-9]*\"", lines[2*i])[0].replace('\"',''))


header = {'Content-Type': 'application/json'}
payload = '{"positions":false,"offsets":false,"term_statistics":false,"filter":{"max_num_terms":6,"min_term_freq":1,"min_doc_freq":1}}'
params = {'fields':'Abstract'}

print('--> Extracting keywords')


try:
    f = open(index + '/' + index + '-keywords.txt', 'w')
except:
    os.makedirs(index)
    f = open(index + '/' + index + '-keywords.txt', 'w')



#elmo = hub.Module("https://tfhub.dev/google/elmo/3")

    
count = 0
wordvecs = []
non_vecs = []
for _id in ids:
    
    url = 'http://localhost:9200/' + index + '/_termvectors/' + _id + '?'
    r = req.get(url=url, headers=header, data=payload, params=params)
    
    start = r.text.find('"terms"')
    #scores = re.findall('[.0-9]+}', r.text[start+9:])
    keywords = re.findall('[\w\s@®↔\.\-\']+\":{' , r.text[start+9:])
    
    tmp = []
    if len(keywords) > 13:                               # checking that abstract is avaliable
        for i in range(len(keywords)):
            key = keywords[i].replace('\":{','')
            if nlp.vocab[key].is_stop == False and nlp.vocab[key].like_num == False:
                #tmp.append([key.lemma_, float(scores[i].replace('}',''))])
                tmp.append(key)
        #[print(token.lemma_) for token in nlp(' '.join(tmp))]
        '''
        embeddings = elmo([token.lemma_ for token in nlp(' '.join(tmp))], signature="default", as_dict=True)["elmo"]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(tf.reduce_mean(embeddings,1))
                            
        #tmp.sort(key=lambda x:x[1], reverse=True)
        '''
        lemms = ' '.join([token.lemma_ for token in nlp(' '.join(tmp))])

        for token in nlp(lemms):
            if token.has_vector == True:
                wordvecs.append(token)
            else:
                non_vecs.append(token)
        
        #print([token.has_vector for token in nlp(' '.join(tmp))])
        f.write(str(wordvecs))
        f.write(str(non_vecs))
        f.write('\n')
            
    count += 1
    print('(', count, '/', len(ids), ')\r', end='')

wordvecs = ' '.join(wordvecs)
non_vecs = ' '.join(non_vecs)
    
f.close()
print('')


