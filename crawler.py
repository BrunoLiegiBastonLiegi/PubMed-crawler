import requests as req
from bs4 import BeautifulSoup
from article import Article
from multiprocessing import Process, Value, Lock, cpu_count

import time, os


query = ['covid-19']
logic_clause = '  '
retstart = 0
retmax = 100000
idlist = []

print('--> Searching Pubmed')

while True:

    #params passed to esearch
    payload = {'db':'pubmed', 'term':logic_clause.join(query), 'retmax':retmax, 'retstart':retstart}

    #esearch
    r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=payload)

    #parsing Ids
    soup = BeautifulSoup(r.text, "xml")
    
    Id = soup.Id
    if Id == None:
        break
    
    while Id != None:
        idlist.append(int(Id.text))
        Id = Id.find_next('Id')

    print(len(idlist), 'articles found\r', end='')
        
    retmax = int(soup.RetMax.text)
    retstart += retmax

    
print(len(idlist), 'articles found\r', end='')
print('')



def worker(index, lock, f):

    lock.acquire()
    ind = index.value
    if ind+batchsize < len(idlist):
        print('(', ind, '/', len(idlist), ')\r', end='')
        index.value += batchsize
        lock.release()
        payload = {'db':'pubmed', 'id':idlist[ind:index.value], 'retmode':'xml'}
    else:
        if ind < len(idlist):
            print('(', ind, '/', len(idlist), ')\r', end='') 
            index.value = len(idlist)
            lock.release()
            payload = {'db':'pubmed', 'id':idlist[ind:], 'retmode':'xml'}
        else:
            lock.release()
            return

    soup = None
    while soup == None:
        #efetch
        r = req.post('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', data=payload)
        #parsing
        soup = BeautifulSoup(r.text, "xml").PubmedArticleSet
        if soup == None:
            time.sleep(1)
        
    children = list(soup.children)[1::2]

    for j in children:
        art = Article(j)
        line = art.line()
        lock.acquire()
        f[0].write(line[0])
        f[0].flush()
        f[1].write(line[1])
        lock.release()
        

        

try:
    f1 = open('DATA/xml/' + '-'.join(query) + '/' + '-'.join(query) + '.xml','w')
    f1.write('<?xml version="1.0" encoding="utf-8"?><root>')
    f1.flush()
except:
    os.makedirs('DATA/xml/' + '-'.join(query))
    f1 = open('DATA/xml/' + '-'.join(query) + '/' + '-'.join(query) + '.xml','w')
    f1.write('<?xml version="1.0" encoding="utf-8"?><root>')


try:
    f2 = open('elasticsearch/json/' + '-'.join(query) + '/' + '-'.join(query) + '.json','w')
except:
    os.makedirs('elasticsearch/json/' + '-'.join(query))
    f2 = open('elasticsearch/json/' + '-'.join(query) + '/' + '-'.join(query) + '.json','w')


    

batchsize = 2000
nproc = cpu_count()

    
if __name__ == '__main__':

    lock = Lock()
    index = Value('i', 0)
    
    print('--> Retrieving articles')
    while index.value < len(idlist):

        jobs = []
        
        for i in range(nproc):
            p = Process(target=worker, args=(index, lock, (f1,f2)))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        
    print('(', len(idlist), '/', len(idlist), ')\r', end='')
    print('')



f1.write('</root>')
f1.close()
f2.close()


