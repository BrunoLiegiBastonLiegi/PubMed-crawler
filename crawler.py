import requests as req
from bs4 import BeautifulSoup
from article import Article
from multiprocessing import Process, Value, Lock, cpu_count
import time, os
from datetime import datetime

query = ['covid-19']

mode = 'psql' # possible modes: 'xml','json','psql'

logic_clause = '  '
retstart = 0
retmax = 100000
idlist = []


if mode == 'psql':
    import psycopg2
    from psycopg2.extras import execute_values
    conn = psycopg2.connect(host='localhost',database='papercrawler',user='crawler',password='crawler2020')
    cur = conn.cursor()
    
    # Create table if not exist 
    cur.execute('create table if not exists papers ( insert_ts BIGINT NOT NULL, source VARCHAR(30) NOT NULL, id BIGINT PRIMARY KEY, title TEXT, keywords TEXT, abstract TEXT, journal TEXT  );')
    
    # Create indices
    cur.execute("CREATE INDEX if not exists text_ind ON papers USING GIN (to_tsvector('english', abstract)) ;")
    
    conn.commit()
    conn.close()
    
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
    conn = None
    cur = None
    if mode == 'psql':
        conn = psycopg2.connect(host='localhost',database='papercrawler',user='crawler',password='crawler2020')
        cur = conn.cursor()
        
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

    datalist = []
    
    for j in children:
        art = Article(j)
        line = art.line()
        lock.acquire()
        
        if mode == 'xml':
            f.write(line[0])
            f.flush()
        
        if mode == 'json':
            f.write(line[1])
        
        if mode == 'psql':
           
            cur.execute("SELECT COUNT(*) FROM papers where id="+str(art.get_id())+";")
            tmp = cur.fetchall()
         
            if tmp[0][0] < 1:
                data = (round(time.time()),'pubmed',art.get_id(),art.get_title(),art.get_keywords(),art.get_abstract(),art.get_journal())
                datalist.append(data)
        
        lock.release()
       
    
    if mode == 'psql':  
        execute_values(cur,"INSERT INTO papers (insert_ts,source,id,title,keywords,abstract,journal) VALUES %s",
    datalist)
        conn.commit()
        conn.close()

f = None

if mode == 'xml':
    try:
        f = open('DATA/xml/' + '-'.join(query) + '/' + '-'.join(query) + '.xml','w')
        f.write('<?xml version="1.0" encoding="utf-8"?><root>')
        f.flush()
    except:
        os.makedirs('DATA/xml/' + '-'.join(query))
        f = open('DATA/xml/' + '-'.join(query) + '/' + '-'.join(query) + '.xml','w')
        f.write('<?xml version="1.0" encoding="utf-8"?><root>')

if mode == 'json':
    try:
        f = open('elasticsearch/json/' + '-'.join(query) + '/' + '-'.join(query) + '.json','w')
    except:
        os.makedirs('elasticsearch/json/' + '-'.join(query))
        f = open('elasticsearch/json/' + '-'.join(query) + '/' + '-'.join(query) + '.json','w')




batchsize = 2000
nproc = cpu_count()

    
if __name__ == '__main__':

    lock = Lock()
    index = Value('i', 0)
    
    print('--> Retrieving articles')
    while index.value < len(idlist):

        jobs = []
        
        for i in range(nproc):
            p = Process(target=worker, args=(index, lock, f))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        
    print('(', len(idlist), '/', len(idlist), ')\r', end='')
    print('')


if mode == 'xml':
    f.write('</root>')
    f.close()

if mode == 'json':
    f.close()


