import requests as req
from bs4 import BeautifulSoup
from article import Article
<<<<<<< HEAD
from multiprocessing import Process, Value, Lock

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity', 'retmax':100000}
=======
from multiprocessing import Process, Value, Lock, current_process, Queue, Array

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity', 'retmax':1000}
>>>>>>> 91a65eb5fd5802895c12e3b04627429d5a783edb

#esearch
print('--> Searching Pubmed')
r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=payload)

#parsing Ids
print('--> Retrieving Ids')
soup = BeautifulSoup(r.text, "xml")
Id = soup.Id

idlist = []

while Id != None:
    idlist.append(int(Id.text))
    Id = Id.find_next('Id')




<<<<<<< HEAD
def worker(index, lock, f):
=======
def worker(index, lock, articles):
>>>>>>> 91a65eb5fd5802895c12e3b04627429d5a783edb

    lock.acquire()
    ind = index.value
    if ind == len(idlist):
        return
    else:
        if ind+batchsize < len(idlist):
            print('(', ind, '/', len(idlist), ')\r', end='')
            index.value += batchsize
            payload = {'db':'pubmed', 'id':idlist[ind:index.value], 'retmode':'xml'}
        else:
            print('(', ind, '/', len(idlist), ')\r', end='')
            index.value = len(idlist)
            payload = {'db':'pubmed', 'id':idlist[ind:], 'retmode':'xml'}
    lock.release()
        
    #efetch
    r = req.post('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', data=payload)
    
    #parsing
    soup = list(BeautifulSoup(r.text, "xml").PubmedArticleSet.children)[1::2]

    for j in soup:
        art = Article(j)
<<<<<<< HEAD
        lock.acquire()
        f.write(art.json_line())
        lock.release()
        



batchsize = 1000
nproc = 2
f = open('json/articles.json','w')


=======
        articles.put(art.json_line())




batchsize = 100
nproc = 2
>>>>>>> 91a65eb5fd5802895c12e3b04627429d5a783edb

if __name__ == '__main__':

    lock = Lock()
<<<<<<< HEAD
=======
    articles = Queue()
>>>>>>> 91a65eb5fd5802895c12e3b04627429d5a783edb
    index = Value('i', 0)
    
    print('--> Retrieving articles')
    while index.value < len(idlist):

        jobs = []
        
        for i in range(nproc):
<<<<<<< HEAD
            p = Process(target=worker, args=(index, lock, f))
            jobs.append(p)
            p.start()

        #while not articles.empty():
            #f.write(articles.get())
            
        for proc in jobs:
            proc.join()
=======
            p = Process(target=worker, args=(index, lock, articles))
            jobs.append(p)
            p.start()
            
        for proc in jobs:
            proc.join()

>>>>>>> 91a65eb5fd5802895c12e3b04627429d5a783edb
        
    print('(', len(idlist), '/', len(idlist), ')\r', end='')
    
    print('')



<<<<<<< HEAD

=======
f = open('json/articles.json','w')
for a in articles:
    f.write(a)
>>>>>>> 91a65eb5fd5802895c12e3b04627429d5a783edb
