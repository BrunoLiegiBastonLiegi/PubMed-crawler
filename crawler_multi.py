import requests as req
from bs4 import BeautifulSoup
from article import Article
from multiprocessing import Process, Value, Lock, current_process, Queue, Array

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity', 'retmax':1000}

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




def worker(index, lock, articles):

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
        articles.put(art.json_line())




batchsize = 100
nproc = 2

if __name__ == '__main__':

    lock = Lock()
    articles = Queue()
    index = Value('i', 0)
    
    print('--> Retrieving articles')
    while index.value < len(idlist):

        jobs = []
        
        for i in range(nproc):
            p = Process(target=worker, args=(index, lock, articles))
            jobs.append(p)
            p.start()
            
        for proc in jobs:
            proc.join()

        
    print('(', len(idlist), '/', len(idlist), ')\r', end='')
    
    print('')



f = open('json/articles.json','w')
for a in articles:
    f.write(a)
