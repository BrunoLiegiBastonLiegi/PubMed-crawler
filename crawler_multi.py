import requests as req
from bs4 import BeautifulSoup
from article import Article
from multiprocessing import Process, Value, Lock

#params passed to esearch
payload = {'db':'pubmed', 'term':'cancer OR smoke', 'retmax':100000}

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




def worker(index, lock, f):

    lock.acquire()
    ind = index.value
    if ind == len(idlist):
        lock.release()
        return
    else:
        if ind+batchsize < len(idlist):
            print('(', ind, '/', len(idlist), ')\r', end='')
            index.value += batchsize
            lock.release()
            payload = {'db':'pubmed', 'id':idlist[ind:index.value], 'retmode':'xml'}
        else:
            print('(', ind, '/', len(idlist), ')\r', end='') 
            index.value = len(idlist)
            lock.release()
            payload = {'db':'pubmed', 'id':idlist[ind:], 'retmode':'xml'}
        
    #efetch
    r = req.post('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', data=payload)
    
    #parsing
    soup = list(BeautifulSoup(r.text, "xml").PubmedArticleSet.children)[1::2]

    for j in soup:
        art = Article(j)
        lock.acquire()
        f.write(art.json_line())
        lock.release()
        



batchsize = 1000
nproc = 3
f = open('json/articles.json','w')



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




