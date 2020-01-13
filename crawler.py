import requests as req
from bs4 import BeautifulSoup
from article import Article

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity', 'retmax':1235}

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

f = open('json/articles.json','w')

batchsize = 100
niter = int(len(idlist)/batchsize)
rest = len(idlist)%batchsize

if rest == 0:
    k = 0
else:
    k = 1
    
index = 0
print('--> Retrieving articles')
for i in range(niter+k):
    
    index = batchsize*i
    print('(', index, '/', len(idlist), ')\r', end='')
    
    #params passed to efetch
    if i == niter+1:
        payload = {'db':'pubmed', 'id':idlist[index:index+rest], 'retmode':'xml'}
    else:
        payload = {'db':'pubmed', 'id':idlist[index:index+batchsize], 'retmode':'xml'}

    #efetch
    r = req.post('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', data=payload)
    
    #parsing
    soup = list(BeautifulSoup(r.text, "xml").PubmedArticleSet.children)[1::2]
    
    for j in soup:
    
        art = Article(j)
        f.write(art.json_line())

        
print('(', len(idlist), '/', len(idlist), ')\r', end='')
    
print('')
