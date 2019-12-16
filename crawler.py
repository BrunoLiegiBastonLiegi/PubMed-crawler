import requests as req
from bs4 import BeautifulSoup
from article import Article

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity', 'retmax':10}

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

f = open('articles.json','w')

print('--> Retrieving abstratcs and titles')
for i in range(len(idlist)):
    
    #params passed to efetch
    payload = {'db':'pubmed', 'id':idlist[i], 'retmode':'xml'}

    #efetch
    r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=payload)

    #parsing
    soup = BeautifulSoup(r.text, "xml")

    art = Article(soup)
    
    f.write(art.json_line())


