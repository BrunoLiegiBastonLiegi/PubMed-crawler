import requests as req
from bs4 import BeautifulSoup

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity'}

#esearch
r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=payload)

#parsing Ids
soup = BeautifulSoup(r.text, "xml")
Id = soup.Id

idlist = []

while Id != None:
    idlist.append(int(Id.text))
    Id = Id.find_next('Id')
    
#params passed to efetch
payload = {'db':'pubmed', 'id':idlist, 'retmode':'xml'}

#efetch
r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=payload)

#parsing
soup = BeautifulSoup(r.text, "xml")

article = soup.find('PubmedArticle')
line = []
f = open('articles.json','w')


def write_line(f, line):
    return f.write('{\"index\":{\"_id\":\"'+str(line[0])+'\"}}\n'+'{\"PMID\":'+str(line[0])+',\"Title\":\"'+str(line[1])+'\",\"Abstract\":\"'+str(line[2]).replace('\n',' ').replace('"',' ')+'\"}\n')


for i in range(len(idlist)):
    line.append([article.PMID.text, article.ArticleTitle.text, article.Abstract.text])
    write_line(f, line[i])
    article = article.find_next('PubmedArticle')






