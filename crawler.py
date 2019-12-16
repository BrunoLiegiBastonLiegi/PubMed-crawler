import requests as req
from bs4 import BeautifulSoup

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity', 'retmax':100}

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

def write_line(f, line):
    return f.write('{\"index\":{\"_id\":\"'+str(line[0])+'\"}}\n'+'{\"PMID\":'+str(line[0])+',\"Title\":\"'+str(line[1])+'\",\"Abstract\":\"'+str(line[2]).replace('\n',' ').replace('"',' ')+'\"}\n')

print('--> Retrieving abstratcs and titles')
for i in range(len(idlist)):
    
    #params passed to efetch
    payload = {'db':'pubmed', 'id':idlist[i], 'retmode':'xml'}

    #efetch
    r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=payload)

    #parsing
    soup = BeautifulSoup(r.text, "xml")

    if soup.Abstract != None:
        line = [soup.PMID.text, soup.ArticleTitle.text, soup.Abstract.text]
    else:
        line = [soup.PMID.text, soup.ArticleTitle.text, 'Abstract not avaliable']
    write_line(f, line)






