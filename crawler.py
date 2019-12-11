import requests as req
from bs4 import BeautifulSoup

#params passed to esearch
payload = {'db':'pubmed', 'term':'ageing OR longevity'}

#esearch
r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params=payload)

#parsing Ids
soup = BeautifulSoup(r.text, "xml")
idlist_raw = soup.IdList.contents

idlist = []
i = 0
while i < 0.5*len(idlist_raw)-1:

    idlist.append(int(idlist_raw[2*i+1].contents[0]))
    i = i+1

#params passed to efetch
payload = {'db':'pubmed', 'id':idlist}

#efetch
r = req.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=payload)

