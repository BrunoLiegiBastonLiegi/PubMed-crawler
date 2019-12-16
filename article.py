

class Article(object):
    """PubMed Article in xml"""

    def __init__(self, soup):
        self.soup = soup
        self.get_id()
        self.get_title()
        self.get_abstract()
        self.get_keywords()
        
    def get_id(self):
        self.id = self.soup.PMID.text
        return self.id

    def get_title(self):
        self.title = self.soup.ArticleTitle.text
        return self.title

    def get_keywords(self):
        tmp = self.soup.KeywordList
        if(tmp == None):
            self.keys = 'Not avaliable'
        else:
            tmp = tmp.text
            self.keys = tmp[1:len(tmp)-1].replace('\n',', ')
        return self.keys

    def get_abstract(self):
        tmp = self.soup.Abstract
        if(tmp == None):
            self.abstract = 'Not avaliable'
        else:
            tmp = tmp.contents
            lenght = len(tmp)
            for i in range(lenght):
                tmp[i] = str(tmp[i]) 
            self.abstract = ' '.join(tmp[1:lenght-1]).replace('"','\'').replace('\n',' ')
        return self.abstract

    def json_line(self):
        line = '{\"index\":{\"_id\":\"'
        line = line + str(self.id) + '\"}}\n'                              # elasticsearch index
        line = line + '{\"PMID\":' + str(self.id)                          # PMID
        line = line + ',\"Title\":\"' + str(self.title)                    # title
        line = line + '\",\"Keywords\":\"' + str(self.keys)                # keywords
        line = line + '\",\"Abstract\":\"' + str(self.abstract) + '\"}\n'  # abstract
        return line
