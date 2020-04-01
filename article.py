

class Article(object):
    """PubMed Article in xml"""

    def __init__(self, soup):
        self.soup = soup
        self.get_id()
        self.get_title()
        self.get_abstract()
        self.get_keywords()
        self.get_journal()
        self.get_date()
        
    def get_id(self):
        self.id = self.soup.PMID.text
        return self.id

    def get_title(self):
        tmp = self.soup.ArticleTitle
        if tmp != None:
            self.title = tmp.text.replace('"','\'').replace('\n',' ').replace('\\','').replace('&',' and ').replace('<', ' lower than ').replace('>', ' greater than ').replace(u'\xa0',u'')
        else:
            self.title = self.soup.BookTitle.text.replace('"','\'').replace('\n',' ').replace('\\','').replace('&',' and ').replace('<', ' lower than ').replace('>', ' greater than ').replace(u'\xa0',u'')
        return self.title

    def get_keywords(self):
        tmp = self.soup.KeywordList
        if(tmp == None):
            self.keys = 'Not avaliable'
        else:
            tmp = tmp.text
            self.keys = tmp[1:len(tmp)-1].replace('\n',', ').replace('"','\'').replace('\t', '').replace('\\','').replace('&',' and ').replace('<','').replace('>','')
        return self.keys

    def get_abstract(self):
        tmp = self.soup.Abstract
        if(tmp == None):
            self.abstract = 'Not avaliable'
        else:
            #tmp = tmp.contents
            tmp = tmp.text
            #length = len(tmp)
            #for i in range(length):
                #tmp[i] = str(tmp[i]) 
            #self.abstract = ' '.join(tmp[1:lenght-1]).replace('"','\'').replace('\n',' ').replace('\\','').replace('\t', '')
            self.abstract = tmp.replace('"','\'').replace('\n',' ').replace('\\','').replace('\t', '').replace('<', ' lower than ').replace('>', ' greater than ').replace('&',' and ').replace(u'\xa0',u'')
        return self.abstract

    def get_journal(self):
        tmp = self.soup.Journal
        if tmp != None:
            self.journal = tmp.Title.text.replace('\n',' ').replace('"','\'')
        else:
            self.journal = self.soup.BookTitle.text.replace('\n',' ').replace('"','\'')
        return self.journal

    def get_date(self):
        tmp = self.soup.PubDate
        month = tmp.Month
        day = tmp.Day
        if month != None:
            self.date = str(month.text) + ' '
            if day != None:
                self.date = self.date + str(day.text) + ' '
        else:
            self.date = ''
        year = tmp.Year
        if year != None:
            self.date =  self.date + str(year.text)
        else:
            self.date = str(tmp.MedlineDate.text)
        return self.date
    

    def line(self, format='xml'):
        return (self.xml_line(), self.json_line())

    def xml_line(self):
        line = '<article><title>' + str(self.title) + '</title>'
        line += '<abstract>' + str(self.abstract) + '</abstract>'
        line += '<keywords>' + str(self.keys) + '</keywords></article>'
        return line
        
    def json_line(self):
        line = '{\"index\":{\"_id\":\"'
        line = line + str(self.id) + '\"}}\n'                              # elasticsearch index
        line = line + '{\"PMID\":' + str(self.id)                          # PMID
        line = line + ',\"Title\":\"' + str(self.title)                    # title
        line = line + '\",\"Journal\":\"' + str(self.journal)              # journal
        line = line + '\",\"Date\":\"' + str(self.date)                    # date
        line = line + '\",\"Keywords\":\"' + str(self.keys)                # keywords
        line = line + '\",\"Abstract\":\"' + str(self.abstract) + '\"}\n'  # abstract
        return line
        
            
