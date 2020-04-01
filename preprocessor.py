import re
from nltk.corpus import stopwords
stop = stopwords.words("english")
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

from multiprocessing import cpu_count, Pool
from random import randrange, shuffle
from bs4 import BeautifulSoup


class Preprocessor(object):

    def __init__(self, text=None, load=None, out_file=None):
        print('Splitting text')
        self.file = out_file
        if self.file != None:
            with open(self.file, 'w') as f:
                f.write('')
        self.nproc = cpu_count()
        if text != None:
            self.text = self.split_text(text)
        if load != None:
            self.load_file(load)

    def split_text(self, text):
        m_bak = 0
        tmp = []
        for m in re.finditer('\n', text):
            tmp.append(text[m_bak:m.start()])
            m_bak = m.end()
        tmp.append(text[m_bak:])
        return tmp
        
    def preprocess_worker(self, text):
        sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        WORD = re.compile(r'\w+')
        for i in range(len(sents)):
            if sents[i] == '':
                del sents[i]
            elif self.tokenize == True:
                tmp = []
                for w in WORD.finditer(sents[i]):
                    if self.remove_stop == True:
                        word = w.group(0).lower()
                        if word not in stop:
                            if self.lemma == True:
                                tmp.append(lemmatizer.lemmatize(word)) if self.lower == True else tmp.append(lemmatizer.lemmatize(w.group(0)))
                            else:
                                tmp.append(word) if self.lower == True else tmp.append(w.group(0))
                    else:
                        tmp.append(w.group(0))
                sents[i] = tmp
                
        if self.file != None:
            with open(self.file, 'a') as f:
                for sent in sents:
                    f.write(str(sent))
                    f.write('\n')
        return sents
        
    def preprocess(self, tokenize=False, remove_stop=False, lower=False, lemma=False):
        print('Preprocessing')
        self.tokenize = tokenize
        self.remove_stop = remove_stop
        self.lower = lower
        self.lemma = lemma
        p = Pool(processes=self.nproc)
        prep_text = p.map(self.preprocess_worker, self.text)
        #self.text = prep_text
        return [sent for chunk in prep_text for sent in chunk]
    
    def load_file(self, file):
        with open(file) as f:
            soup = BeautifulSoup(f.read(), "xml")
        self.text = []
        for abs in soup.find_all('abstract'):
            text = re.sub('\</*abstract\>', '', str(abs))
            if len(text) > 13:
                self.text.append(text)
        
    '''        
    def rand_sent(self):
        tmp = self.text[randrange(len(self.text))]
        return tmp[randrange(len(tmp))]
    
    def split_sets(self, c=0.3):
        train_set = shuffle([sent for sent in sents for sents in self.text])
        test_set = [train_set.pop(randrange(len(train_set))) for i in int(c*len(train_set))]
        return train_set, test_set
    '''
                
