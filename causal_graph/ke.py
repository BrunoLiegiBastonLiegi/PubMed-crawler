import random
import torch
import math

from transformers import AutoTokenizer, AutoModel



class Pipeline(torch.nn.Module):

    def __init__(self, bert):
        super().__init__()

        self.sm = torch.nn.Softmax(dim=1)
        
        # BERT
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(bert)
        self.pretrained_model = AutoModel.from_pretrained(bert)
        self.bert_dim = 768  # BERT encoding dimension
        
        # NER
        self.ner_dim = 10  # dimension of NER tagging scheme
        self.ner_lin = torch.nn.Linear(self.bert_dim, self.ner_dim)

        # NED
        self.ned_dim = 300  # dimension of the KB graph embedding space
        self.ned_lin = torch.nn.Linear(self.bert_dim + self.ner_dim, self.ned_dim)

        # Head-Tail
        self.ht_dim = 128  # dimension of head/tail embedding
        self.h_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        self.t_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)

    def forward(self, x):
        x = self.BERT(x)
        ner = self.NER(x)                                           # this is the output of the linear layer, should we use this as
        x = torch.cat((x, self.sm(ner)), 1)                                  # as embedding or rather the softmax of this?
        
        # remove non-entity tokens, before this we need to merge multi-token entities
        tmp = []
        for xx in x:
            if int(torch.argmax(xx[-self.ner_dim:])) != self.ner_dim - 1 :
                tmp.append(xx)
        x = torch.stack(tmp, dim=0)

        ned = self.NED(x)
        x = torch.cat((x, ned), 1)
        x, y = self.HeadTail(x)

        # building candidate pairs
        for 
        

    def BERT(self, x):
        inputs = self.pretrained_tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return self.pretrained_model(**inputs).last_hidden_state[0][1:-1]     # [0] explaination:
                                                                   # The output of model(**x) is of shape (a,b,c) with a = batchsize,
                                                                   # which in our case equals 1 since we pass a single sentence,
                                                                   # b = max_lenght of the sentences in the batch and c = encoding_dim.
                                                                   # In case we want to use batchsize > 1 we need to change also the
                                                                   # other modules of the pipeline, for example using convolutional
                                                                   # layers in place of linear ones.
                                                                   # [1:-1] : we want to get rid of [CLS] and [SEP] tokens

    def NER(self, x):
        return self.ner_lin(x)

    def NED(self, x):
        return self.ned_lin(x) # it's missing the dot product in the graph embedding space, the idea would be find the closest
                               # concepts in embedding space and then return the closest in a greedy approach, or the closests
                               # with beam search

    def HeadTail(self, x):
        h = self.h_lin(x)
        t = self.t_lin(x)
        return (h,t)

    def Biaffine(self, x, y):
