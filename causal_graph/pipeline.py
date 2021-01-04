import random
import torch
import math

from transformers import AutoTokenizer, AutoModel


class BERT(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return self.model(**inputs).last_hidden_state[0][1:-1]     # [0] explaination:
                                                                   # The output of model(**x) is of shape (a,b,c) with a = batchsize,
                                                                   # which in our case equals 1 since we pass a single sentence,
                                                                   # b = max_lenght of the sentences in the batch and c = encoding_dim.
                                                                   # In case we want to use batchsize > 1 we need to change also the
                                                                   # other modules of the pipeline.
                                                                   # [1:-1] : we want to get rid of [CLS] and [SEP] tokens

class NER(torch.nn.Module):

    def __init__(self, encoding_dim, ner_dim):
        super().__init__()
        self.lin = torch.nn.Linear(encoding_dim, ner_dim)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return torch.cat((x, self.sm(self.lin(x))), 1)             # should I perform the softmax here already?
                                                                   # or maybe just keep the output of the linear layer as
                                                                   # embedding and perform the softmax externally?        
                                                                   
class Merge_entities(torch.nn.Module):

    def __init__(self):
        super.__init__()

    def forward(self, x): # depends on which tagging scheme/classification space we use for NER
        pass              # averaging the embeddings has proven to be a solid alternative to concatenation


class Entity_filter(torch.nn.Module):

    def __init__(self, ner_dim):
        super().__init__()
        self.ner_dim = ner_dim

    def forward(self, x):
        out = []
        pads = []
        for xx in x:
            if int(torch.argmax(xx[-self.ner_dim:])) != self.ner_dim - 1 :  
                out.append(xx)
            else :                                         # padding:
                #pads.append(torch.zeros(xx.size()[0]))    # the tokens that are not entities are padded to zeros tensors
                #pads[-1][-1] = 1                          # with a single 1 in the last position corresponding to the "O" tag.
                                                           # In addition they are grouped and moved to the end of the output.
                out.append(torch.zeros(xx.size()[0]))      # Alternatively we set everything to zero and keep propagating these
                                                           # zeros in the following layers, however for this to work we need
                                                           # to disable the bias in the Head_Tail layer.
        #return torch.stack(out+pads, dim=0) 
        return torch.stack(out, dim=0)
    
class Head_Tail(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = torch.nn.Linear(in_dim, out_dim, bias=False) # we disable the biases in order to map zeros in input to 
        self.tail = torch.nn.Linear(in_dim, out_dim, bias=False) # zeros in output

    def forward(self, x):
        head = self.head(x)                       # problem: the tensors that have been padded cause they weren't entities
        tail = self.tail(x)                       # don't have to be fed to the head/tail FFNN.
        return torch.stack([head,tail], dim=1)    # If we disable the biases, we can froget about this and just keep
                                                  # propoagating zero tensors
        


    
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

sents = ['Obesity is a cause for Diabetes Mellitus', 'Diabetes type 2a is common in obese people']
sent = ['Obesity is a cause for Diabetes Mellitus']
label = torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,0.,1.],[0.,0.,1.],[0.,0.,1.],[0.,0.,1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

model = torch.nn.Sequential(
    BERT(bert),
    NER(768,3),
    Entity_filter(3),
    Head_Tail(768+3,768+3)
)

x = sent
print(x)
for l in model:
    x = l(x)
    print(l)
    print(x)
    
#print(model(sent))
    
'''
sm = torch.nn.Softmax(dim=1)
out = sm(model(sent))

loss = torch.nn.MSELoss()
err = loss(out,label)
err.backward()
'''
