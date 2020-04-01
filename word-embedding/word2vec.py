import gensim.models
import sys
sys.path.append('../')

from preprocessor import Preprocessor

print('loading data')
p = Preprocessor(load=sys.argv[1])
data = p.preprocess(tokenize=True, remove_stop=True)
    
print('building word2vec model')
model = gensim.models.Word2Vec(sentences=data, window=6, size=300, min_count=7, iter=20, workers=12)

model.save('models/word2vec_window=6_size=300_min_count=7_iter=20.model')
print('model saved to models/')
