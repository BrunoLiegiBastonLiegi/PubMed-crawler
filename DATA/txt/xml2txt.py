import sys, re
sys.path.append('../../')

from preprocessor import Preprocessor

f = sys.argv[1]
p = Preprocessor(load=f, out_file=re.search('[A-Za-z0-9-_]+\.xml', f).group().replace('.xml', '.txt'))
p.preprocess()
