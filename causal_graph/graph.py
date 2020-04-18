
class Vertex:
    """Vertex of graph represented as an adjacency list"""
    def __init__(self, v, edges, neighbors):
        self.me = v
        
        assert type(edges)==list, "Expected type:list for edges"
        assert type(neighbors)==list, "Expected type:list for neighbors"
                
        self.edges = edges
        self.neighbors = neighbors

    def set_me(self, me):
        self.me = me

    def set_edges(self, edges):
        self.edges = edges

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
        
'''
class Edge:
    def __init__(self, text, source, target):
        self.me = text
        self.source = source
        self.target = target
'''

import graph_tool as gt
from graph_tool.all import graph_draw, graphviz_draw, minimize_blockmodel_dl, fruchterman_reingold_layout, arf_layout
import re, random
import numpy as np
from tensorflow.keras.models import Model
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.layers import Embedding, Input, Reshape, Dot, Dense

class Graph(object):

    def __init__(self, vertices=None):
        self.g = gt.Graph()
        self.verts_text = self.g.new_vertex_property("string")
        self.edges_text = self.g.new_edge_property("string")
        self.v_mapping = {}
        self.causal_preds = ['CAUSES','PREVENTS','DISRUPTS','INHIBITS','PREDISPOSES','PRODUCES']
        self.bidir_preds = ['COEXISTS_WITH','ASSOCIATED_WITH']
        if vertices!=None:
            [self.add_vertex(v) for v in vertices]

    def add_vertex(self, vertex):
        try:
            v_i = self.v_mapping[vertex.me]
        except:
            self.v_mapping[vertex.me] = self.g.add_vertex()
            v_i = self.v_mapping[vertex.me]
            self.verts_text[v_i] = vertex.me
                    
        v_f = []
        for n in vertex.neighbors:
            try:
                v_f.append(self.v_mapping[n])
            except:
                self.v_mapping[n] = self.g.add_vertex()
                v_f.append(self.v_mapping[n])
                #self.verts_text[v_f[-1]] = n
                self.verts_text[self.v_mapping[n]] = n
                
        #e = [self.add_edge(vertex.edges[j], v_i, v_f[j]) for j in range(len(vertex.edges))]
        for j in range(len(vertex.edges)):
            self.add_edge(vertex.edges[j], v_i, v_f[j], dir='bi') if vertex.edges[j] in self.bidir_preds else self.add_edge(vertex.edges[j], v_i, v_f[j])
        
    def add_edge(self, edge, gt_v1, gt_v2, dir='straight'):
        assert dir in {'straight', 'inverted', 'bi'}, 'Unsupported edge direction'
        if dir == 'bi':
            e = self.g.add_edge(gt_v1, gt_v2)
            self.edges_text[e] = edge
            e = self.g.add_edge(gt_v2, gt_v1)
            self.edges_text[e] = edge
        else:
            if dir == 'straight':
                e = self.g.add_edge(gt_v1, gt_v2)
            elif dir == 'inverted':
                e = self.g.add_edge(gt_v2, gt_v1)
            self.edges_text[e] = edge

    def adjacency_list(self):
        dict = {}
        for v in self.g.vertices():
            dict[v] = []
            for n in v.out_neighbors():
                dict[v].append(n)
        return dict

    def clean(self):
        v_list = []
        for v in self.g.vertices():
            if v.out_degree() + v.in_degree() < 1:
                v_list.append(v)
        self.g.remove_vertex(v_list)
    
    def redundancy(self, k=2):
        redundancy_map = self.g.new_edge_property("bool") # have to use filtering cause removal was causing core dumped
        for v in self.g.vertices():
            dict = {}
            for e in v.out_edges():
                t = e.target()
                try:
                    dict[self.verts_text[t]].append(e)
                except:
                    dict[self.verts_text[t]] = [e]
            for i in dict.values():
                if len(i) < k:
                    for j in i:
                        redundancy_map[j] = False
                else:
                    for j in i:
                        redundancy_map[j] = True
                        
        self.g.set_edge_filter(redundancy_map)
        self.clean()

    def word_embedding(self, model, target):
        embedding_map = self.g.new_vertex_property("bool")
        WORD = re.compile(r'\w+')
        for v in self.g.vertices():
            ent = WORD.findall(self.verts_text[v])
            if model.wv.n_similarity(ent, target) > 0.4:
                embedding_map[v] = True
            else:
                embedding_map[v] = False

    def filter_by(self, method, **kwargs):
        assert method in ['causal','co-occurrence','word_embedding','redundancy'], 'Unsupported pruning method'
        if method == 'causal':
            return self.causal()
        if method == 'co-occurrence':
            return self.co_occurrence(**kwargs)
        if method == 'word_embedding':
            return self.word_embedding(**kwargs)
        if method == 'redundancy':
            return self.redundancy(**kwargs)
            

    def causal(self):
        causal_map = self.g.new_edge_property("bool")
        for e in self.g.edges():
            if self.edges_text[e] in self.causal_preds or self.edges_text[e] in self.bidir_preds:
                causal_map[e] = True
            else:
                causal_map[e] = False
        self.g.set_edge_filter(causal_map)
        self.clean()

    def co_occurrence(self, threshold):
        co_map = self.g.new_edge_property("bool")
        for v in self.g.vertices():
            for n in v.out_neighbors():
                edges = self.g.edge(v, n, all_edges=True)
                #print(float(len(edges)/(self.g.get_total_degrees([v])[0] + self.g.get_total_degrees([n])[0])))
                if float(len(edges)/(self.g.get_total_degrees([v])[0] + self.g.get_total_degrees([n])[0])) > threshold:
                    for e in edges:
                        co_map[e] = True
                else:
                    for e in edges:
                        co_map[e] = False
                        
        self.g.set_edge_filter(co_map)
        self.clean()

    def random_walk(self, v=None, length=20):
        if v == None:
            v = random.choice(self.g.get_vertices())
        start = v
        walk = [v]
        for i in range(length):
            neighbors = [self.g.vertex(n) for n in self.g.get_out_neighbors(walk[-1])]
            walk.append(random.choice(neighbors)) if len(neighbors) != 0 else walk.append(start)
        return walk

    def deep_walk(self, walk_length=20, window=5, embedding_dim=150):
        corpus = []
        vocab = self.g.get_vertices()
        for i in range(5):
            random.shuffle(vocab)
            for v in vocab:
                corpus.append([self.g.vertex_index[r] for r in self.random_walk(v=v)])

        # skipgram with negative sampling
        vocab_size = len(vocab)
        table = make_sampling_table(vocab_size)
        couples = []
        labels = []
        for sent in corpus:
            tmp1, tmp2 = skipgrams(sent, vocab_size, window_size=window)
            couples += tmp1
            labels += tmp2
            
        target, context = zip(*couples)  # unpack couples in two lists: the list of targets and the list of the relative context
        target = np.array(target)
        context = np.array(context)
        # keras input layer
        input_target = Input((1,))
        input_context = Input((1,))
        # keras embedding layer
        embedding = Embedding(vocab_size, embedding_dim, input_length=1, name='embedding')
        output_target = embedding(input_target)
        #output_target = Reshape((embedding_dim, 1))(output_target) # reshape needed for cosine similarity
        output_context = embedding(input_context)
        #output_context = Reshape((embedding_dim, 1))(output_context)
        similarity = Dot(normalize=True, axes=1)([output_target, output_context])
        #similarity = Reshape((1,))(similarity)
        # final output
        output = Dense(1, activation='sigmoid')(similarity)

        model = Model(inputs=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        # training
        model.fit(x=[target,context], y=np.asarray(labels), batch_size=32, epochs=20, validation_split=0.2, workers=12, use_multiprocessing=True)
        
    '''
    def skipgrams(self, sent, window, vocab):
        couples = []
        labels = []
        for i in range(len(sent)):
            win = []
            for j in range(1, window+1):
                try:
                    win.append(sent[i-j])    #beware list[-1], list[-3], ecc... all exist!
                except:
                    pass
                try:
                    win.append(sent[i+j])
                except:
                    pass
            for w in win:
                couples.append([sent[i], w])
                labels.append(1)
                ran = random.choice(vocab)
                while ran in win:
                    ran = random.choice(vocab)
                couples.append([sent[i], ran])
                labels.append(0)

        return couples, labels
    '''                   
        
    def merge_vertices(self, v1, v2):         # not working, why??????
        del_list = []
        for e in v1.out_edges():
            self.add_edge(self.edges_text[e], v2, e.target())
            del_list.append(e)
        for e in v1.in_edges():
            self.add_edge(self.edges_text[e], e.source(), v2)
            del_list.append(e)
        for e in reversed(sorted(del_list)):
            self.g.remove_edge(e)
        self.g.remove_vertex(v1)

    def json(self):
        nodes = []
        links = []
        for v in self.g.vertices():
            nodes.append({"id": self.verts_text[v], "group": 1})
        for e in self.g.edges():
            links.append({"source": self.verts_text[e.source()], "target": self.verts_text[e.target()], "value": 1})
        with open('graph.json', 'w') as f:
            f.write('{\n\t\"nodes\": [\n')
            for n in range(len(nodes)):
                f.write('\t\t')
                f.write(str(nodes[n]))
                if n != len(nodes) - 1:
                    f.write(',')
                f.write('\n')
            f.write('\t],\n\t\"links\": [\n')
            for l in range(len(links)):
                f.write('\t\t')
                f.write(str(links[l]))
                if n != len(links) - 1:
                    f.write(',')
                f.write('\n')
            f.write('\n\t]\n}')
        
    def draw(self):
        
        vprops = {
            'text': self.verts_text,
            'font_size':16,
            'size':1,
            'text_color':'black'
        }
        
        eprops = {
            #'text': self.edges_text,
            'pen_width':2,
            'end_marker':'arrow',
            'marker_size':12,
        }
        
        graph_draw(self.g, vprops=vprops, eprops=eprops, output_size=(2000, 2000))
        #graphviz_draw(self.g, layout='sfdp', vprops=vprops, eprops=eprops, size=(25,25))
        '''
        dot = Digraph(comment='Test')
        for v in self.g.vertices():
            dot.node(self.verts_text[v])
            for n in v.out_neighbors():
                dot.edge(self.verts_text[v],self.verts_text[n])

        dot.render(view=True)
        '''
