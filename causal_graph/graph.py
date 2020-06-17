import warnings

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
import networkx as nx
import re, random, json
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.layers import Embedding, Input, Reshape, Dot, Dense
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat


class Graph(ABC):

    def __init__(self, vertices=None):
        self.g = None
        self.vertex2label = None
        self.edge2label = None
        self.label2vertex = None
        self.vertex2cluster = {}
        self.embedding = {}
        self.bidir_preds = ['COEXISTS_WITH','ASSOCIATED_WITH']
        self.causal_preds = ['CAUSES','PREVENTS','DISRUPTS','INHIBITS','PREDISPOSES','PRODUCES']
        self.init(vertices)
        
    @abstractmethod
    def init(self, vertices):
        pass
        
    @abstractmethod
    def add_vertex(self, vertex):
        pass
        
    @abstractmethod   
    def add_edge(self, edge, v1, v2, dir='straight'):
        pass

    @abstractmethod
    def get_vertex(self, v):
        pass

    @abstractmethod
    def get_vertices(self):
        pass
    
    @abstractmethod
    def get_edges(self, v1, v2, dir='out'):
        pass

    @abstractmethod
    def get_neighbors(self, v, dir='out'):
        pass

    @abstractmethod
    def remove_vertices(self, vl):
        pass

    @abstractmethod
    def remove_edges(self, el):
        pass
    
    @abstractmethod
    def draw(self):
        pass

    def get_degree(self, v, dir='all'):
        assert dir in ['in', 'out', 'all'], 'Unsupported direction'
        deg = 0
        for e in self.get_edges(v, dir=dir):
            deg += e[3]
        return deg

    def clean(self):
        vl = []
        for v in self.get_vertices():
            if self.get_degree(v) < 1:
                vl.append(v)
        self.remove_vertices(vl)  

    def degree(self, d=2):
        del_list = []
        for v in self.get_vertices():
            if self.get_degree(v) < d:
                del_list.append(v)
        self.remove_vertices(del_list)
        self.clean()
        
    def redundancy(self, k=2): # to be fixed!
        del_list = []
        for e in self.get_edges():
            if e[3] < k:
                del_list.append(e)

        self.remove_edges(del_list)
        self.clean()

    def filter_by(self, method, **kwargs):
        assert method in ['degree','causal','co-occurrence','word_embedding','redundancy'], 'Unsupported pruning method'
        if method == 'causal':
            return self.causal()
        if method == 'co-occurrence':
            return self.co_occurrence(**kwargs)
        if method == 'word_embedding':
            return self.word_embedding(**kwargs)
        if method == 'redundancy':
            return self.redundancy(**kwargs)
        if method == 'degree':
            return self.degree(**kwargs)
            
    def causal(self):
        del_list = []
        for e in self.get_edges():
            if e[2] in self.causal_preds or e[2] in self.bidir_preds:
                pass
            else:
                del_list.append(e)
        self.remove_edges(del_list)
        self.clean()

    def co_occurrence(self, threshold): # to be fixed!
        del_list = []
        for v in self.get_vertices():
            for n in self.get_neighbors(v):
                edges = self.get_edges(v, n)
                #print(float(len(edges)/(self.g.get_total_degrees([v])[0] + self.g.get_total_degrees([n])[0])))
                if float(len(edges)/(self.get_degree(v) + self.get_degree(n))) > threshold:
                    pass
                else:
                    [del_list.append(e) for e in edges]
        self.remove_edges(del_list)
        self.clean()

    def random_walk(self, v=None, length=20, with_edges=False):
        if v == None:
            v = random.choice(self.get_vertices())
        #start = v
        walk = [v]
        for i in range(length):
            neighbors = []
            for e in self.get_edges(walk[-1], dir='out'):
                for i in range(e[3]):
                    neighbors.append([e[2], e[1]])
            '''
            for n in self.get_neighbors(walk[-1]):
                for e in self.get_edges(walk[-1], n):
                    for i in range(e[3]):
                        neighbors.append(n)
            '''
            if len(neighbors) != 0:
                random.shuffle(neighbors)
                if with_edges == True:
                    [walk.append(i) for i in random.choice(neighbors)]
                else:
                    walk.append(random.choice(neighbors)[1])
            else:
                #walk.append(start)
                break
        return walk

    def skipgrams(self, sent, window_size, vocab_size, negative_samples=5):
        couples = []
        labels = []
        for i in range(len(sent)-1):
            window = [sent[j] for j in range(i, min(i+window_size, len(sent)))]
            for v in window[1:]:
                couples.append([window[0],v])
                labels.append(1)
                for k in range(negative_samples):
                    r = random.randint(0, vocab_size-1)
                    while r in window:
                        r = random.randint(0, vocab_size-1)
                    couples.append([window[0],r])
                    labels.append(0)
        return (couples, labels)
            

    def deep_walk(self, walk_length=20, window=10, walks_per_node=10, embedding_dim=100, with_edges=False):
        # preparing vocabulary and corpus
        corpus = []
        vocab = self.get_vertices()
        edges_vocab = {e : int(len(vocab)+i) for i,e in enumerate(self.causal_preds+self.bidir_preds)} # using also edges (only causal ones) with their relative label in the embedding
        for n in range(walks_per_node):
            print('Generating Walks:', int(n/walks_per_node*100), '%\r', end='')
            #random.shuffle(vocab)
            starts = [random.choice(vocab) for j in range(len(vocab))]
            p = Pool(processes=12)
            sents = p.starmap(self.random_walk, zip(starts, repeat(walk_length), repeat(with_edges)))
            p.close()
            p.join()
            for s in sents:
                if with_edges == True:
                    for i, e in enumerate(s[1::2]):
                        try:
                            s[2*i+1] = edges_vocab[e]
                        except:
                            edges_vocab[e] = len(vocab) + len(edges_vocab.keys())
                            s[2*i+1] = edges_vocab[e]
                corpus.append(s)
        print('Generating Walks:', 100, '% ')

        if with_edges == True:
            for val in edges_vocab.values():
                vocab.append(val)
        
        # skipgram with negative sampling
        # usually the training samples are generated depending on the frequency of a particular word in the corpus, for example the word 'the' will appear a lot in a text and therefore it will have small probability to be used for a training sample.
        # in our case however, the edges are the ones that will appear frequently in the corpus but we don't want to get rid of them since we are trying to take them into account for embedding and compare the result with edge-independent embedding
        vocab_size = len(vocab)
        table = make_sampling_table(vocab_size)
        couples = []
        labels = []
        for sent in corpus:
            if with_edges == True:
                tmp1, tmp2 = self.skipgrams(sent, window, vocab_size)
            else :
                tmp1, tmp2 = skipgrams(sent, vocab_size, window_size=window)
            couples += tmp1
            labels += tmp2
            
        target, context = zip(*couples)  # unpack couples in two lists: the list of targets and the list of the relative contexts
        target = np.array(target)
        context = np.array(context)
        # keras input layer
        input_target = Input((1,))
        input_context = Input((1,))
        # keras embedding layer
        embedding = Embedding(vocab_size, embedding_dim, input_length=1, name='embedding')
        output_target = embedding(input_target)
        output_target = Reshape((embedding_dim, 1))(output_target) # reshape needed for cosine similarity
        output_context = embedding(input_context)
        output_context = Reshape((embedding_dim, 1))(output_context)
        similarity = Dot(normalize=True, axes=1)([output_target, output_context])
        similarity = Reshape((1,))(similarity)
        # final output
        output = Dense(1, activation='sigmoid')(similarity)

        model = Model(inputs=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        # training
        model.fit(x=[target,context], y=np.asarray(labels), batch_size=32, epochs=1, validation_split=0.2, workers=12, use_multiprocessing=True)

        weights = embedding.get_weights()
        if with_edges == True:
            vocab = vocab[:-len(edges_vocab.keys())]
        for n in vocab:
            self.embedding[n] = weights[0][n]

        return self.embedding

    def k_means(self, n_clusters=None, elbow_range=(2,11)):
        x = np.array([v for v in self.embedding.values()])
        sse = {}
        kmeans = {}
        if n_clusters != None:
            kmeans = KMeans(n_clusters=n_clusters).fit(x)
        else:
            for i in range(elbow_range[0], elbow_range[1]):
                kmeans[i] = KMeans(n_clusters=i).fit(x)
                print('K-Means with', i, 'clusters done')
                sse[i] = kmeans[i].inertia_
            plt.figure()
            plt.plot(list(sse.keys()), list(sse.values()))
            plt.xlabel("Number of clusters")
            plt.ylabel("SSE")
            plt.show()
            n = int(input('Insert optimal number of clusters: '))
            kmeans = kmeans[n]

        for i in self.get_vertices():
            self.vertex2cluster[i] = int(kmeans.labels_[i])
        return self.vertex2cluster
        
    def json(self, file='graph.json'):
        nodes = []
        links = []
        for v in self.get_vertices():
            #nodes.append(json.dumps({'id': self.get_vertex(v), 'cluster': self.vertex2cluster[v], 'category': 'cat', 'degree': self.get_degree(v)}))
            nodes.append(json.dumps({'id': self.get_vertex(v), 'cluster': 0, 'category': 'cat', 'degree': self.get_degree(v)}))
        for e in self.get_edges():
            links.append(json.dumps({'source': self.get_vertex(e[0]), 'target': self.get_vertex(e[1]), 'label': e[2], 'weight': e[3]}))
        with open(file, 'w') as f:
            f.write('{\n\t\"nodes\": [\n')
            for n in range(len(nodes)):
                f.write('\t\t')
                f.write(nodes[n])
                if n != len(nodes) - 1:
                    f.write(',')
                f.write('\n')
            f.write('\t],\n\t\"links\": [\n')
            for l in range(len(links)):
                f.write('\t\t')
                f.write(links[l])
                if l != len(links) - 1:
                    f.write(',')
                f.write('\n')
            f.write('\n\t]\n}')
        





class Graph_tool(Graph):

    def init(self, vertices=None):
        
        self.g = gt.Graph()
        self.label2vertex = {}
        self.vertex2label = self.g.new_vertex_property("string")
        self.edge2label = self.g.new_edge_property("string")
        self.edge2weight = self.g.new_edge_property("int")
        if vertices !=None:
            [self.add_vertex(v) for v in vertices]

    def add_vertex(self, vertex):
        try:
            v_i = self.label2vertex[vertex.me]
        except:
            self.label2vertex[vertex.me] = self.g.add_vertex()
            v_i = self.label2vertex[vertex.me]
            self.vertex2label[v_i] = vertex.me
                    
        v_f = []
        for n in vertex.neighbors:
            try:
                v_f.append(self.label2vertex[n])
            except:
                self.label2vertex[n] = self.g.add_vertex()
                v_f.append(self.label2vertex[n])
                self.vertex2label[self.label2vertex[n]] = n
                
        for j in range(len(vertex.edges)):
            self.add_edge(vertex.edges[j], v_i, v_f[j], dir='bi') if vertex.edges[j] in self.bidir_preds else self.add_edge(vertex.edges[j], v_i, v_f[j])

    def add_edge(self, edge, v1, v2, dir='straight'):
        assert dir in {'straight', 'inverted', 'bi'}, 'Unsupported edge direction'
        if dir == 'inverted':
            return self.add_edge(edge, v2, v1)
        w = 1
        for e in self.g.edge(v1, v2, all_edges=True):
            if edge == self.edge2label[e]:
                w = self.edge2weight[e] + 1
                break
        if w == 1:
            e = self.g.add_edge(v1, v2)
            self.edge2label[e] = edge
            self.edge2weight[e] = w
        else:
            self.edge2weight[e] = w
        if dir == 'bi':
            return self.add_edge(edge, v2, v1)

    def get_vertex(self, v):
        if type(v) == str:
            return self.label2vertex[v]
        if type(v) == int:
            return self.vertex2label[v]

    def get_vertices(self):
        return [int(v) for v in self.g.get_vertices()]
        
    def get_edges(self, v1=None, v2=None, dir='out'):
        if v1 == None:
            e = [ [self.g.vertex_index[i.source()], self.g.vertex_index[i.target()], self.edge2label[i], self.edge2weight[i]] for i in self.g.edges() ]
            return e    
        else:
            assert type(v1) == str or type(v1) == int, 'Unsupported vertex represenation'
            if v2 == None:
                assert dir in ['in', 'out', 'all'], 'Unsupported edges direction'
                if type(v1) == str:
                    w1 = int(self.label2vertex[v1])
                else:
                    w1 = v1 #w1 = self.g.vertex(v1)
                if dir == 'out':
                    e = [ed for n in self.get_neighbors(w1) for ed in self.get_edges(w1,int(n))]
                    return e
                    #return self.g.get_out_edges(w1, eprops=[self.edge2weight]) # the label cannot be obtained for some reason... eprops=[self.edge2label] cause an error
                if dir == 'in':
                    e = [ed for n in self.get_neighbors(w1, dir='in') for ed in self.get_edges(int(n),w1)]
                    return e
                    #return self.g.get_in_edges(w1, eprops=[self.edge2weight])
                if dir == 'all':
                    e = []
                    [e.append(i) for i in self.get_edges(w1,dir='in')]
                    [e.append(i) for i in self.get_edges(w1,dir='out')]
                    return e
                    #return self.g.get_all_edges(w1, eprops=[self.edge2weight])
            else:                                                 # warning here dir is not used, it returns always only edges from v1 to v2
                assert type(v1) == type(v2), 'Different vertex representations passed'
                if type(v1) == str:
                    w1 = self.label2vertex[v1]
                    w2 = self.label2vertex[v2]
                else:
                    w1 = v1
                    w2 = v2
                e = [ [self.g.vertex_index[i.source()], self.g.vertex_index[i.target()], self.edge2label[i], self.edge2weight[i]] for i in self.g.edge(w1, w2, all_edges=True)]
                return e
        
    def get_neighbors(self, v, dir='out'):
        assert dir in ['in', 'out', 'all'], 'Unsupported direction'
        neighbors = []
        if dir == 'out':
            [ neighbors.append(int(n)) if n not in neighbors else None for n in self.g.get_out_neighbors(v) ]
            return neighbors
        if dir == 'in':
            [ neighbors.append(int(n)) if n not in neighbors else None for n in self.g.get_in_neighbors(v) ]
            return neighbors
        if dir == 'all':
            [ neighbors.append(int(n)) if n not in neighbors else None for n in self.g.get_all_neighbors(v) ]
            return neighbors

    def remove_vertices(self, vl):
        tmp = [ self.vertex2label[v] for v in vl ]
        self.g.remove_vertex(vl)
        # fixing the dictionary
        for i in tmp:
            del self.label2vertex[i]
        for v in self.get_vertices():
            self.label2vertex[self.vertex2label[v]] = v
            
    def remove_edges(self, el):
        for e in el:
            tmp = self.g.edge(e[0], e[1], all_edges=True)
            for i in tmp:
                if self.edge2label[i] == e[2]:
                    self.g.remove_edge(i)
                    break
        
    def draw(self):
        vprops = {
            'text': self.vertex2label,
            'font_size':16,
            'size':1,
            'text_color':'black'
        }
        eprops = {
            'text': self.edge2label,
            'pen_width':2,
            'end_marker':'arrow',
            'marker_size':12,
        }
        graph_draw(self.g, vprops=vprops, eprops=eprops, output_size=(2000, 2000))
        #graphviz_draw(self.g, layout='sfdp', vprops=vprops, eprops=eprops, size=(25,25))
        '''
        dot = Digraph(comment='Test')
        for v in self.g.vertices():
            dot.node(self.vertex2label[v])
            for n in v.out_neighbors():
                dot.edge(self.vertex2label[v],self.vertex2label[n])

        dot.render(view=True)
        '''



        
class Networkx(Graph):

    def init(self, vertices):
        self.g = nx.MultiDiGraph()
        self.label2vertex = {}
        self.vertex2label = {}
        self.edge2label = {}
        if vertices != None:
            [self.add_vertex(v) for v in vertices]

    def add_vertex(self, vertex):
        try:
            v_i = self.label2vertex[vertex.me]
        except:
            v_i = self.g.number_of_nodes() 
            self.g.add_node(v_i, label=vertex.me)
            self.label2vertex[vertex.me] = v_i
            #self.vertex2label[v_i] = vertex.me

        v_f = []
        index = self.g.number_of_nodes()
        for n in vertex.neighbors:
            try:
                v_f.append(self.label2vertex[n])
            except:
                v_f.append(index)
                self.g.add_node(index, label=n)
                self.label2vertex[n] = index
                #self.vertex2label[index] = n
                index += 1

        for j in range(len(vertex.edges)):
            self.add_edge(vertex.edges[j], v_i, v_f[j], dir='bi') if vertex.edges[j] in self.bidir_preds else self.add_edge(vertex.edges[j], v_i, v_f[j])

    def add_edge(self, edge, v1, v2, dir='straight'):
        assert dir in {'straight', 'inverted', 'bi'}, 'Unsupported edge direction'
        if dir == 'inverted':
            return self.add_edge(edge, v2, v1)
        try:
            w = self.g[v1][v2][edge]['weight'] + 1
        except:
            w = 1
        self.g.add_edge(v1, v2, key=edge, label=edge, weight=w)
        if dir == 'bi':
            return self.add_edge(edge, v2, v1)

    def get_vertex(self, v):
        if type(v) == str:
            return self.label2vertex[v]
        if type(v) == int:
            return self.g.nodes[v]['label']
            #return self.vertex2label[v]

    def get_vertices(self):
        return [v for v in self.g]
        
    def get_edges(self, v1=None, v2=None, dir='out'):
        if v1 == None:
            e = [ [i[0], i[1], i[2]['label'], i[2]['weight']] for i in list(self.g.edges(data=True)) ]
            return e
        else:
            if v2 == None:
                assert type(v1) == str or type(v1) == int, 'Unsupported vertex representation'
                assert dir in ['in', 'out', 'all'], 'Unsupported edges direction'
                if type(v1) == str:
                    w1 = self.label2vertex[v1]
                else:
                    w1 = v1
                if dir == 'out':
                    e = []
                    for n in self.g.neighbors(w1):
                        [ e.append([w1, n, val['label'], val['weight']]) for val in self.g[w1][n].values()]
                    return e
                if dir == 'in':
                    e = []
                    for n in self.g.predecessors(w1):
                        [ e.append([n, w1, val['label'], val['weight']]) for val in self.g[n][w1].values()]
                    return e
                if dir == 'all':
                    e = []
                    for n in self.g.neighbors(w1):
                        [ e.append([w1, n, val['label'], val['weight']]) for val in self.g[w1][n].values()]
                    for n in self.g.predecessors(w1):
                        [ e.append([n, w1, val['label'], val['weight']]) for val in self.g[n][w1].values()]
                    return e
            else:                                                 
                assert type(v1) == type(v2), 'Different vertex representations passed'
                if type(v1) == str:
                    w1 = self.label2vertex[v1]
                    w2 = self.label2vertex[v2]
                else:
                    w1 = v1
                    w2 = v2
                e = []
                [ e.append([w1, w2, val['label'], val['weight']]) for val in self.g[w1][w2].values() ]
                #[ e.append([w2, w1, lab['label']]) for lab in self.g[w2][w1].values() ] # considering only parallel edges from v1 to v2 and not viceversa
                return e
        
    def get_neighbors(self, v, dir='out'):         
        assert dir in ['in', 'out', 'all'], 'Unsupported edges direction'
        if dir == 'out':
            return [n for n in self.g.neighbors(v)]   
        if dir == 'in':
            return [n for n in self.g.predecessors(v)]
        if dir == 'all':
            neigh = []
            [neigh.append(n) for n in self.g.neighbors(v) ]   
            [neigh.append(n) for n in self.g.predecessors(v) ]
            return neigh
        
    def remove_vertices(self, vl):
        self.g.remove_nodes_from(vl)  
        vertices = []
        mem = {} # I need to keep memory of already added bidirectional preds in order to not add the inverse too
        for e in self.get_edges():
            source = self.get_vertex(e[0])
            target = [self.get_vertex(e[1])]
            if e[2] in self.bidir_preds:
                mem[source+'-'+e[2]+'-'+target[0]] = True
                try:
                    mem[target[0]+'-'+e[2]+'-'+source] # checking the memory
                except:
                    [ vertices.append(Vertex(source, [e[2]], target)) for i in range(e[3]) ] 
            else:
                [ vertices.append(Vertex(source, [e[2]], target)) for i in range(e[3]) ]
        self.__init__(vertices=vertices)

    def remove_edges(self, el):
        for e in el:
            self.g.remove_edge(e[0], e[1], key=e[2])
        
    def draw(self, **kwargs):
        plt.subplot()
        #options = {
         #   'labels' : self.vertex2label, 
          #  'arrows' : True
        #}
        node_labels = {v:self.get_vertex(v) for v in self.get_vertices()}
        edge_labels = {(e[0], e[1]): 'w = '+str(e[3]) for e in self.get_edges()}
        nx.draw_networkx(self.g, labels=node_labels, arrows=True, **kwargs)
        nx.draw_networkx_edge_labels(self.g, edge_labels=edge_labels,**kwargs)
        plt.show()
