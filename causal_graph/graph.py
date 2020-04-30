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
import re, random
import numpy as np
from tensorflow.keras.models import Model
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.layers import Embedding, Input, Reshape, Dot, Dense
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Graph(ABC):

    def __init__(self, vertices=None):
        self.g = None
        self.vertex2label = None
        self.edge2label = None
        self.label2vertex = None
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
    
    def redundancy(self, k=2):
        redundancy_map = self.g.new_edge_property("bool") # have to use filtering cause removal was causing core dumped
        for v in self.g.vertices():
            dict = {}
            for e in v.out_edges():
                t = e.target()
                try:
                    dict[self.vertex2label[t]].append(e)
                except:
                    dict[self.vertex2label[t]] = [e]
            for i in dict.values():
                if len(i) < k:
                    for j in i:
                        redundancy_map[j] = False
                else:
                    for j in i:
                        redundancy_map[j] = True
                        
        self.g.set_edge_filter(redundancy_map)
        self.clean()

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
        del_list = []
        for e in self.get_edges():
            if e[2] in self.causal_preds or e[2] in self.bidir_preds:
                pass
            else:
                del_list.append(e)
        self.remove_edges(del_list)
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
            v = random.choice(self.get_vertices())
        #start = v
        walk = [v]
        for i in range(length):
            neighbors = []
            for n in self.get_neighbors(walk[-1]):
                for e in self.get_edges(walk[-1], n):
                    for i in range(e[3]):
                        neighbors.append(n)
            if len(neighbors) != 0:
                random.shuffle(neighbors)
                walk.append(random.choice(neighbors))
            else:
                #walk.append(start)
                break
        return walk

    def deep_walk(self, walk_length=20, window=5, embedding_dim=100):
        corpus = []
        vocab = self.get_vertices()
        for i in range(30):
            #random.shuffle(vocab)
            for j in range(len(vocab)):
                corpus.append(self.random_walk(v=random.choice(vocab)))

        # skipgram with negative sampling
        vocab_size = len(vocab)
        table = make_sampling_table(vocab_size)
        couples = []
        labels = []
        for sent in corpus:
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
        #print(weights[0].shape)
        self.embedding = {}
        for n in vocab:
            self.embedding[n] = weights[0][n]

        return self.embedding
        
    def merge_vertices(self, v1, v2):         # not working, why??????
        del_list = []
        for e in v1.out_edges():
            self.add_edge(self.edge2label[e], v2, e.target())
            del_list.append(e)
        for e in v1.in_edges():
            self.add_edge(self.edge2label[e], e.source(), v2)
            del_list.append(e)
        for e in reversed(sorted(del_list)):
            self.g.remove_edge(e)
        self.g.remove_vertex(v1)

    def json(self, file='graph.json'):
        nodes = []
        links = []
        for v in self.get_vertices():
            nodes.append({"id": self.get_vertex(v), "cluster": 1, "category": 'cat', "degree": self.get_degree(v)})
        for e in self.get_edges():
            links.append({"source": self.get_vertex(e[0]), "target": self.get_vertex(e[1]), "label": e[2], "weight": e[3]})
        with open(file, 'w') as f:
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
        
    def draw(self):
        plt.subplot()
        #options = {
         #   'labels' : self.vertex2label, 
          #  'arrows' : True
        #}
        nx.draw_networkx(self.g, labels=self.vertex2label, arrows=True)
        plt.show()
