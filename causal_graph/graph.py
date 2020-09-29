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
from scipy.special import comb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

    @abstractmethod
    def draw_cluster(self, clusters):
        pass

    @abstractmethod
    def draw_embedding(self):
        pass

    @abstractmethod
    def find_path(self, s, t):
        pass

    @abstractmethod
    def louvain_communities(self):
        pass

    def get_degree(self, v, dir='all'):
        assert dir in ['in', 'out', 'all'], 'Unsupported direction'
        deg = 0
        for e in self.get_edges(v, dir=dir):
            deg += e[3]
        return deg
    
    def to_unoriented(self):
        for e in self.get_edges():
            if e[2] not in self.bidir_preds:
                self.add_edge(e[2],e[1],e[0])

    def get_clusters(self):
        clusters = {}
        for k, v in self.vertex2cluster.items():
            try:
                clusters[v]['nodes'].append(k)
            except:
                clusters[v] = {'nodes': [k], 'edges': {}}
        for k in clusters.keys():
            for c in clusters.keys():
                clusters[k]['edges'][c] = {}
        for e in self.get_edges():
            c1 = self.vertex2cluster[self.get_vertex(e[0])]
            c2 = self.vertex2cluster[self.get_vertex(e[1])]
            try:
                clusters[c1]['edges'][c2][e[2]][0] += e[3]
            except:
                clusters[c1]['edges'][c2][e[2]] = [e[3],0]
        for c in clusters.keys():
            tmp = 0
            for k in clusters[c]['edges'].keys():
                for n in clusters[c]['edges'][k].values():
                    tmp += n[0]
            for k in clusters[c]['edges'].keys():
                for e in clusters[c]['edges'][k].keys(): 
                    clusters[c]['edges'][k][e][1] = clusters[c]['edges'][k][e][0] / tmp * 100
                    
        return clusters
                    
    def clean(self):
        vl = []
        for v in self.get_vertices():
            if self.get_degree(v) < 1:
                vl.append(v)
        self.remove_vertices(vl)

    def remove_disconnected_components(self, th=10):
        g = self.g.to_undirected()
        comp = sorted(nx.connected_components(g), key = len, reverse=True)
        del_list = []
        for i in comp:
            if len(i) < th:
                for j in i:
                    del_list.append(j)
        self.remove_vertices(del_list)
        
    def degree(self, d=10):
        del_list = []
        for v in self.get_vertices():
            deg = 0
            for n in self.get_neighbors(v):
                deg += self.get_degree(n)
            if deg < d :
                del_list.append(v)
        self.remove_vertices(del_list)
        self.clean()

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
 
    def skipgrams(self, sent, window_size, vocab, negative_samples=1):
        couples = []
        labels = []
        #print(sent)
        for i in range(len(sent)-1):
            vocab_bak = vocab[:]
            vocab_bak.remove(sent[i]) # is this necessary? do couples like (7,7) count as negative samples?
            window = []
            for j in range(i+1, min(i+window_size, len(sent))):
                window.append(sent[j])
                try: 
                    vocab_bak.remove(window[-1])
                except:
                    pass
            #window = [sent[j] for j in range(i+1, min(i+window_size, len(sent)))]
            #print(window)
            for v in window:
                couples.append([sent[i],v])
                labels.append(1)
                for k in range(negative_samples):
                    #r = random.randint(0, vocab_size-1)
                    try: 
                        r = random.choice(vocab_bak)
                        couples.append([sent[i],r])
                        labels.append(0)
                    except:
                        print('W: Impossible to generate negative samples, the window equals the vocabulary!')
        tmp = list(zip(couples,labels))
        random.shuffle(tmp)
        return list(zip(*tmp))#(couples, labels)

    def deep_walk(self, walk_length=20, window=10, walks_per_node=10, embedding_dim=100, with_edges=False):
        # preparing vocabulary and corpus
        corpus = []
        vocab = self.get_vertices()
        edges_vocab = {}
        if with_edges :
            window = 2*window
        #edges_vocab = {e : int(len(vocab)+i) for i,e in enumerate(self.causal_preds+self.bidir_preds)} # using also edges (only causal ones) with their relative label in the embedding
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
        #table = make_sampling_table(vocab_size)
        couples = []
        labels = []
        c = 0
        for sent in corpus:
            
            try:
                tmp1, tmp2 = self.skipgrams(sent, window, vocab, negative_samples=1)
                couples += tmp1
                labels += tmp2
            except:
                pass
            
            #try:
             #   tmp1, tmp2 = skipgrams(sent, vocab_size, window_size=window)
            #except:
             #   pass
            
            #couples += tmp1
            #labels += tmp2
            c += 1
            print('Generating skipgrams: ', int(c/len(corpus)*100), '%\r', end='')
        print('Generating skipgrams: ', 100, '%\r', end='\n')
            
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
        model.fit(x=[target,context], y=np.asarray(labels), batch_size=128, epochs=1, validation_split=0.2, workers=12, use_multiprocessing=True)

        weights = embedding.get_weights()
        if with_edges == True:
            edges_vocab = {v: k for k, v in edges_vocab.items()}
        for n in vocab:
            if n < vocab_size - len(edges_vocab.keys()):
                self.embedding[self.get_vertex(n)] = weights[0][n]
            else:
                self.embedding[edges_vocab[n]] = weights[0][n]
        
        return self.embedding

    def k_means(self, n_clusters=None, elbow_range=(2,11)):
        x = np.array([v for v in self.embedding.values()])
        sse = {}
        kmeans = {}
        if n_clusters != None:
            kmeans = KMeans(n_clusters=n_clusters).fit(x)
        else:
            for i in range(elbow_range[0], elbow_range[1] + 1):
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

        i = 0
        for k in self.embedding.keys():
            self.vertex2cluster[k] = int(kmeans.labels_[i])
            i += 1
        return self.vertex2cluster

    def agglomerative_clustering(self, n_cluster):
        clusters = AgglomerativeClustering(n_cluster).fit(list(self.embedding.values())).labels_
        for i,k in enumerate(self.embedding.keys()):
            self.vertex2cluster[k] = clusters[i]
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

    def draw_cluster(self, clusters):
        pass

    def draw_embedding(self):
        pass


        
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

    def louvain_communities(self):
        g = self.g.to_undirected()
        partition = community_louvain.best_partition(g)
        for k,v in partition.items():
            self.vertex2cluster[self.get_vertex(k)] = v
        return self.vertex2cluster
        
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
        
    def draw(self, pos=None, edge_label=None, highlight=None, **kwargs):
        plt.subplot()
        #options = {
         #   'labels' : self.vertex2label, 
          #  'arrows' : True
        #}
        node_labels = {v:self.get_vertex(v) for v in self.get_vertices()}

        if pos == None:
            pos = nx.drawing.layout.spring_layout(self.g)
        node_color = [ 'lightblue' for i in self.get_vertices() ]
        if highlight != None:
            for v in self.get_vertices():
                if v in highlight:
                    node_color[v]='red'
                else:
                    node_color[v]='lightblue'
  
        nx.draw_networkx(self.g, pos=pos, labels=node_labels, node_color=node_color, arrows=True, edge_color='grey', connectionstyle='arc3, rad = 0.0', **kwargs)
        if edge_label != None :
            if edge_label == 'type':
                edge_labels = {(e[0], e[1]): e[2] for e in self.get_edges()}
            elif edge_label == 'weight':
                edge_labels = {(e[0], e[1]): 'w='+str(e[3]) for e in self.get_edges()}
            nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=edge_labels, **kwargs)
        plt.show()

    def draw_cluster(self, clusters):
        nodes = {}
        labels = {}
        pos = nx.spring_layout(self.g)
        for k,v in self.vertex2cluster.items():
            if v in clusters:
                try:                                           # avoid fictitious edge-nodes
                    nodes[self.get_vertex(k)] = v
                    labels[self.get_vertex(k)] = k
                except:
                    pass
        nx.draw_networkx_nodes(self.g, pos=pos, nodelist=list(nodes.keys()), node_color=list(nodes.values()), cmap=plt.cm.get_cmap('tab10'))
        nx.draw_networkx_labels(self.g, pos=pos, labels=labels, font_size=8)
        edges = []
        e_lab = {}
        for e in self.get_edges():
            try:
                tmp = nodes[e[0]]
                tmp = nodes[e[1]] 
                edges.append((e[0],e[1]))
                e_lab[(e[0],e[1])] = e[2]
            except:
                pass
        nx.draw_networkx_edges(self.g, pos, edgelist=edges, edge_color='grey', arrows=True, connectionstyle='arc3, rad = 0.03')
        nx.draw_networkx_edge_labels(self.g, pos=pos, edge_labels=e_lab, font_size=8)
        plt.show()

    def find_path(self, s, t, cut=20):
        if type(s) == str:
            self.get_vertex(s)
        if type(t) == str:
            self.get_vertex(t)
        try:
            paths = list(nx.all_simple_paths(self.g, s, t, cutoff=cut))
            return [[self.get_vertex(v) for v in p] for p in paths]
        except:
            return False
        
    def draw_embedding(self, annotations=None):
        evecs = np.array([v for v in self.embedding.values()])
        
        #d = 50 if len(evecs) > 50 else 2
        if len(evecs) > 2:
            pca = PCA(n_components=2)
            pca.fit(evecs)
            evecs = pca.transform(evecs)
        #if d == 50:
            #evecs = TSNE(n_components=2).fit_transform(evecs)
        
        twod_emb = {}
        for k,i in zip(self.embedding.keys(),evecs):
            twod_emb[k] = i

        clusters = self.get_clusters()
        cmap = plt.cm.get_cmap('tab10')
        c = [ cmap(k) for k in range(len(list(clusters.keys()))) ]
        
        for k, v in clusters.items():
            x = np.zeros(len(v['nodes']))
            y = np.zeros(len(v['nodes']))
            i = 0
            for n in v['nodes']:
                x[i] = twod_emb[n][0]
                y[i] = twod_emb[n][1]
                i += 1
            plt.scatter(x=x, y=y, c=[ c[k] for j in range(x.shape[0]) ], label=k)
        plt.legend()
        
        if annotations == None:
            for k,v in twod_emb.items():
                plt.annotate(k, v, fontsize=8)
        else:
            for a in annotations:
                plt.annotate(a, twod_emb[a], fontsize=8)

        

class MultilayerGraph(object):
    def __init__(self, layers):
        assert type(layers) == int or type(layers) == list, '\'layers\' must be a list or an int'
        if type(layers) == int:
            self.G = {str(i) : nx.DiGraph() for i in range(layers)}
        elif type(layers) == list:
            self.G = {str(i) : nx.DiGraph() for i in layers}
            
        self.l = 0.5 #jump probability between layers
        self.embedding = {}
        self.vertex2cluster = {}

    def add_edge(self, layer, u, v, w=1, dir='straight'):
        try:
            tmp = self.G[layer][u]
        except:
            self.add_vertex(u)
        try:
            tmp = self.G[layer][v]
        except:
            self.add_vertex(v)
        self.G[layer].add_edge(u, v, w=w)
        if dir == 'bi':
            return self.add_edge(layer, v, u, w=w)

    def add_vertex(self, v, label=None): #the vertex is added to each layer
        for g in self.G.values():
            g.add_node(v, label=label)
            
    def add_layer(self, label):
        self.G[label] = nx.DiGraph()
        self.G[label].add_nodes_from(list(self.G.values())[0])

    def edges(self):
        return {l : list(g.edges()) for l, g in self.G.items()}

    def vertices(self):
        return list(self.G[list(self.G.keys())[0]].nodes())
        
    def neighbors(self, v):
        return {l : list(g.neighbors(v)) for l, g in self.G.items()}

    def adjacency_matrix(self, layer=None):
        if layer != None:
            return nx.convert_matrix.to_numpy_matrix(self.G[layer])
        else:
            A = [ nx.convert_matrix.to_numpy_matrix(g) for g in self.G.values()]
            return np.sum(A, axis=0)

    def merged_graph(self):
        g = nx.MultiDiGraph()
        for i in self.G.values():
            g.add_edges_from(i.edges)
        return g
    
    def draw(self, merged=False, **kwargs):
        try:
            ncols = kwargs['ncols']
        except:
            ncols = len(self.G)
        try:
            nrows = kwargs['nrows']
        except:
            nrows = 1
        fig = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,10))
        
        try:
            pos = kwargs.pop('pos')
        except:
            pos = nx.spring_layout(self.merged_graph())
            
        if self.vertex2cluster != {}:
            clusters = self.vertex2cluster.values()
        else:
            clusters = [0 for i in range(len(self.vertices()))]

        axi = 0
        for l, g in self.G.items():
            nx.draw_networkx(g, ax=fig[1][axi], pos=pos, node_color=[c for c in clusters], with_labels=True, arrows=True, cmap=plt.cm.get_cmap('cool'), edge_color='grey', **kwargs)
            fig[1][axi].set_title(l)
            axi += 1
        plt.show()   

    def random_walk(self, v=None, layer=None, length=64):
        if v == None:
            v = random.choice(self.vertices())
        if layer == None:
            tmp = []
            for l, n in self.neighbors(v).items():
                if len(n) != 0:
                    tmp.append(l)
            layer = random.choice(tmp)    

        walk = [v]
        layer_walk = [layer]
        for i in range(length):
            if random.randrange(0,1,1) < self.l:
                layer = random.choice(list(set(self.G.keys()) - set(layer)))
            layer_walk.append(layer)
            neighbors = list(self.G[layer].neighbors(walk[-1]))
            if len(neighbors) != 0:
                random.shuffle(neighbors)
                walk.append(random.choice(neighbors))
            else:
                break
        return walk

    def skipgrams(self, sent, window_size, vocab, negative_samples=1):
        couples = []
        labels = []
        #print(sent)
        for i in range(len(sent)-1):
            vocab_bak = vocab[:]
            vocab_bak.remove(sent[i]) # is this necessary? do couples like (7,7) count as negative samples?
            window = []
            for j in range(i+1, min(i+window_size, len(sent))):
                window.append(sent[j])
                try: 
                    vocab_bak.remove(window[-1])
                except:
                    pass
            #window = [sent[j] for j in range(i+1, min(i+window_size, len(sent)))]
            #print(window)
            for v in window:
                couples.append([sent[i],v])
                labels.append(1)
                for k in range(negative_samples):
                    #r = random.randint(0, vocab_size-1)
                    try: 
                        r = random.choice(vocab_bak)
                        couples.append([sent[i],r])
                        labels.append(0)
                    except:
                        print('W: Impossible to generate negative samples, the window equals the vocabulary!')
                    '''
                    counter = 0
                    while r in window:
                        r = random.randint(0, vocab_size-1)
                        counter +=1
                        if counter > vocab_size:
                            r = False
                            break
                    if r:
                        couples.append([i,r])
                        labels.append(0)
                    '''
        tmp = list(zip(couples,labels))
        #print(tmp)
        random.shuffle(tmp)
        return list(zip(*tmp))#(couples, labels)

    def deep_walk(self, walk_length=64, window=10, walks_per_node=32, embedding_dim=128, layer=None):
        corpus = []
        vocab = self.vertices()
        self.vocab2index = { vocab[i] : i for i in range(len(vocab)) }
        
        for n in range(walks_per_node):
            print('Generating Walks:', int(n/walks_per_node*100), '%\r', end='')
            random.shuffle(vocab)
            starts = [random.choice(vocab) for j in range(len(vocab))]
            p = Pool(processes=12)
            sents = p.starmap(self.random_walk, zip(starts, repeat(layer), repeat(walk_length)))
            p.close()
            p.join()
            for s in sents:
                tmp = [ self.vocab2index[w] for w in s ]
                corpus.append(tmp)
        #print(self.vocab2index)
        print('Generating Walks:', 100, '% ')

        vocab_size = len(vocab)
        couples = []
        labels = []
        for sent in corpus:
            #print('# SENT:\n', sent)
            #t = self.skipgrams(sent, window, vocab_size, negative_samples=1)  #PROBLEM WITH SKIPGRAMS: (7,7) --> label 1
            #print(t[0])
            #print(t[1])
            if len(sent) > 1:
                tmp1, tmp2 = self.skipgrams(sent, window, list(self.vocab2index.values()), negative_samples=1)
                couples += tmp1
                labels += tmp2
        print(self.vocab2index)
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
       
        for k, v in self.vocab2index.items():
            self.embedding[k] = weights[0][v]

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

        for i in self.vertices():
            self.vertex2cluster[i] = int(kmeans.labels_[self.vocab2index[i]])
        return self.vertex2cluster

    
            


        

class Modularity(object):
    def __init__(self, A, M, clusters):
        self.RM = A
        self.M = M
        self.ERC = A - 1 #np.zeros((A.shape[0],A.shape[1]))
        self.clusters = clusters
        #for i in range(A.shape[0]):
            #for j in range(A.shape[1]):
                #self.ERC[i][j] = max(A[i][j] -1, 0)

    def mu(self, m):
        counter = 0
        for i in range(self.ERC.shape[0]):
            for j in range(self.ERC.shape[1]):
                if self.ERC[i][j] == m :
                    counter += 1
        return counter

    def r(self, i, m):
        counter = 0
        for j in self.RM[i]:
            counter += 1 if j == m+1 else 0
        return counter

    def delta(self, i, j):
        if self.clusters[i] == self.clusters[j]:
            return 1
        else:
            return 0
    
    def Q(self):
        mu = np.sum(self.RM)
        Q = 0
        for i in range(self.RM.shape[0]):
            for j in range(self.RM.shape[1]):
                if self.clusters[i] == self.clusters[j]:
                    C = 2*mu*self.M * np.sum([ ( comb(m+1,1)/comb(self.M,m+1) ) * (m+1)**2 * self.r(i,m)*self.r(j,m) / (2*self.mu(m))**2 for m in range(self.M)])
                    Q += self.RM[i][j] - C
        return Q / (2*mu)


def similarity(g, clusters):
    verts = g.get_vertices()
    sim = 0
    for i in verts:
        for j in verts:
            if clusters[i] == clusters[j] and i != j:
                ni = set(g.get_neighbors(i))
                nj = set(g.get_neighbors(j))
                sim += len(ni.intersection(nj)) / len(ni.union(nj))
    return sim / len(verts)
    

