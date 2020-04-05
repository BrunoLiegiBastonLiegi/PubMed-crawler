
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
from graph_tool.all import graph_draw, graphviz_draw
import re
from graphviz import Digraph

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
                
        e = [self.add_edge(vertex.edges[j], v_i, v_f[j]) for j in range(len(vertex.edges))]
        
        
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
    
    def redundancy_filter(self, k=2):
        redundancy_map = self.g.new_edge_property("bool") # have to use filtering cause removal was causing core dumped
        for v in self.g.vertices():
            for n in v.out_neighbors():
                edges = self.g.edge(v, n, all_edges=True)
                if len(edges) < k:
                    for e in edges:
                        redundancy_map[e] = False                
                else:
                    for e in edges:
                        redundancy_map[e] = True
                
        self.g.set_edge_filter(redundancy_map)
        self.clean()

    def word_embedding_filter(self, model, target):
        embedding_map = self.g.new_vertex_property("bool")
        WORD = re.compile(r'\w+')
        for v in self.g.vertices():
            ent = WORD.findall(self.verts_text[v])
            if model.wv.n_similarity(ent, target) > 0.4:
                embedding_map[v] = True
            else:
                embedding_map[v] = False

    def causal(self):
        causal_map = self.g.new_edge_property("bool")
        for e in self.g.edges():
            if self.edges_text[e] in self.causal_preds or self.edges_text[e] in self.bidir_preds:
                causal_map[e] = True
            else:
                causal_map[e] = False
        self.g.set_edge_filter(causal_map)
        self.clean()

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
