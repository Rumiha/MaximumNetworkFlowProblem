# Ford-Fulkerson algorith in Python

from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import copy

def main():

    example1 = [[0, 3, 8, 0, 0, 0],
                [0, 0, 0, 4, 7, 0],
                [0, 0, 0, 0, 9, 0],
                [0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0]]

    not_direct_example1 = [[0, 3, 8, 0, 0, 0],
                        [0, 0, 0, 4, 7, 0],
                        [0, 0, 0, 0, 9, 0],
                        [0, 0, 0, 0, 0, 5],
                        [0, 7, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0, 0]]

    example2 = [[0, 9, 3, 0],
                [0, 0, 5, 8],
                [0, 0, 0, 9],
                [0, 0, 0, 0]] 


    example3 = [[0, 15, 8, 12, 10, 0, 0, 0, 0, 4, 7, 0],
                [0, 0, 0, 4, 7, 6, 0, 0, 0, 4, 7, 0],
                [0, 0, 0, 7, 9, 0, 6, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 8, 5, 0, 6, 0, 4, 7, 0],
                [0, 7, 0, 0, 0, 2, 0, 0, 6, 4, 7, 0],
                [0, 0, 0, 4, 7, 0, 5, 0, 0, 4, 7, 0],
                [0, 0, 0, 0, 9, 0, 0, 8, 0, 4, 7, 0],
                [0, 0, 0, 0, 0, 5, 0, 0, 9, 4, 7, 0],
                [0, 7, 0, 0, 0, 2, 0, 0, 0, 4, 7, 8],
                [0, 0, 0, 0, 0, 5, 0, 0, 0, 4, 7, 7],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 9],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    graph = example1;            
    g = Graph(graph)

    source = 0
    sink = len(graph)-1
    
    print("Max Flow: %d " % g.ford_fulkerson(source, sink))
    g.print_graph(False)
    


class Graph:

    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)
        self.old_graph = copy.deepcopy(graph)
        self.edge_values = [[0 for j in range(len(graph))] for i in range(len(graph))]
        self.node_in_use = [[False for j in range(len(graph))] for i in range(len(graph))]


    def print_graph(self, new):
        if new == False:
            G = nx.DiGraph()

            for i in range(len(self.old_graph)):
                for j in range(len(self.old_graph[i])):
                    if self.old_graph[i][j] != 0:
                        if(i == 0):
                            G.add_edge('S', chr(64+j), weight = self.old_graph[i][j])
                        elif(i == len(self.old_graph)-1):
                            G.add_edge(chr(64+i), chr(64+j), weight = self.old_graph[i][j])
                        else:
                            G.add_edge(chr(64+i), chr(64+j), weight = self.old_graph[i][j])

            for i in range(len(self.old_graph)):
                if(i==len(self.old_graph)-1):
                    G.add_node(chr(64+i), pos = ((i/2 + 1), 3))
                empty = 1
                for j in range(len(self.old_graph[i])):
                    if self.old_graph[i][j] != 0:
                        empty=0
                if empty==0:
                    if(i==0):
                        G.add_node('S', pos = (0 , 3))
                    elif(i%2==1):
                        G.add_node(chr(64 + i), pos = ((i+1)/2, 0))
                    else:
                        G.add_node(chr(64 + i), pos = (i/2, 5))
                
            G.nodes(data=True)
            G.edges(data=True)
            
            pos = nx.get_node_attributes(G,'pos')
            weight = nx.get_edge_attributes(G,'weight')

            nx.draw(G, pos, with_labels=True, arrows=True, arrowsize=15, arrowstyle='-|>', node_size = 400, font_weight = 'bold') 
            nx.draw_networkx_edge_labels(G, pos, edge_labels = weight)
            plt.show()

        if new == True:
            G = nx.DiGraph()
            for i in range(len(self.edge_values)):
                for j in range(len(self.edge_values[i])):
                    if self.edge_values[i][j] != 0:
                        if self.node_in_use[i][j]==True:
                            if(i == 0):
                                G.add_edge('S', chr(64+j), weight = self.edge_values[i][j])
                            elif(i == len(self.edge_values)-1):
                                G.add_edge(chr(64+i), chr(64+j), weight = self.edge_values[i][j])
                            else:
                                G.add_edge(chr(64+i), chr(64+j), weight = self.edge_values[i][j])
                        else:
                            if(i == 0):
                                G.add_edge('S', chr(64+j), weight = 0)
                            elif(i == len(self.edge_values)-1):
                                G.add_edge(chr(64+i), chr(64+j), weight = 0)
                            else:
                                G.add_edge(chr(64+i), chr(64+j), weight = 0)
                       

            for i in range(len(self.edge_values)):
                if(i==len(self.edge_values)-1):
                    G.add_node(chr(64+i), pos = ((i/2 + 1), 3))
                empty = 1
                for j in range(len(self.edge_values[i])):
                    if self.edge_values[i][j] != 0:
                        empty=0
                if empty==0:
                    if(i==0):
                        G.add_node('S', pos = (0 , 3))
                    elif(i%2==1):
                        G.add_node(chr(64 + i), pos = ((i+1)/2, 0))
                    else:
                        G.add_node(chr(64 + i), pos = (i/2, 5))
                
            G.nodes(data=True)
            G.edges(data=True)
            
            pos = nx.get_node_attributes(G,'pos')
            weight = nx.get_edge_attributes(G,'weight')

            nx.draw(G, pos, with_labels=True, arrows=True, arrowsize=15, arrowstyle='-|>', node_size = 400, font_weight = 'bold') 
            nx.draw_networkx_edge_labels(G, pos, edge_labels = weight)
            plt.show()


    def searching_algo_BFS(self, s, t, parent):

        visited = [False] * (self.ROW)
        queue = []

        queue.append(s)                      
        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    def ford_fulkerson(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0

        while self.searching_algo_BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            max_flow += path_flow

            v = sink
            while(v != source):
                u = parent[v]

                if self.edge_values[u][v] > path_flow:
                    self.edge_values[u][v] += path_flow
                else:
                    self.edge_values[u][v] = path_flow    

                self.node_in_use[u][v] = True
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


if __name__ == "__main__":
    main()