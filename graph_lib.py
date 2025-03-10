import random
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, n):
        self.n = n
        self.edges = set()

    def add_edge(self, u, v):
        if u == v:
            return
        edge = (min(u, v), max(u, v))
        self.edges.add(edge)

    @staticmethod
    def from_adjacency_matrix(mat):
        n = len(mat)
        g = Graph(n)
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i][j] == 1:
                    g.add_edge(i + 1, j + 1)
        return g

    @staticmethod
    def from_incidence_matrix(inc_mat):
        n = len(inc_mat)
        if n == 0:
            return Graph(0)
        m = len(inc_mat[0])
        g = Graph(n)
        for col in range(m):
            ones = [i for i in range(n) if inc_mat[i][col] == 1]
            if len(ones) == 2:
                g.add_edge(ones[0] + 1, ones[1] + 1)
            else:
                raise ValueError("Kolumna macierzy incydencji nie zawiera dokladnie dwoch jedynek.")
        return g

    @staticmethod
    def from_adjacency_list(adj_list):
        n = max(adj_list.keys())
        g = Graph(n)
        for u, neighbors in adj_list.items():
            for v in neighbors:
                if u < v:
                    g.add_edge(u, v)
        return g

    def to_adjacency_matrix(self):
        mat = [[0] * self.n for _ in range(self.n)]
        for (u, v) in self.edges:
            mat[u - 1][v - 1] = 1
            mat[v - 1][u - 1] = 1
        return mat

    def to_adjacency_list(self):
        adj_list = {i: [] for i in range(1, self.n + 1)}
        for (u, v) in self.edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        for key in adj_list:
            adj_list[key].sort()
        return adj_list

    def to_incidence_matrix(self):
        m = len(self.edges)
        mat = [[0] * m for _ in range(self.n)]
        for col, (u, v) in enumerate(self.edges):
            mat[u - 1][col] = 1
            mat[v - 1][col] = 1
        return mat
    
    def visualize(self):
        G = nx.Graph()
        G.add_nodes_from(range(1, self.n + 1))
        G.add_edges_from(self.edges)
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue")
        plt.title("Graph")
        plt.show()

    @staticmethod
    def random_graph_G_n_l(n, l):
        g = Graph(n)
        possible_edges = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
        if l > len(possible_edges):
            raise ValueError("Liczba krawędzi przekracza maksymalna dla grafu prostego.")
        chosen = random.sample(possible_edges, l)
        for (u, v) in chosen:
            g.add_edge(u, v)
        return g

    @staticmethod
    def random_graph_G_n_p(n, p):
        g = Graph(n)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if random.random() < p:
                    g.add_edge(i, j)
        return g

def load_adjacency_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    matrix = [[int(x) for x in line.split()] for line in lines]
    return Graph.from_adjacency_matrix(matrix)

def load_incidence_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    matrix = [[int(x) for x in line.split()] for line in lines]
    return Graph.from_incidence_matrix(matrix)

def load_adjacency_list_from_file(filename):
    adj_list = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                vertex = int(tokens[0].rstrip('.'))
            except ValueError:
                continue
            neighbors = []
            for token in tokens[1:]:
                try:
                    neighbors.append(int(token.rstrip('.')))
                except ValueError:
                    continue
            adj_list[vertex] = neighbors
    return Graph.from_adjacency_list(adj_list)

def save_graph_representations(graph, out_filename):
    with open(out_filename, 'w') as f:
        f.write("=== Macierz sasiedztwa ===\n")
        for row in graph.to_adjacency_matrix():
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")

        f.write("=== Lista sasiedztwa ===\n")
        for v, neighbors in sorted(graph.to_adjacency_list().items()):
            f.write(f"{v}: " + " ".join(map(str, neighbors)) + "\n")
        f.write("\n")

        f.write("=== Macierz incydencji ===\n")
        for row in graph.to_incidence_matrix():
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")