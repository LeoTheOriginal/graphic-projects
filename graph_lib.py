import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict


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
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue")
        plt.axis('equal')
        plt.title("Graph")
        plt.show()

    def visualize_components(self):
        """
        Wizualizuje graf, przy czym wierzchołki należące do różnych spójnych składowych
        mają różne kolory (wykorzystujemy colormap 'tab10').
        """
        comp = self.get_connected_components()
        comp_ids = [comp[i] for i in range(1, self.n + 1)]
        unique_comps = sorted(set(comp_ids))

        import matplotlib.pyplot as plt
        import networkx as nx
        cmap = plt.get_cmap("tab10")
        color_map = {cid: cmap(i % 10) for i, cid in enumerate(unique_comps)}

        node_colors = [color_map[comp[i]] for i in range(1, self.n + 1)]

        G = nx.Graph()
        G.add_nodes_from(range(1, self.n + 1))
        G.add_edges_from(self.edges)
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500)
        plt.title("Graf z oznaczonymi komponentami spójnymi")
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

    @staticmethod
    def is_graphic_sequence(seq):
        """
        Zwraca True/False w zależności od tego, czy ciąg seq jest graficzny.
        Implementacja wg Havel-Hakimi (z pominięciem tworzenia krawędzi).
        """
        if sum(seq) % 2 != 0:
            return False

        degrees = sorted(seq, reverse=True)

        while True:
            degrees.sort(reverse=True)
            if degrees[0] == 0:
                return True

            d = degrees[0]
            degrees[0] = 0

            if d > len(degrees) - 1:
                return False

            for i in range(1, d + 1):
                degrees[i] -= 1
                if degrees[i] < 0:
                    return False

    @staticmethod
    def graph_from_degree_sequence(seq):
        """
        Buduje i zwraca obiekt Graph zadanego ciągu stopni, jeśli jest on graficzny.
        Jeśli nie jest graficzny, rzuca ValueError.
        Implementacja w oparciu o Havel-Hakimi z jednoczesnym zapisem krawędzi.
        """
        if sum(seq) % 2 != 0:
            raise ValueError("Suma stopni jest nieparzysta, ciąg nie może być graficzny.")

        n = len(seq)
        vertices = [(i + 1, seq[i]) for i in range(n)]
        edges = set()

        while True:
            vertices.sort(key=lambda x: x[1], reverse=True)

            if vertices[0][1] == 0:
                break

            v_id, d = vertices[0]
            vertices[0] = (v_id, 0)

            if d > len(vertices) - 1:
                raise ValueError("Stopień wierzchołka większy niż liczba pozostałych wierzchołków.")

            for i in range(1, d + 1):
                w_id, w_deg = vertices[i]
                w_deg -= 1
                if w_deg < 0:
                    raise ValueError("W trakcie redukcji uzyskano ujemny stopień - ciąg nie jest graficzny.")
                vertices[i] = (w_id, w_deg)

                edge = (min(v_id, w_id), max(v_id, w_id))
                edges.add(edge)

        g = Graph(n)
        for (u, v) in edges:
            g.add_edge(u, v)

        return g

    def randomize_graph(g, num_swaps):
        """
        Wykonuje num_swaps prób "randomizacji" krawędzi w grafie g,
        tak aby zachować stopnie wierzchołków.
        """
        edges_list = list(g.edges)
        m = len(edges_list)

        for _ in range(num_swaps):
            if m < 2:
                break

            i1, i2 = random.sample(range(m), 2)
            a, b = edges_list[i1]
            c, d = edges_list[i2]

            if len({a, b, c, d}) < 4:
                continue

            ad = (min(a, d), max(a, d))
            bc = (min(b, c), max(b, c))

            if ad in g.edges or bc in g.edges:
                continue

            g.edges.remove((min(a, b), max(a, b)))
            g.edges.remove((min(c, d), max(c, d)))

            g.edges.add(ad)
            g.edges.add(bc)

            edges_list[i1] = ad
            edges_list[i2] = bc

        return g

    def get_connected_components(self):
        """
        Zwraca listę comp o długości n+1 (zakładając, że wierzchołki są numerowane od 1 do n),
        gdzie comp[v] to numer spójnej składowej wierzchołka v.
        """
        comp = [-1] * (self.n + 1)
        current_comp = 0

        def dfs(start, c):
            stack = [start]
            comp[start] = c
            while stack:
                v = stack.pop()
                for w in range(1, self.n + 1):
                    if (min(v, w), max(v, w)) in self.edges:
                        if comp[w] == -1:
                            comp[w] = c
                            stack.append(w)

        for v in range(1, self.n + 1):
            if comp[v] == -1:
                current_comp += 1
                dfs(v, current_comp)

        return comp

    @staticmethod
    def random_eulerian_graph(n):
        """
        Generuje i zwraca losowy graf eulerowski (spójny, każdy wierzchołek ma parzysty stopień)
        na n wierzchołkach, numerowanych od 1 do n.
        Pomysł:
        1) Tworzymy losowe drzewo spinające (zapewnia spójność).
        2) W drzewie niektóre wierzchołki będą miały stopień parzysty, inne nie.
        3) Łączymy w pary wierzchołki o stopniach nieparzystych i dodajemy między nimi krawędź.
           To czyni ich stopnie parzystymi.
        """
        g = Graph(n)
        vertices = list(range(1, n + 1))
        random.shuffle(vertices)

        for i in range(n - 1):
            g.add_edge(vertices[i], vertices[i + 1])

        def odd_vertices():
            deg = {v: 0 for v in range(1, n + 1)}
            for (u, w) in g.edges:
                deg[u] += 1
                deg[w] += 1
            return [v for v in deg if deg[v] % 2 == 1]

        odds = odd_vertices()
        while len(odds) >= 2:
            a = random.choice(odds)
            b = random.choice(odds)
            if a == b:
                continue
            e = (min(a, b), max(a, b))
            if e not in g.edges:
                g.add_edge(a, b)
            odds = odd_vertices()

        return g

    def find_euler_cycle_fleury(self):
        """
        Znajduje i zwraca cykl Eulera w grafie eulerowskim (spójnym, wszystkie stopnie parzyste),
        korzystając z idei algorytmu Fleury’ego.
        Zwraca listę wierzchołków w kolejności przechodzenia cyklu.
        """
        adj = defaultdict(set)
        for (u, v) in self.edges:
            adj[u].add(v)
            adj[v].add(u)

        start = 1
        for v in range(1, self.n + 1):
            if len(adj[v]) > 0:
                start = v
                break

        stack = [start]
        euler_cycle = []

        def is_bridge(u, w):
            """
            Sprawdza, czy krawędź (u,w) jest mostem w aktualnym 'adj'.
            Realizujemy to przez:
            1) usunięcie krawędzi (u,w),
            2) sprawdzenie, czy wierzchołek w jest nadal osiągalny z u (DFS),
            3) przywrócenie krawędzi.
            """
            adj[u].remove(w)
            adj[w].remove(u)

            visited = set()

            def dfs(x):
                visited.add(x)
                for nei in adj[x]:
                    if nei not in visited:
                        dfs(nei)

            dfs(u)
            adj[u].add(w)
            adj[w].add(u)
            return (w not in visited)

        while stack:
            u = stack[-1]
            if len(adj[u]) == 0:
                euler_cycle.append(u)
                stack.pop()
            else:
                found_non_bridge = False
                for w in adj[u]:
                    if len(adj[u]) == 1 or not is_bridge(u, w):
                        adj[u].remove(w)
                        adj[w].remove(u)
                        stack.append(w)
                        found_non_bridge = True
                        break
                if not found_non_bridge:
                    w = adj[u].pop()
                    adj[w].remove(u)
                    stack.append(w)

        return euler_cycle

    def is_eulerian(g):
        """
        Sprawdza, czy graf g jest eulerowski:
        1) spójny,
        2) każdy wierzchołek ma parzysty stopień.
        Zwraca True/False.
        """
        comp = g.get_connected_components()
        if max(comp[1:]) > 1:
            return False

        degrees = [0] * g.n
        for (u, v) in g.edges:
            degrees[u - 1] += 1
            degrees[v - 1] += 1

        for deg in degrees:
            if deg % 2 != 0:
                return False

        return True

    @staticmethod
    def random_k_regular(n, k, max_tries=1000, randomization_steps=100):
        """
        Generuje losowy graf k-regularny (bez pętli i krawędzi wielokrotnych).
        1) Sprawdza, czy k < n i n*k jest parzyste.
        2) Próbuje max_tries razy wylosować pary stubów (model konfiguracyjny).
        3) (Opcjonalnie) wykonuje randomization_steps losowych zamian krawędzi
           (tak jak w Zadaniu 2) dla "u-losowienia" struktury.

        Zwraca obiekt klasy Graph lub rzuca ValueError, jeśli nie udało się wygenerować.
        """
        if k >= n:
            raise ValueError("k musi być mniejsze od n (w prostym grafie nie możemy mieć pętli).")
        if (n * k) % 2 != 0:
            raise ValueError("n*k musi być parzyste, by istniał k-regularny graf.")

        for attempt in range(max_tries):
            stubs = []
            for v in range(1, n + 1):
                stubs.extend([v] * k)

            random.shuffle(stubs)
            edges = set()
            valid = True

            for i in range(0, len(stubs), 2):
                a, b = stubs[i], stubs[i + 1]
                if a == b:
                    valid = False
                    break
                edge = (min(a, b), max(a, b))
                if edge in edges:
                    valid = False
                    break
                edges.add(edge)

            if valid:
                g = Graph(n)
                for e in edges:
                    g.add_edge(*e)

                Graph.randomize_graph(g, randomization_steps)
                return g

        raise ValueError(f"Nie udało się wygenerować k-regularnego grafu po {max_tries} próbach.")

    def find_hamiltonian_cycle(self):
        """
        Próbuje znaleźć cykl Hamiltona w grafie (o wierzchołkach 1..n).
        Zwraca listę wierzchołków w kolejności przejścia cyklu (pierwszy == ostatni),
        jeśli taki cykl istnieje. W przeciwnym razie zwraca None.

        Metoda backtrackingowa (DFS) - dla małych n jest wystarczająca.
        """
        n = self.n
        comp = self.get_connected_components()
        if max(comp[1:]) != 1:
            return None

        adjacency_list = self.to_adjacency_list()

        visited = set()
        path = []

        def backtrack(current):
            if len(path) == n:
                start_vertex = path[0]
                if start_vertex in adjacency_list[current]:
                    path.append(start_vertex)
                    return True
                else:
                    return False

            for neighbor in adjacency_list[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    if backtrack(neighbor):
                        return True
                    visited.remove(neighbor)
                    path.pop()
            return False

        for start in range(1, n + 1):
            visited.clear()
            path.clear()
            visited.add(start)
            path.append(start)

            if backtrack(start):
                return path

        return None


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

class WeightedGraph(Graph):
    def __init__(self, n):
        super().__init__(n)
        self.weights = {}  

    def add_edge(self, u, v, weight=1):
        super().add_edge(u, v)
        edge = (min(u, v), max(u, v))
        self.weights[edge] = weight

    def randomize_weights(self, min_weight=1, max_weight=10):
        """Przypisz losowe wagi wszystkim istniejącym krawędziom"""
        for edge in self.edges:
            self.weights[edge] = random.randint(min_weight, max_weight)

    def visualize_weighted(self):
        """Wizualizacja grafu z wagami krawędzi"""
        G = nx.Graph()
        G.add_nodes_from(range(1, self.n + 1))
        for (u, v), weight in self.weights.items():
            G.add_edge(u, v, weight=weight)
        
        pos = nx.circular_layout(G)
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue")
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Graph with edge weights")
        plt.show()

    def to_adjacency_matrix(self):
        """Macierz sąsiedztwa z wagami (0 oznacza brak krawędzi)"""
        mat = [[0] * self.n for _ in range(self.n)]
        for (u, v), weight in self.weights.items():
            mat[u - 1][v - 1] = weight
            mat[v - 1][u - 1] = weight
        return mat

    def save_weighted_graph(self, filename):
        """Zapisanie grafu do pliku"""
        with open(filename, 'w') as f:
            for (u, v), weight in self.weights.items():
                f.write(f"{u} {v} {weight}\n")

    @staticmethod
    def load_weighted_graph(filename):
        """Wczytanie grau z pliku"""
        with open(filename, 'r') as f:
            edges = []
            max_vertex = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    u, v, weight = map(int, parts)
                    edges.append((u, v, weight))
                    max_vertex = max(max_vertex, u, v)
            
            g = WeightedGraph(max_vertex)
            for u, v, weight in edges:
                g.add_edge(u, v, weight)
            return g

    @staticmethod
    def random_connected_weighted_graph(n, p=0.5, min_weight=1, max_weight=10):
        """
        Generuje spójny graf losowy z wagami krawędzi
        n - liczba wierzchołków
        p - prawdopodobieństwo istnienia krawędzi (dla zapewnienia spójności najpierw generowane jest drzewo)
        min_weight, max_weight - zakres wag
        """
        g = WeightedGraph(n)
        vertices = list(range(1, n + 1))
        random.shuffle(vertices)
        
        for i in range(n - 1):
            weight = random.randint(min_weight, max_weight)
            g.add_edge(vertices[i], vertices[i + 1], weight)
        
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if (i, j) not in g.edges and random.random() < p:
                    weight = random.randint(min_weight, max_weight)
                    g.add_edge(i, j, weight)
        
        return g
    
    def dijkstra(self, start):
        """
        Algorytm Dijkstry znajdowania najkrótszych ścieżek od zadanego wierzchołka.
        Zwraca słownik odległości i słownik poprzedników.
        """
        distances = {v: float('inf') for v in range(1, self.n + 1)}
        distances[start] = 0
        predecessors = {v: None for v in range(1, self.n + 1)}
        
        priority_queue = []
        heapq.heappush(priority_queue, (0, start))
        
        visited = set()
        
        while priority_queue:
            current_dist, current_vertex = heapq.heappop(priority_queue)
            
            if current_vertex in visited:
                continue
                
            visited.add(current_vertex)
            
            for neighbor in range(1, self.n + 1):
                edge = (min(current_vertex, neighbor), max(current_vertex, neighbor))
                if edge in self.weights:
                    weight = self.weights[edge]
                    distance = current_dist + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        predecessors[neighbor] = current_vertex
                        heapq.heappush(priority_queue, (distance, neighbor))
        
        return distances, predecessors

    def get_shortest_paths(self, start):
        """
        Zwraca sformatowany wynik najkrótszych ścieżek od wierzchołka startowego.
        """
        distances, predecessors = self.dijkstra(start)
        
        result = []
        result.append(f"START : s = {start}")
        
        for vertex in range(1, self.n + 1):
            if vertex == start:
                path = [start]
            else:
                path = []
                current = vertex
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                path.reverse()
            
            distance = distances[vertex]
            result.append(f"d ({vertex}) = {distance} ==> [{' - '.join(map(str, path))}]")
        
        return result

    def compute_distance_matrix(self):
        """
        Oblicza macierz odległości między wszystkimi parami wierzchołków.
        Zwraca macierz n x n, gdzie n to liczba wierzchołków.
        """
        distance_matrix = []
        
        for start in range(1, self.n + 1):
            distances, _ = self.dijkstra(start)
            row = [distances[end] for end in range(1, self.n + 1)]
            distance_matrix.append(row)
        
        return distance_matrix

    def print_distance_matrix(self):
        """
        Wyświetla macierz odległości 
        """
        distance_matrix = self.compute_distance_matrix()
        
        for row in distance_matrix:
            print(" ".join(f"{val:3}" for val in row))

    def find_graph_center(self):
        """
        Znajduje centrum grafu - wierzchołek o minimalnej sumie odległości do innych wierzchołków.
        Zwraca krotkę (numer_wierzchołka, suma_odległości)
        """
        distance_matrix = self.compute_distance_matrix()
        sums = [sum(row) for row in distance_matrix]
        min_sum = min(sums)
        center = sums.index(min_sum) + 1 
        return center, min_sum

    def find_minimax_center(self):
        """
        Znajduje centrum minimax - wierzchołek o minimalnej maksymalnej odległości do innych wierzchołków.
        Zwraca krotkę (numer_wierzchołka, maksymalna_odległość)
        """
        distance_matrix = self.compute_distance_matrix()
        max_distances = [max(row) for row in distance_matrix]
        min_max = min(max_distances)
        center = max_distances.index(min_max) + 1 
        return center, min_max
    
    def prim_mst(self):
        """
        Znajduje minimalne drzewo rozpinające (MST) za pomocą algorytmu Prima.
        Zwraca obiekt WeightedGraph reprezentujący MST.
        """
        if self.n == 0:
            return WeightedGraph(0)

        mst = WeightedGraph(self.n)
        visited = set()
        start_node = 1
        visited.add(start_node)
        edges = []
        
        for neighbor in range(1, self.n + 1):
            edge = (min(start_node, neighbor), max(start_node, neighbor))
            if edge in self.weights:
                heapq.heappush(edges, (self.weights[edge], edge))

        while len(visited) < self.n and edges:
            weight, (u, v) = heapq.heappop(edges)
            
            if u in visited and v in visited:
                continue
                
            new_node = v if u in visited else u
            mst.add_edge(u, v, weight)
            visited.add(new_node)
            
            for neighbor in range(1, self.n + 1):
                edge = (min(new_node, neighbor), max(new_node, neighbor))
                if edge in self.weights and neighbor not in visited:
                    heapq.heappush(edges, (self.weights[edge], edge))

        return mst

    def visualize_mst(self):
        """
        Wizualizuje graf i jego minimalne drzewo rozpinające.
        """
        mst = self.prim_mst()
        
        G_original = nx.Graph()
        G_mst = nx.Graph()
        
        for v in range(1, self.n + 1):
            G_original.add_node(v)
            G_mst.add_node(v)
            
        for (u, v), weight in self.weights.items():
            G_original.add_edge(u, v, weight=weight)
            
        for (u, v), weight in mst.weights.items():
            G_mst.add_edge(u, v, weight=weight)
        
        plt.figure(figsize=(12, 6))
        
        pos = nx.circular_layout(G_original)
        
        plt.subplot(121)
        nx.draw(G_original, pos, with_labels=True, node_color='lightblue', 
                node_size=500, edge_color='gray')
        edge_labels = nx.get_edge_attributes(G_original, 'weight')
        nx.draw_networkx_edge_labels(G_original, pos, edge_labels=edge_labels)
        plt.title("Oryginalny graf z wagami")
        
        plt.subplot(122)
        nx.draw(G_mst, pos, with_labels=True, node_color='lightgreen', 
                node_size=500, edge_color='green', width=2)
        mst_edge_labels = nx.get_edge_attributes(G_mst, 'weight')
        nx.draw_networkx_edge_labels(G_mst, pos, edge_labels=mst_edge_labels)
        plt.title("Minimalne drzewo rozpinające (MST)")
        
        plt.tight_layout()
        plt.show()
