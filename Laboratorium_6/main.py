from graph_lib import DirectedGraph
import random
import numpy as np


def pagerank_random_walk(graph, d=0.15, steps=100000):
    n = graph.n
    visits = [0] * n
    current = random.randint(1, n)
    adj_list = graph.to_adjacency_list()
    for _ in range(steps):
        if random.random() < d or not adj_list[current]:
            current = random.randint(1, n)
        else:
            current = random.choice(adj_list[current])
        visits[current - 1] += 1
    total = sum(visits)
    return [v / total for v in visits]


def pagerank_power_iteration(graph, d=0.15, max_iter=100, tol=1e-8):
    n = graph.n
    A = np.array(graph.to_adjacency_matrix())
    out_deg = np.sum(A, axis=1)
    P = np.zeros((n, n))
    for i in range(n):
        if out_deg[i] == 0:
            P[i, :] = 1.0 / n
        else:
            P[i, :] = (1 - d) * A[i, :] / out_deg[i] + d / n
    p = np.ones(n) / n
    for _ in range(max_iter):
        p_new = P.T @ p
        if np.linalg.norm(p_new - p, 1) < tol:
            break
        p = p_new
    return p.tolist()


def parse_custom_adjacency_list(adj_text):
    label_to_idx = {}
    idx_to_label = {}
    edges = []
    lines = adj_text.strip().split('\n')
    for line in lines:
        parts = line.split(':')
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ''
        idx_label, node_label = left.split()
        idx = int(idx_label)
        label_to_idx[node_label] = idx
        idx_to_label[idx] = node_label
    for line in lines:
        parts = line.split(':')
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ''
        idx_label, node_label = left.split()
        u = int(idx_label)
        if right:
            neighbors = [x.strip().strip(',') for x in right.split() if x.strip(',')]
            for v_label in neighbors:
                v = label_to_idx[v_label]
                edges.append((u, v))
    n = len(label_to_idx)
    return n, edges, idx_to_label


def zadanie1(adj_text=None):
    print("=== ZADANIE 1: PageRank dla digrafu ===")
    d = 0.15
    if adj_text:
        n, edges, idx_to_label = parse_custom_adjacency_list(adj_text)
        digraph = DirectedGraph(n)
        for u, v in edges:
            digraph.add_edge(u, v)
        print("Macierz sąsiedztwa:")
        for row in digraph.to_adjacency_matrix():
            print(row)
    else:
        n = 7
        p = 0.4
        print(f"\nGenerowanie losowego digrafu G({n}, {p})...")
        digraph = DirectedGraph.random_digraph(n, p)
        print("Macierz sąsiedztwa:")
        for row in digraph.to_adjacency_matrix():
            print(row)
    print("\nObliczanie PageRank metodą błądzenia przypadkowego z teleportacją...")
    pr_walk = pagerank_random_walk(digraph, d=d, steps=100000)
    print("Wyniki (random walk):")
    pr_walk_sorted = sorted(enumerate(pr_walk, 1), key=lambda x: -x[1])
    for i, val in pr_walk_sorted:
        label = idx_to_label[i] if adj_text else i
        print(f"Wierzchołek {label}: {val:.6f}")
    print("\nObliczanie PageRank metodą iteracji wektora obsadzeń...")
    pr_power = pagerank_power_iteration(digraph, d=d, max_iter=100)
    pr_power_sorted = sorted(enumerate(pr_power, 1), key=lambda x: -x[1])
    print("Wyniki (power iteration):")
    for i, val in pr_power_sorted:
        label = idx_to_label[i] if adj_text else i
        print(f"Wierzchołek {label}: {val:.6f}")
    print("\nPorównanie wyników:")
    comparison = [(i, idx_to_label[i] if adj_text else i, v1, v2, abs(v1-v2)) for i, (v1, v2) in enumerate(zip(pr_walk, pr_power), 1)]
    comparison_sorted = sorted(comparison, key=lambda x: -max(x[2], x[3]))
    for i, label, v1, v2, diff in comparison_sorted:
        print(f"Wierzchołek {label}: random walk = {v1:.4f}, power iteration = {v2:.4f}, różnica = {diff:.4e}")


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def total_path_length(path, points):
    return sum(distance(points[path[i]], points[path[(i+1)%len(path)]]) for i in range(len(path)))


def two_opt_swap(path):
    n = len(path)
    i, j = sorted(random.sample(range(n), 2))
    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
    return new_path


def simulated_annealing_tsp(points, T_start=1000.0, T_end=1e-6, alpha=0.9995, max_iter=1000000):
    n = len(points)
    path = list(range(n))
    random.shuffle(path)
    best_path = path[:]
    best_length = total_path_length(path, points)
    T = T_start
    for step in range(max_iter):
        new_path = two_opt_swap(path)
        old_len = total_path_length(path, points)
        new_len = total_path_length(new_path, points)
        if new_len < old_len or random.random() < np.exp((old_len - new_len) / T):
            path = new_path
            if new_len < best_length:
                best_length = new_len
                best_path = new_path[:]
        T *= alpha
        if step % 10000 == 0:
            print(f"Step {step}, T={T:.6f}, Best length so far: {best_length:.2f}")
        if T < T_end:
            break
    return best_path, best_length


def zadanie2():
    print("\n=== ZADANIE 2: Najkrótsza zamknięta droga przez zadane wierzchołki (symulowane wyżarzanie) ===")
    # Load points from xqf131.dat
    filename = "xqf131.dat"
    points = []
    with open(filename, 'r') as f:
        for line in f:
            x, y = map(int, line.strip().split())
            points.append((x, y))
    n = len(points)
    print(f"Wczytano {n} punktów z pliku {filename}.")
    print("Współrzędne pierwszych 10 punktów:")
    for i, (x, y) in enumerate(points[:10], 1):
        print(f"{i}: ({x}, {y})")
    print("\nUruchamianie algorytmu symulowanego wyżarzania (2-opt, Metropolis-Hastings)...")
    best_path, best_length = simulated_annealing_tsp(points, T_start=1000.0, T_end=1e-6, alpha=0.9995, max_iter=1000000)
    print("Najlepsza znaleziona trasa (indeksy od 1):")
    print([i+1 for i in best_path])
    print(f"Długość trasy: {best_length:.2f}")
    # Opcjonalnie: wizualizacja
    try:
        import matplotlib.pyplot as plt
        px = [points[i][0] for i in best_path] + [points[best_path[0]][0]]
        py = [points[i][1] for i in best_path] + [points[best_path[0]][1]]
        plt.figure(figsize=(8,8))
        plt.plot(px, py, 'o-', label='Trasa')
        plt.title('Najkrótsza zamknięta trasa (symulowane wyżarzanie)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib nie jest zainstalowany, pomijam wizualizację.")


def main():
    custom_adj = '''
    1 A : E , F , I
    2 B : A , C , F
    3 C : B , D , E , L
    4 D : C , E , H , I , K
    5 E : C , G , H , I
    6 F : B , G
    7 G : E , F , H
    8 H : D , G , I , L
    9 I : D , E , H , J
    10 J : I
    11 K : D , I
    12 L : A , H
    '''

    zadanie1(adj_text=custom_adj)
    zadanie2()


if __name__ == "__main__":
    main()
