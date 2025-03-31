from graph_lib import DirectedGraph, WeightedDirectedGraph

def zadanie1():
    print("=== ZADANIE 1: DIGRAFY I GENEROWANIE LOSOWYCH DIGRAFÓW ===")
    
    # 1. Tworzenie przykładowego digrafu zgodnego z przykładem
    print("\n1. Przykładowy digraf (zgodny z przykładem):")
    n = 7
    digraph = DirectedGraph(n)
    
    # Dodawanie krawędzi zgodnie z listą sąsiedztwa z przykładu
    digraph.add_edge(1, 2)
    digraph.add_edge(1, 3)
    digraph.add_edge(1, 5)
    
    digraph.add_edge(2, 1)
    digraph.add_edge(2, 3)
    digraph.add_edge(2, 4)
    digraph.add_edge(2, 5)
    digraph.add_edge(2, 7)
    
    digraph.add_edge(3, 6)
    
    digraph.add_edge(4, 2)
    digraph.add_edge(4, 7)
    
    digraph.add_edge(5, 7)
    
    digraph.add_edge(6, 2)
    
    digraph.add_edge(7, 6)
    
    print("Macierz sąsiedztwa:")
    for row in digraph.to_adjacency_matrix():
        print(row)
    
    print("\nLista sąsiedztwa:")
    adj_list = digraph.to_adjacency_list()
    for vertex, neighbors in adj_list.items():
        print(f"{vertex}: {neighbors}")
    
    print("\nStopnie wierzchołków (wchodzące/wychodzące):")
    in_degrees, out_degrees = digraph.get_degrees()
    for v in range(1, n + 1):
        print(f"Wierzchołek {v}: in={in_degrees[v]}, out={out_degrees[v]}")
    
    # Wizualizacja przykładowego digrafu
    digraph.visualize()
    
    # 2. Generowanie losowego digrafu z G(n,p)
    print("\n2. Losowy digraf z G(n,p):")
    n_random = 7  # zmienione na 7 dla spójności z przykładem
    p = 0.4  # zwiększone prawdopodobieństwo dla lepszego pokrycia
    random_digraph = DirectedGraph.random_digraph(n_random, p)
    
    print(f"Generowanie digrafu z G({n_random}, {p}):")
    print("Macierz sąsiedztwa:")
    for row in random_digraph.to_adjacency_matrix():
        print(row)
    
    print("\nLista sąsiedztwa:")
    adj_list = random_digraph.to_adjacency_list()
    for vertex, neighbors in adj_list.items():
        print(f"{vertex}: {neighbors}")
    
    # Wizualizacja losowego digrafu
    random_digraph.visualize()
    
    # 3. Zapisywanie i wczytywanie digrafu
    print("\n3. Zapisywanie i wczytywanie digrafu:")
    random_digraph.save_digraph("random_digraph.txt")
    loaded_digraph = DirectedGraph.load_digraph("random_digraph.txt")
    
    print("Wczytany digraf - macierz sąsiedztwa:")
    for row in loaded_digraph.to_adjacency_matrix():
        print(row)
    
    return random_digraph  # zwracamy wylosowany digraf do następnego zadania

def zadanie2(digraph):
    print("\n=== ZADANIE 2: ALGORYTM KOSARAJU - SILNIE SPÓJNE SKŁADOWE ===")
    
    # Znajdowanie silnie spójnych składowych
    sccs = digraph.kosaraju_scc()
    
    print("\nSilnie spójne składowe:")
    for i, scc in enumerate(sccs, 1):
        print(f"SCC {i}: {scc}")
    
    # Sprawdzenie czy digraf jest silnie spójny
    is_strongly_connected = digraph.is_strongly_connected()
    print(f"\nCzy digraf jest silnie spójny? {'Tak' if is_strongly_connected else 'Nie'}")
    
    return is_strongly_connected  # zwracamy informację czy digraf jest silnie spójny

def zadanie3():
    print("\n=== ZADANIE 3: LOSOWY SILNIE SPÓJNY DIGRAF Z WAGAMI I ALGORYTM BELLMANA-FORDA ===")
    
    # Generowanie losowego silnie spójnego digrafu z wagami
    n = 6  # zmniejszona liczba wierzchołków
    weighted_digraph = WeightedDirectedGraph.random_weighted_digraph(n, p=0.3, min_weight=-5, max_weight=10)
    
    print("\nWylosowany digraf z wagami:")
    print("Macierz sąsiedztwa (z wagami):")
    for u in range(1, n + 1):
        row = []
        for v in range(1, n + 1):
            if (u, v) in weighted_digraph.weights:
                row.append(weighted_digraph.weights[(u, v)])
            else:
                row.append(0)
        print(row)
    
    print("\nLista sąsiedztwa (z wagami):")
    adj_list = weighted_digraph.to_adjacency_list()
    for vertex, neighbors in adj_list.items():
        weights = [weighted_digraph.weights[(vertex, v)] for v in neighbors]
        print(f"{vertex}: {list(zip(neighbors, weights))}")
    
    # Wizualizacja digrafu z wagami
    weighted_digraph.visualize_weighted()
    
    # Znajdowanie najkrótszych ścieżek od wierzchołka 1
    print("\nNajkrótsze ścieżki od wierzchołka 1 (algorytm Bellmana-Forda):")
    shortest_paths = weighted_digraph.get_shortest_paths(1)
    for line in shortest_paths:
        print(line)
    
    return weighted_digraph  # zwracamy digraf do następnego zadania

def zadanie4(weighted_digraph):
    print("\n=== ZADANIE 4: ALGORYTM JOHNSONA - ODLEGŁOŚCI MIĘDZY WSZYSTKIMI PARAMI WIERZCHOŁKÓW ===")
    
    # Wyświetlanie macierzy odległości między wszystkimi parami wierzchołków
    weighted_digraph.print_all_pairs_distances()

def main():
    # Wykonujemy zadania w kolejności
    random_digraph = zadanie1()
    is_strongly_connected = zadanie2(random_digraph)
    
    # Jeśli digraf jest silnie spójny, możemy przejść do następnych zadań
    if is_strongly_connected:
        print("\nDigraf jest silnie spójny - przechodzimy do zadań 3 i 4")
        weighted_digraph = zadanie3()
        zadanie4(weighted_digraph)
    else:
        print("\nDigraf nie jest silnie spójny - nie można przejść do zadań 3 i 4")

if __name__ == "__main__":
    main()

