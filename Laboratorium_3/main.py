from graph_lib import WeightedGraph

n = 10  # liczba wierzchołków
p = 0.4  # prawdopodobieństwo krawędzi
graph = WeightedGraph.random_connected_weighted_graph(n, p)

#zadanie 1
print("=== ZADANIE 1 ===")
graph.visualize_weighted()
print("\nMacierz sąsiedztwa z wagami:")
for row in graph.to_adjacency_matrix():
    print(row)
graph.save_weighted_graph("weighted_graph.txt")

#zadanie 2
print("\n=== ZADANIE 2 ===")
start_vertex = 1
shortest_paths = graph.get_shortest_paths(start_vertex)
for line in shortest_paths:
    print(line)

#zadanie 3 
print("\n=== ZADANIE 3 ===")
print("Macierz odległości między wszystkimi wierzchołkami:")
graph.print_distance_matrix()

#zadanie 4
print("\n=== ZADANIE 4 ===")
center, total_distance = graph.find_graph_center()
minimax_center, max_distance = graph.find_minimax_center()
print(f"Centrum = {center} (suma odległości: {total_distance})")
print(f"Centrum minimax = {minimax_center} (odległość od najdalszego: {max_distance})")

#zadanie 5
print("\n=== ZADANIE 5 ===")
print("Minimalne drzewo rozpinające:")
graph.visualize_mst()
mst = graph.prim_mst()
for edge, weight in sorted(mst.weights.items()):
    print(f"{edge[0]} - {edge[1]} : {weight}")