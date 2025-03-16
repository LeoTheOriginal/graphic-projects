from graph_lib import (
    load_adjacency_matrix_from_file,
    load_incidence_matrix_from_file,
    load_adjacency_list_from_file,
    save_graph_representations,
    Graph
)

def main():
    input_file_adj = "graph_adjacency_matrix.txt"
    output_file_adj = "graph_adjacency_matrix_out.txt"

    graph_adj = load_adjacency_matrix_from_file(input_file_adj)
    save_graph_representations(graph_adj, output_file_adj)
    print(f"Graf z {input_file_adj} zostal zapisany w {output_file_adj}")
    graph_adj.visualize()

    input_file_inc = "graph_incidence_matrix.txt"
    output_file_inc = "graph_incidence_matrix_out.txt"
    graph_inc = load_incidence_matrix_from_file(input_file_inc)
    save_graph_representations(graph_inc, output_file_inc)
    print(f"Graf z {input_file_inc} zostal zapisany w {output_file_inc}")
    graph_inc.visualize()

    input_file_list = "graph_adjacency_list.txt"
    output_file_list = "graph_adjacency_list_out.txt"
    graph_list = load_adjacency_list_from_file(input_file_list)
    save_graph_representations(graph_list, output_file_list)
    print(f"Graf z {input_file_list} zostal zapisany w {output_file_list}")
    graph_list.visualize()


    n, l = 7, 10
    random_graph_l = Graph.random_graph_G_n_l(n, l)
    print(f"Wygenerowano graf losowy G({n}, {l}):")
    save_graph_representations(random_graph_l, "random_G_n_l_out.txt")
    random_graph_l.visualize()

    n, p = 7, 0.4
    random_graph_p = Graph.random_graph_G_n_p(n, p)
    print(f"Wygenerowano graf losowy G({n}, {p}):")
    save_graph_representations(random_graph_p, "random_G_n_p_out.txt")
    random_graph_p.visualize()

if __name__ == "__main__":
    main()