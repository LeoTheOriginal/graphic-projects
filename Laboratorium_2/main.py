from graph_lib import Graph, load_adjacency_matrix_from_file
import random


def main():
    print("Wybierz zadanie do wykonania:")
    print("1 - Sprawdzenie ciągu i budowa grafu")
    print("2 - Randomizacja grafu")
    print("3 - Znajdowanie największej spójnej składowej")
    print("4 - Tworzenie losowego grafu eulerowskiego i cykl Eulera")
    print("5 - Tworzenie losowego grafu k-regularnego")
    print("6 - Sprawdzanie czy graf jest hamiltonowski")
    choice = input("Podaj numer zadania (1/2/3/4/5/6): ").strip()

    if choice == '1':
        # Zadanie 1: ciąg graficzny
        seq_str = input("Podaj ciąg stopni (np. '4 2 2 3 2 1 4 2 2 2 2'): ")
        seq = list(map(int, seq_str.split()))

        if not Graph.is_graphic_sequence(seq):
            print("Ciąg NIE jest graficzny!")
            return

        print("Ciąg jest graficzny. Buduję graf...")
        try:
            g = Graph.graph_from_degree_sequence(seq)
            print("Graf został zbudowany.\n")

            # Wyświetlamy różne reprezentacje:
            print("=== Macierz sąsiedztwa ===")
            for row in g.to_adjacency_matrix():
                print(" ".join(map(str, row)))

            print("\n=== Lista sąsiedztwa ===")
            for v, neighbors in sorted(g.to_adjacency_list().items()):
                print(f"{v}: {neighbors}")

            print("\n=== Macierz incydencji ===")
            for row in g.to_incidence_matrix():
                print(" ".join(map(str, row)))

            g.visualize()

            do_random = input("\nCzy chcesz wykonać randomizację na tym grafie? (t/n): ").strip().lower()
            if do_random == 't':
                k_str = input("Ile randomizacji (ile prób zamiany krawędzi) wykonać? ")
                k = int(k_str)
                comp_before = g.get_connected_components()
                num_comp_before = max(comp_before[1:]) if len(comp_before) > 1 else 0
                print(f"Liczba spójnych składowych przed randomizacją: {num_comp_before}")
                Graph.randomize_graph(g, k)
                comp_after = g.get_connected_components()
                num_comp_after = max(comp_after[1:]) if len(comp_after) > 1 else 0
                print(f"Liczba spójnych składowych po randomizacji: {num_comp_after}")
                g.visualize()

        except ValueError as e:
            print("Błąd podczas budowania grafu:", e)

    elif choice == '2':
        # Zadanie 2: randomizacja grafu
        seq_str = input("Podaj ciąg stopni (graf musi być graficzny): ")
        seq = list(map(int, seq_str.split()))
        if not Graph.is_graphic_sequence(seq):
            print("Ciąg NIE jest graficzny, nie mogę zbudować grafu!")
            return

        g = Graph.graph_from_degree_sequence(seq)
        print("Graf zbudowany z ciągu.\n")

        comp_before = g.get_connected_components()
        num_comp_before = max(comp_before[1:]) if len(comp_before) > 1 else 0
        print(f"Liczba spójnych składowych przed randomizacją: {num_comp_before}")

        k_str = input("Ile randomizacji (ile prób zamiany krawędzi) wykonać? ")
        k = int(k_str)
        Graph.randomize_graph(g, k)

        comp_after = g.get_connected_components()
        num_comp_after = max(comp_after[1:]) if len(comp_after) > 1 else 0
        print(f"Liczba spójnych składowych po randomizacji: {num_comp_after}")
        g.visualize()

    elif choice == '3':
        # Zadanie 3: Znajdowanie największej spójnej składowej
        seq_str = input("Podaj ciąg stopni (graf musi być graficzny): ")
        seq = list(map(int, seq_str.split()))
        if not Graph.is_graphic_sequence(seq):
            print("Ciąg NIE jest graficzny, nie mogę zbudować grafu!")
            return

        g = Graph.graph_from_degree_sequence(seq)
        print("Graf zbudowany z ciągu.\n")

        comp = g.get_connected_components()
        components = {}
        for v in range(1, g.n + 1):
            cid = comp[v]
            if cid not in components:
                components[cid] = []
            components[cid].append(v)

        for cid in sorted(components.keys()):
            print(f"{cid}) " + " ".join(map(str, components[cid])))

        largest_comp = max(components.items(), key=lambda x: len(x[1]))[0]
        print(f"Największa składowa ma numer {largest_comp}.\n")

        g.visualize_components()

    elif choice == '4':
        # Zadanie 4: Losowy graf eulerowski (na podstawie losowego ciągu parzystych stopni)
        n_str = input("Podaj liczbę wierzchołków (n). Jeśli puste, wylosuję n z przedziału 5..10: ").strip()
        if n_str == "":
            n = random.randint(5, 10)
            print(f"Nie podano n, wylosowano n = {n}")
        else:
            n = int(n_str)

        if (n - 1) % 2 == 0:
            max_even = n - 1
        else:
            max_even = n - 2

        max_attempts = 1000
        valid_sequence = None
        for attempt in range(max_attempts):
            seq = [random.choice(list(range(2, max_even + 1, 2))) for _ in range(n)]
            if Graph.is_graphic_sequence(seq):
                try:
                    g = Graph.graph_from_degree_sequence(seq)
                    comp = g.get_connected_components()
                    if max(comp[1:]) == 1:
                        valid_sequence = seq
                        break
                except Exception as e:
                    continue

        if valid_sequence is None:
            print("Nie udało się wygenerować grafu eulerowskiego na podstawie losowego ciągu stopni.")
            return

        print("Wylosowany ciąg stopni:", " ".join(map(str, valid_sequence)))
        g.visualize()

        cycle = g.find_euler_cycle_fleury()
        print("Cykl Eulera (kolejne wierzchołki):")
        print(" - ".join(map(str, cycle)))

    elif choice == '5':
        # Zadanie 5: Generowanie losowego grafu k-regularnego
        n_str = input("Podaj liczbę wierzchołków n: ")
        k_str = input("Podaj stopień k: ")
        if not n_str or not k_str:
            print("Błędne dane, przerwanie.")
            return

        n = int(n_str)
        k = int(k_str)

        try:
            g = Graph.random_k_regular(n, k, max_tries=1000, randomization_steps=100)
            print(f"Wygenerowano losowy graf {k}-regularny o {n} wierzchołkach.")
            print(f"Liczba krawędzi: {len(g.edges)}")

            g.visualize()

        except ValueError as e:
            print("Błąd przy generowaniu k-regularnego grafu:", e)

    elif choice == '6':
        # Zadanie 6: Sprawdzenie, czy graf jest hamiltonowski, i znalezienie cyklu Hamiltona
        print("Możesz wczytać graf z pliku (np. macierz sąsiedztwa) albo podać ciąg stopni.")
        method = input("Wybierz (1 - plik z macierzą sąsiedztwa, 2 - ciąg stopni): ")

        if method == '1':
            filename = input("Podaj nazwę pliku z macierzą sąsiedztwa: ")
            g = load_adjacency_matrix_from_file(filename)
        elif method == '2':
            seq_str = input("Podaj ciąg stopni: ")
            seq = list(map(int, seq_str.split()))
            if not Graph.is_graphic_sequence(seq):
                print("Ciąg nie jest graficzny. Przerywam.")
                return
            g = Graph.graph_from_degree_sequence(seq)
        else:
            print("Nieprawidłowa opcja.")
            return

        g.visualize()

        cycle = g.find_hamiltonian_cycle()
        if cycle is None:
            print("Graf nie jest hamiltonowski (nie znaleziono cyklu Hamiltona).")
        else:
            print("Graf jest hamiltonowski. Oto znaleziony cykl Hamiltona:")
            print(" - ".join(map(str, cycle)))

    else:
        print("Nieprawidłowy wybór. Uruchom ponownie program.")


if __name__ == "__main__":
    main()
