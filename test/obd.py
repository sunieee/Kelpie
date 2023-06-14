import math
from collections import defaultdict
import numpy as np

def overlapping_block_division(neighbors, m):
    n = len(neighbors)
    k = math.ceil(math.log(n, m))
    N = m ** k
    cnt = n // m
    print(f"n: {n}, m: {m}, k: {k}, N: {N}, cnt: {cnt}")

    group_id_to_elements = {}
    element_id_to_groups = defaultdict(list)

    # fill neighbors with -1 until it has N elements
    neighbors += [-1] * (N - n)
    # create a k-dim matrix with m elements in each dimension, and fill it with the elements in neighbors
    matrix = np.array(neighbors).reshape((m,) * k)

    for i in range(k):
        # get m slices from the i-th dimension and store them in a list(group), group_id = m * i + j
        for j in range(m):
            group = matrix.take(j, axis=i).flatten()
            group_id = m * i + j
            group_id_to_elements[group_id] = [element for element in group if element != -1]
            for element in group_id_to_elements[group_id]:
                element_id_to_groups[element].append(group_id)

    return group_id_to_elements, element_id_to_groups


if __name__ == "__main__":
    neighbors = list(range(1, 245))  # Example list of elements
    m = 4  # Maximum group count

    group_id_to_elements, element_id_to_groups = overlapping_block_division(neighbors, m)

    print("Group ID to Elements:")
    for group_id, elements in group_id_to_elements.items():
        print(f"Group {group_id}: {elements}")

    print("\nElement ID to Groups:")
    for element_id, groups in element_id_to_groups.items():
        print(f"Element {element_id}: {groups}")
