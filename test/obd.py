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
    # neighbors = list(range(1, 245))  # Example list of elements
    # m = 4  # Maximum group count

    neighbors = [0, 2562, 6150, 3590, 6153, 6159, 10769, 2068, 1046, 27, 32, 3104, 1058, 6694, 4135, 1576, 2600, 2091, 1585, 8754, 1080, 4154, 9281, 8770, 68, 4682, 8277, 8791, 10327, 5209, 92, 607, 6755, 3686, 3176, 622, 119, 8312, 637, 640, 1164, \
                1682, 8339, 9372, 158, 6816, 1705, 9901, 181, 202, 208, 7385, 11491, 5349, 9448, 2282, 2794, 1260, 7916, 748, 2795, 235, 1786, 1280, 7429, 8970, 6411, 4875, 8461, 3855, 274, 1304, 7467, 7468, 301, 816, 5426, 8500, 308, 2874, \
                1340, 834, 853, 2394, 3420, 4445, 1891, 2410, 1394, 4474, 382, 3456, 7553, 4483, 1928, 6029, 1934, 6547, 8603, 6561, 2978, 419, 7590, 7080, 945, 6583, 10168, 10167, 449, 8139, 9174, 9177, 474, 8667, 2015, 5087, 993, 8677, 8168, 2046]
    m = max(math.ceil(len(neighbors) / 30), 3)

    group_id_to_elements, element_id_to_groups = overlapping_block_division(neighbors, m)

    print("Group ID to Elements:")
    for group_id, elements in group_id_to_elements.items():
        print(f"Group {group_id}: {elements}")

    print("\nElement ID to Groups:")
    for element_id, groups in element_id_to_groups.items():
        print(f"Element {element_id}: {groups}")
