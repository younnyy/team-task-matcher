from itertools import combinations, permutations

from algorithms import hungarian_maximum_matching_stepwise, km_optimal_matching_stepwise


def brute_force_maximum_cardinality(left_nodes, right_nodes, edges):
    edge_set = set(edges)
    best_size = 0
    for size in range(min(len(left_nodes), len(right_nodes)) + 1):
        for selected_left in combinations(left_nodes, size):
            for selected_right in combinations(right_nodes, size):
                for permuted_right in permutations(selected_right):
                    if all((student, task) in edge_set for student, task in zip(selected_left, permuted_right)):
                        best_size = max(best_size, size)
    return best_size


def test_hungarian_matches_bruteforce_on_small_graph():
    left_nodes = ["a", "b", "c"]
    right_nodes = ["1", "2", "3"]
    edges = [("a", "1"), ("a", "2"), ("b", "2"), ("c", "2"), ("c", "3")]

    matching, _ = hungarian_maximum_matching_stepwise(left_nodes, right_nodes, edges)

    assert len(matching) == brute_force_maximum_cardinality(left_nodes, right_nodes, edges)


def test_km_finds_maximum_weight_assignment():
    left_nodes = ["a", "b", "c"]
    right_nodes = ["1", "2", "3"]
    weights = {
        "a": {"1": 3.0, "2": 1.0, "3": 2.0},
        "b": {"1": 2.0, "2": 5.0, "3": 1.0},
        "c": {"1": 4.0, "2": 2.0, "3": 6.0},
    }

    matching, total_weight, _ = km_optimal_matching_stepwise(left_nodes, right_nodes, weights)

    assert matching == {"1": "a", "2": "b", "3": "c"}
    assert total_weight == 14.0
