import random
from typing import Optional, Tuple

import numpy as np

from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.visualization import visualize_tree_graphviz


def test_best_first_search_expansion_order():
    """Test that BestFirstSearchAlgo correctly selects highest-scoring nodes for expansion."""
    # Use a fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create generate functions with deterministic scores
    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        # Create a unique, deterministic score based on the state
        if state is None:  # Root node
            base_score = random.random()
        else:
            # Extract score from previous state if possible
            try:
                prev_score = float(state.split("score ")[1].split(")")[0])
                # Slightly modify the score to create variety
                base_score = (prev_score + random.random()) / 2
            except (IndexError, ValueError):
                base_score = random.random()

        # Ensure score is between 0 and 1
        score = min(max(base_score, 0.0), 1.0)
        return f"State (score {score})", score

    # Create the algorithm
    num_samples = 2
    algo = BestFirstSearchAlgo(num_samples=num_samples)
    state = algo.init_tree()

    # Create a mapping of actions to generate functions
    generate_fns = {"A": generate_fn, "B": generate_fn}

    # Run several steps
    for step in range(15):
        # Simulate the heappop to find the highest priority node without modifying the real heap.
        # This is necessary because the step function modifies the heap, so we need to know
        # what the heap looks like *before* the step to verify the selection.
        expected_best_node = (
            state.leaves[0].node if len(state.leaves) > 0 else state.tree.root
        )

        for _ in range(num_samples * len(generate_fns)):
            # Perform the step
            state = algo.step(state, generate_fns)

        actual_parent_node = state.tree.get_nodes()[-1].parent
        assert actual_parent_node.expand_idx == expected_best_node.expand_idx, (
            f"Step {step}: BFS should select the highest-scoring node for expansion. "
            f"Expected Node: {expected_best_node}, Got Node: {actual_parent_node}"
            f"Tree: {state.tree}"
        )

    # Visualize the final tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/bfs_expansion_order",
        title="BFS Expansion Order Test",
        format="png",
    )

    # Final check: Verify heap property still holds
    if state.leaves:
        for i in range(1, len(state.leaves)):
            assert state.leaves[0].sort_index <= state.leaves[i].sort_index, (
                "Heap should maintain sort order (highest score first)"
            )
