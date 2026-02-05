import random
from typing import Optional, Tuple

import numpy as np

from treequest.algos.standard_mcts import StandardMCTS, softmax
from treequest.visualization import visualize_tree_graphviz


def test_standard_algo_mcts():
    """Test the StandardAlgo MCTS implementation."""
    # Use a fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define two generate functions with different score distributions
    def generate_fn_high(state: Optional[str]) -> Tuple[str, float]:
        # Action that tends to produce higher scores but with some variability
        parent_score = 0.5
        if state is not None and "score=" in state:
            try:
                parent_score = float(state.split("score=")[1].split(")")[0])
            except (IndexError, ValueError):
                pass

        # Scores tend to increase but with randomness
        score = min(parent_score + random.uniform(-0.1, 0.3), 1.0)
        return f"High(score={score:.2f})", score

    def generate_fn_low(state: Optional[str]) -> Tuple[str, float]:
        # Action that tends to produce lower scores
        parent_score = 0.5
        if state is not None and "score=" in state:
            try:
                parent_score = float(state.split("score=")[1].split(")")[0])
            except (IndexError, ValueError):
                pass

        # Scores tend to decrease but with randomness
        score = min(max(parent_score + random.uniform(-0.3, 0.1), 0.0), 1.0)
        return f"Low(score={score:.2f})", score

    # Create the algorithm with moderate exploration weight
    algo = StandardMCTS(samples_per_action=2, exploration_weight=1.0)
    state = algo.init_tree()

    # Create a mapping of actions to generate functions
    generate_fns = {"high": generate_fn_high, "low": generate_fn_low}

    # Run several steps to build the tree
    n_steps = 50
    for step in range(n_steps):
        state = algo.step(state, generate_fns)

    # Check that we have the expected number of nodes
    state_score_pairs = algo.get_state_score_pairs(state)
    assert len(state_score_pairs) == n_steps

    # Visualize the tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/standard_mcts",
        title="MCTS Standard Algorithm",
        format="png",
    )

    # Test prior calculations
    # Find a node with children to test
    for node in state.tree.get_nodes():
        if len(node.children) > 1:
            # Check if priors were calculated correctly
            child_scores = [child.score for child in node.children]
            expected_priors = softmax(child_scores)

            # Check that priors in state match expected values
            for child, expected_prior in zip(node.children, expected_priors):
                stored_prior = state.priors.get(child.expand_idx, 0.0)
                assert abs(stored_prior - expected_prior) < 1e-5, (
                    f"Prior for child node should be {expected_prior}, but is {stored_prior}"
                )

            # No need to check more nodes
            break

    # Test UCT scoring - indirectly by checking if high-scoring nodes are selected more
    # Group nodes by their depth in the tree
    nodes_by_depth = {}
    for node in state.tree.get_nodes():
        if node != state.tree.root:
            depth = node.depth
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

    # For each depth, check if nodes with "High" in their state are more common
    for depth, nodes in nodes_by_depth.items():
        if len(nodes) >= 5:  # Only check depths with enough nodes
            high_nodes = sum(1 for node in nodes if "High" in str(node.state))
            low_nodes = sum(1 for node in nodes if "Low" in str(node.state))

            # Print statistics for debugging
            print(f"Depth {depth}: {high_nodes} high nodes, {low_nodes} low nodes")

            # This assertion might not always be true due to the randomness,
            # especially early in the tree, but should generally hold after
            # enough samples at deeper depths
            if depth >= 3:
                assert high_nodes >= low_nodes, (
                    f"At depth {depth}, expected more 'High' nodes than 'Low' nodes, "
                    f"but found {high_nodes} 'High' and {low_nodes} 'Low'"
                )

    # Test backpropagation by checking a path from a leaf to root
    leaves = [
        node
        for node in state.tree.get_nodes()
        if not node.children and node != state.tree.root
    ]
    if leaves:
        # Pick a random leaf
        leaf = random.choice(leaves)

        # Follow path to root
        current = leaf
        node_ids = []
        while current:
            node_ids.append(current.expand_idx)
            current = current.parent

        # All nodes in the path should have visit counts
        for node_id in node_ids:
            assert node_id in state.visit_counts, (
                f"Node {node_id} should have a visit count"
            )
            assert state.visit_counts[node_id] > 0, (
                f"Node {node_id} should have a positive visit count"
            )


def test_standard_algo_exploration():
    """
    Test that StandardAlgo properly balances exploration and exploitation.

    This test creates a scenario where exploration is critical for finding the optimal solution:
    - Path A gives high immediate rewards (0.9) but poor long-term rewards (0.1)
    - Path B gives modest immediate rewards (0.3-0.4) but increases with depth

    An algorithm focused too much on exploitation will get stuck in the A path,
    while one that properly explores will discover the superior long-term value of the B path.

    We test this by comparing two identical MCTS implementations with different exploration weights:
    - High exploration (2.0): Should find more B-path nodes and deeper B-path nodes
    - Low exploration (0.1): Should focus more on the initially promising A-path
    """
    # Set up a scenario where exploration is important with separate actions

    def generate_fn_a(parent: Optional[str]) -> Tuple[str, float]:
        """Action that initially appears promising but leads to poor rewards."""
        if parent is None:
            # From root, create an A path node
            return "A:0.0", 0.0

        # For non-root nodes, continue A path
        parent_path = parent.split(":")[0]
        depth = parent_path.count("-") + 1 if "-" in parent_path else 1

        # First step after root gets high reward to attract exploitation
        if depth == 1:
            score = 0.9  # Very high initial score - exploitation trap
        else:
            score = 0.1  # Poor subsequent scores - exploitation penalty

        new_state = f"{parent_path}-A:{score:.2f}"
        return new_state, score

    def generate_fn_b(parent: Optional[str]) -> Tuple[str, float]:
        """
        Action that initially appears worse but improves with depth.
        This represents the optimal long-term strategy that requires exploration to discover.
        """
        if parent is None:
            # From root, create a B path node
            return "B:0.0", 0.0

        # For non-root nodes, continue B path
        parent_path = parent.split(":")[0]
        depth = parent_path.count("-") + 1 if "-" in parent_path else 1

        # Score increases with depth - better long-term strategy
        score = 0.3 + (depth * 0.1)  # Gradually increasing scores
        score = min(score, 1.0)  # Cap at 1.0

        new_state = f"{parent_path}-B:{score:.2f}"
        return new_state, score

    # Create algorithms with different exploration weights
    high_explore_algo = StandardMCTS(
        samples_per_action=1, exploration_weight=2.0
    )  # Higher exploration weight = more willingness to try less promising paths
    low_explore_algo = StandardMCTS(
        samples_per_action=1, exploration_weight=0.1
    )  # Lower exploration weight = more focus on exploiting known good paths

    high_explore_state = high_explore_algo.init_tree()
    low_explore_state = low_explore_algo.init_tree()

    # Use two separate actions representing different strategic choices
    generate_fns = {"A": generate_fn_a, "B": generate_fn_b}

    # Run enough steps to allow deep exploration of both paths
    n_steps = 40
    for _ in range(n_steps):
        high_explore_state = high_explore_algo.step(high_explore_state, generate_fns)
        low_explore_state = low_explore_algo.step(low_explore_state, generate_fns)

    # ---------- Analysis of Results ----------

    # Measure 1: Count nodes at deeper levels of the B path
    # The better explorer should find more deep B-path nodes
    def count_deep_b_nodes(state, min_depth=3):
        deep_b_count = 0
        for node in state.tree.get_nodes():
            if node.depth >= min_depth and "B" in str(node.state):
                deep_b_count += 1
        return deep_b_count

    high_explore_deep_b = count_deep_b_nodes(high_explore_state)
    low_explore_deep_b = count_deep_b_nodes(low_explore_state)

    # Measure 2: Count total nodes in each path to compare distribution
    def count_path_nodes(state, path_marker):
        return sum(
            1
            for node in state.tree.get_nodes()
            if node != state.tree.root and path_marker in str(node.state)
        )

    high_a_nodes = count_path_nodes(high_explore_state, "A")
    high_b_nodes = count_path_nodes(high_explore_state, "B")
    low_a_nodes = count_path_nodes(low_explore_state, "A")
    low_b_nodes = count_path_nodes(low_explore_state, "B")

    print(f"High exploration: {high_a_nodes} A-path nodes, {high_b_nodes} B-path nodes")
    print(f"Low exploration: {low_a_nodes} A-path nodes, {low_b_nodes} B-path nodes")
    print(f"High exploration found {high_explore_deep_b} deep B-path nodes")
    print(f"Low exploration found {low_explore_deep_b} deep B-path nodes")

    # Measure 3: Calculate B-to-A ratio - higher ratio means more focus on the better long-term path
    high_ratio = high_b_nodes / high_a_nodes if high_a_nodes > 0 else float("inf")
    low_ratio = low_b_nodes / low_a_nodes if low_a_nodes > 0 else float("inf")

    print(f"High exploration B-to-A ratio: {high_ratio:.2f}")
    print(f"Low exploration B-to-A ratio: {low_ratio:.2f}")

    # Measure 4: Compare average scores at leaf nodes for overall performance
    high_leaves = [
        n
        for n in high_explore_state.tree.get_nodes()
        if not n.children and n != high_explore_state.tree.root
    ]
    low_leaves = [
        n
        for n in low_explore_state.tree.get_nodes()
        if not n.children and n != low_explore_state.tree.root
    ]

    high_avg_score = (
        sum(n.score for n in high_leaves) / len(high_leaves) if high_leaves else 0
    )
    low_avg_score = (
        sum(n.score for n in low_leaves) / len(low_leaves) if low_leaves else 0
    )

    print(f"High exploration average leaf score: {high_avg_score:.2f}")
    print(f"Low exploration average leaf score: {low_avg_score:.2f}")

    # The test passes if any of our exploratory metrics show improvement (higher B-to-A ratio,
    # more deep B nodes, or a higher average leaf score).
    # This makes the test robust to variations in random initialization
    assert (
        high_ratio > low_ratio  # Better path balance
        or high_explore_deep_b > low_explore_deep_b  # More deep exploration
        or high_avg_score > low_avg_score  # Better overall performance
    ), (
        f"Expected high exploration to show better long-term strategy, but:\n"
        f"B/A ratio: high={high_ratio:.2f}, low={low_ratio:.2f}\n"
        f"Deep B nodes: high={high_explore_deep_b}, low={low_explore_deep_b}\n"
        f"Avg leaf score: high={high_avg_score:.2f}, low={low_avg_score:.2f}"
    )

    # Visualize both trees for manual inspection
    visualize_tree_graphviz(
        high_explore_state.tree,
        save_path="tests/mcts_high_exploration",
        title="MCTS with High Exploration",
        format="png",
    )

    visualize_tree_graphviz(
        low_explore_state.tree,
        save_path="tests/mcts_low_exploration",
        title="MCTS with Low Exploration",
        format="png",
    )
