import random
from collections import defaultdict
from typing import Optional, Tuple

from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo
from treequest.visualization import visualize_tree_graphviz


def test_tree_of_thoughts_bfs():
    """Test the TreeOfThoughtsBFSAlgo implementation."""
    # Fix RNG for determinism across environments
    random.seed(42)

    # Define a deterministic generate function that increases score with depth
    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        if state is None:  # Root node
            depth = 0
        else:
            # Extract depth and base score from previous state
            try:
                parts = state.split("depth=")[1].split(",")
                depth = int(parts[0])
            except (IndexError, ValueError):
                depth = 0

        # New depth is one more than parent
        new_depth = depth + 1

        # Score increases with depth but add some randomness
        # This will test that higher-scoring nodes at the same depth are preferred
        new_score = 0.1 * new_depth + random.random() * 0.3

        new_score = min(max(new_score, 0.0), 1.0)

        return f"State(depth={new_depth}, score={new_score:.2f})", new_score

    # Create the algorithm with small limits for testing
    breadth_limit = 2  # select 2 best nodes at each depth
    size_limit = 3  # generate 3 children for each selected node

    algo = TreeOfThoughtsBFSAlgo(breadth_limit=breadth_limit, size_limit=size_limit)
    state = algo.init_tree()

    # Create a mapping with a single action (ToT-BFS uses only one action)
    generate_fns = {"action": generate_fn}

    # Run several steps
    n_steps = 20
    for _ in range(n_steps):
        state = algo.step(state, generate_fns)

    # Get all nodes in the tree
    all_nodes = state.tree.get_nodes()
    # Group nodes by depth
    nodes_by_depth = defaultdict(list)
    for node in all_nodes:
        if not node.is_root():  # Skip the root node
            nodes_by_depth[node.depth].append(node)

    # Check that expanded nodes are among the top-scoring at their depth
    for depth, nodes in nodes_by_depth.items():
        # Find nodes that have children (expanded nodes)
        expanded_nodes = [node for node in nodes if len(node.children) > 0]

        if expanded_nodes:  # Only check if there are expanded nodes at this depth
            # Sort nodes by score in descending order
            sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)
            top_nodes = sorted_nodes[: len(expanded_nodes)]
            # Use a score threshold to tolerate score ties deterministically
            threshold = top_nodes[-1].score if top_nodes else float("-inf")

            assert len(expanded_nodes) <= breadth_limit, (
                f"There should be at most {breadth_limit} expanded nodes at depth {depth}, "
                f"but there are {len(expanded_nodes)}"
            )
            # Check that all expanded nodes meet the top-k score threshold
            for expanded_node in expanded_nodes:
                assert expanded_node.score + 1e-9 >= threshold, (
                    f"Expanded node at depth {depth} with score {expanded_node.score} "
                    f"should be >= top-{breadth_limit} threshold {threshold}"
                )

    # Visualize the tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/tot_bfs_test",
        title="Tree of Thoughts BFS",
        format="png",
    )
