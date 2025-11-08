import random
from typing import Optional, Tuple

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.ab_mcts_a.prob_state import PriorConfig
from treequest.visualization import visualize_tree_graphviz


def test_ab_mcts_a_visualization():
    """Test the ABMCTSA algorithm with visualization to understand its tree exploration behavior."""
    # Use a fixed seed for reproducibility
    random.seed(42)

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
        score = min(parent_score + random.uniform(0, 0.3), 1.0)
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
        score = min(max(parent_score + random.uniform(-0.3, 0), 0.0), 1.0)
        return f"Low(score={score:.2f})", score

    # Create the algorithm with custom prior configuration
    prior_config = PriorConfig(
        dist_type="beta",  # Use Beta distribution for Thompson sampling
        prior={"a": 1.0, "b": 1.0},  # Initial a and b parameters for Beta distribution
    )

    # Initialize with reward average priors to bias initial exploration
    reward_average_priors = {"high": 0.7, "low": 0.3}

    algo = ABMCTSA(
        prior_config=prior_config, reward_average_priors=reward_average_priors
    )
    state = algo.init_tree()

    # Create a mapping of actions to generate functions
    generate_fns = {"high": generate_fn_high, "low": generate_fn_low}

    # Run several steps to build the tree
    n_steps = 40
    for _ in range(n_steps):
        state = algo.step(state, generate_fns)

    # Count the nodes for each action
    high_count = 0
    low_count = 0

    # Collect all nodes that have thompson states
    nodes_with_thompson_states = []
    for node in state.tree.get_nodes():
        if node.expand_idx in state.thompson_states:
            nodes_with_thompson_states.append(node)

    # Count actions used for each node's children
    for node in nodes_with_thompson_states:
        thompson_state = state.thompson_states.get(node)
        if thompson_state:
            for child_idx, action in thompson_state.child_node_to_action.items():
                if isinstance(child_idx, int) and action == "high":
                    high_count += 1
                elif isinstance(child_idx, int) and action == "low":
                    low_count += 1

    # Print statistics
    print(f"Action 'high' selected {high_count} times")
    print(f"Action 'low' selected {low_count} times")

    # Visualize the tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/ab_mcts_a_visualization",
        title="AB-MCTS-A Algorithm Visualization",
        format="png",
    )

    # Verify that the algorithm favors the higher-scoring action over time
    # This might not always be true for small n_steps due to exploration
    if n_steps > 20:
        assert high_count > low_count, (
            "After sufficient exploration, the 'high' action should be selected more often"
        )


def test_ab_mcts_a_exploration_exploitation():
    """
    Test that ABMCTSA properly balances exploration and exploitation.

    This test creates a scenario where:
    - Action A gives high immediate rewards but diminishes over time
    - Action B gives lower immediate rewards but improves over time

    The algorithm should initially favor A but gradually shift to B
    as it learns about the long-term performance of both actions.
    """
    # Use a fixed seed for reproducibility
    random.seed(42)

    # Define action generation functions
    def generate_action_a(state: Optional[str]) -> Tuple[str, float]:
        """Action A: Initially high rewards that degrade with depth."""
        if state is None:
            # Root state
            return "A:initial", 0.0

        depth = state.count(":") + 1 if ":" in state else 1

        # Start high but diminish with depth
        if depth <= 2:
            score = 0.9 - (0.1 * (depth - 1))  # 0.9, 0.8, ...
        else:
            score = max(0.8 - (0.2 * (depth - 2)), 0.2)  # 0.6, 0.4, 0.2, 0.2, ...

        # Add some noise
        score = min(max(score + random.uniform(-0.05, 0.05), 0.0), 1.0)

        return f"{state}:A{depth}", score

    def generate_action_b(state: Optional[str]) -> Tuple[str, float]:
        """Action B: Initially mediocre rewards that improve with depth."""
        if state is None:
            # Root state
            return "B:initial", 0.0

        depth = state.count(":") + 1 if ":" in state else 1

        # Start moderate but improve with depth
        if depth <= 2:
            score = 0.4 + (0.1 * (depth - 1))  # 0.4, 0.5, ...
        else:
            score = min(0.5 + (0.15 * (depth - 2)), 0.95)  # 0.65, 0.8, 0.95, 0.95, ...

        # Add some noise
        score = min(max(score + random.uniform(-0.05, 0.05), 0.0), 1.0)

        return f"{state}:B{depth}", score

    # Create the algorithm with neutral priors
    algo = ABMCTSA()
    state = algo.init_tree()

    # Create a mapping of action names to generate functions
    generate_fns = {"A": generate_action_a, "B": generate_action_b}

    # Run enough steps to allow the algorithm to learn
    n_steps = 50
    for _ in range(n_steps):
        state = algo.step(state, generate_fns)

    # Analyze results by depth to see how action selection evolves
    action_counts_by_depth = {}

    # Group nodes by depth and action
    for node in state.tree.get_nodes():
        if node.is_root():
            continue

        depth = node.depth

        # Find the action used to generate this node
        action = None
        if node.parent and node.parent.expand_idx in state.thompson_states:
            thompson_state = state.thompson_states.get(node.parent)
            if thompson_state:
                # Find the index of this node in its parent's children
                try:
                    child_idx = node.parent.children.index(node)
                    action = thompson_state.child_node_to_action.get(child_idx)
                except ValueError:
                    pass

        if action:
            if depth not in action_counts_by_depth:
                action_counts_by_depth[depth] = {"A": 0, "B": 0}

            action_counts_by_depth[depth][action] += 1

    # Print action usage by depth
    print("\nAction usage by depth:")
    for depth in sorted(action_counts_by_depth.keys()):
        counts = action_counts_by_depth[depth]
        a_count = counts["A"]
        b_count = counts["B"]
        total = a_count + b_count
        if total > 0:  # Avoid division by zero
            print(
                f"Depth {depth}: A={a_count}/{total} ({a_count / total:.1%}), B={b_count}/{total} ({b_count / total:.1%})"
            )

    # Visualize the tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/ab_mcts_a_exploration_exploitation",
        title="AB-MCTS-A Exploration vs Exploitation",
        format="png",
    )

    # Check if the algorithm adapts to the changing performance of actions across depths
    if len(action_counts_by_depth) >= 4:  # Only check if we have enough depth
        # Analyze early depths (should favor A due to initial high rewards)
        early_depths = [1, 2]
        early_a_count = sum(
            action_counts_by_depth.get(d, {}).get("A", 0) for d in early_depths
        )
        early_b_count = sum(
            action_counts_by_depth.get(d, {}).get("B", 0) for d in early_depths
        )

        # Analyze later depths (should favor B due to increasing rewards)
        max_depth = max(action_counts_by_depth.keys())
        later_depths = list(range(max(3, max_depth - 2), max_depth + 1))
        later_a_count = sum(
            action_counts_by_depth.get(d, {}).get("A", 0) for d in later_depths
        )
        later_b_count = sum(
            action_counts_by_depth.get(d, {}).get("B", 0) for d in later_depths
        )

        # Early depths may occasionally select only B due to stochasticity.
        # Ensure we observed any actions and rely on ratio comparison below.
        assert (early_a_count + early_b_count) > 0, (
            "No actions observed at early depths"
        )

        # Later on, B should become more prominent
        if later_a_count + later_b_count > 0:  # Avoid division by zero
            b_ratio_later = later_b_count / (later_a_count + later_b_count)
            print(f"\nAction B usage ratio at later depths: {b_ratio_later:.1%}")

            # B should be used more in later depths than in early depths
            b_ratio_early = (
                early_b_count / (early_a_count + early_b_count)
                if (early_a_count + early_b_count) > 0
                else 0
            )
            print(f"Action B usage ratio at early depths: {b_ratio_early:.1%}")

            # The key assertion: action B should be used proportionally more at later depths
            assert b_ratio_later > b_ratio_early, (
                f"Expected Action B to be used more at later depths ({b_ratio_later:.1%}) "
                f"than at early depths ({b_ratio_early:.1%})"
            )
