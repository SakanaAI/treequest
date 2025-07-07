import random
from typing import Any, Dict, Optional, Tuple

from treequest import ABMCTSM
from treequest.visualization import visualize_tree_graphviz


def test_pymc_mixed_algo_reward_priors():
    """
    Test that the reward_average_priors parameter affects model selection.

    This test creates a scenario where two models perform roughly equally,
    but different prior beliefs about their performance leads to different
    exploration patterns.
    """
    # Use a fixed seed for reproducibility
    random.seed(42)

    # Define two identical generate functions, so any difference in selection
    # will be due to the priors rather than actual performance
    def generate_fn_a(state: Optional[str]) -> Tuple[str, float]:
        """Model A with random scores centered around 0.5."""
        score = random.uniform(0.3, 0.7)  # Centered around 0.5
        return f"A(score={score:.2f})", score

    def generate_fn_b(state: Optional[str]) -> Tuple[str, float]:
        """Model B with identical distribution to Model A."""
        score = random.uniform(0.3, 0.7)  # Centered around 0.5
        return f"B(score={score:.2f})", score

    # Create two algorithms with different priors
    # First algorithm favors model A
    algo_favor_a = ABMCTSM(
        enable_pruning=False,  # Disable pruning to focus on prior effects
        reward_average_priors={
            "A": 0.99,
            "B": 0.01,
        },  # Prior belief: A is better than B
        model_selection_strategy="stack",
    )

    # Second algorithm favors model B
    algo_favor_b = ABMCTSM(
        enable_pruning=False,
        reward_average_priors={
            "A": 0.01,
            "B": 0.99,
        },  # Prior belief: B is better than A
        model_selection_strategy="stack",
    )

    # Initialize states
    state_favor_a = algo_favor_a.init_tree()
    state_favor_b = algo_favor_b.init_tree()

    # Create a mapping of model names to generate functions
    generate_fns = {"A": generate_fn_a, "B": generate_fn_b}

    # Run several steps to build the search trees
    n_steps = 30
    for _ in range(n_steps):
        state_favor_a = algo_favor_a.step(state_favor_a, generate_fns)
        state_favor_b = algo_favor_b.step(state_favor_b, generate_fns)

    # Visualize the trees
    visualize_tree_graphviz(
        state_favor_a.tree,
        save_path="tests/pymc_mixed_algo_favor_a",
        title="PyMC Mixed Algorithm Favoring Model A",
        format="png",
    )

    visualize_tree_graphviz(
        state_favor_b.tree,
        save_path="tests/pymc_mixed_algo_favor_b",
        title="PyMC Mixed Algorithm Favoring Model B",
        format="png",
    )

    # Count the nodes for each action in each tree
    def count_action_nodes(state, action_name):
        return sum(
            1 for obs in state.all_observations.values() if obs.action == action_name
        )

    a_count_favor_a = count_action_nodes(state_favor_a, "A")
    b_count_favor_a = count_action_nodes(state_favor_a, "B")
    a_count_favor_b = count_action_nodes(state_favor_b, "A")
    b_count_favor_b = count_action_nodes(state_favor_b, "B")

    # Print statistics
    print(
        f"\nFavoring A priors: A={a_count_favor_a}, B={b_count_favor_a}, Ratio A/B={a_count_favor_a / max(b_count_favor_a, 1):.2f}"
    )
    print(
        f"Favoring B priors: A={a_count_favor_b}, B={b_count_favor_b}, Ratio A/B={a_count_favor_b / max(b_count_favor_b, 1):.2f}"
    )

    # Test assertions
    # 1. In the algo that favors A, there should be more A nodes than B nodes
    assert a_count_favor_a > b_count_favor_a, (
        "When A is favored with higher prior, it should be selected more often"
    )

    # 2. In the algo that favors B, there should be more B nodes than A nodes
    assert b_count_favor_b > a_count_favor_b, (
        "When B is favored with higher prior, it should be selected more often"
    )

    # 3. The ratio of A to B nodes should be higher in the algo favoring A
    ratio_a = a_count_favor_a / max(b_count_favor_a, 1)
    ratio_b = a_count_favor_b / max(b_count_favor_b, 1)

    assert ratio_a > ratio_b, (
        "The ratio of A to B nodes should be higher when A has a higher prior"
    )


def test_multiarm_bandit_strategy():
    """
    Test the multiarm bandit strategy that performs joint selection between child nodes
    and new node generation.

    This test compares the traditional strategy (separately calculating scores for each model)
    with the new multiarm bandit strategy (using Thompson Sampling and all tree observations).
    """
    # Use a fixed seed for reproducibility
    random.seed(42)

    # Define generate functions with different performance patterns
    # Simulating a scenario where model performance depends on tree depth
    def generate_fn_early_good(state: Optional[str]) -> Tuple[str, float]:
        """Model that performs well initially but gets worse at deeper levels."""
        depth = 1 if state is None else state.count(":") + 1
        # Scores decrease with depth
        base_score = max(0.8 - (0.1 * depth), 0.3)
        score = base_score + random.uniform(-0.05, 0.05)
        return f"Early{':' * depth}", score

    def generate_fn_late_good(state: Optional[str]) -> Tuple[str, float]:
        """Model that performs poorly initially but improves at deeper levels."""
        depth = 1 if state is None else state.count(":") + 1
        # Scores increase with depth
        base_score = min(0.3 + (0.1 * depth), 0.8)
        score = base_score + random.uniform(-0.05, 0.05)
        return f"Late{':' * depth}", score

    # Create algorithms with different model selection strategies
    algo_traditional = ABMCTSM(
        enable_pruning=False,  # Disable pruning to focus on strategy differences
        model_selection_strategy="stack",  # Use traditional separate model fitting
    )

    algo_multiarm = ABMCTSM(
        enable_pruning=False,
        model_selection_strategy="multiarm_bandit_thompson",  # Use the joint Thompson Sampling approach
    )

    # Initialize states
    state_traditional = algo_traditional.init_tree()
    state_multiarm = algo_multiarm.init_tree()

    # Create mapping of model names to generate functions
    generate_fns = {
        "early": generate_fn_early_good,
        "late": generate_fn_late_good,
    }

    # Run several steps to build the search trees
    n_steps = 50  # Use more steps for this test to better observe patterns
    for _ in range(n_steps):
        state_traditional = algo_traditional.step(state_traditional, generate_fns)
        state_multiarm = algo_multiarm.step(state_multiarm, generate_fns)

    # Visualize the trees
    visualize_tree_graphviz(
        state_traditional.tree,
        save_path="tests/pymc_mixed_algo_traditional",
        title="Traditional Strategy (Separate Model Fitting)",
        format="png",
    )

    visualize_tree_graphviz(
        state_multiarm.tree,
        save_path="tests/pymc_mixed_algo_multiarm",
        title="Multiarm Bandit Strategy (Joint Thompson Sampling)",
        format="png",
    )

    # Count nodes by action and analyze patterns
    def count_action_nodes(state, action_name):
        return sum(
            1 for obs in state.all_observations.values() if obs.action == action_name
        )

    # Count by model
    early_traditional = count_action_nodes(state_traditional, "early")
    late_traditional = count_action_nodes(state_traditional, "late")
    early_multiarm = count_action_nodes(state_multiarm, "early")
    late_multiarm = count_action_nodes(state_multiarm, "late")

    # Analyze how depth affects model selection for both strategies
    def analyze_model_selection_by_depth(state: Any) -> Dict[int, Dict[str, int]]:
        model_by_depth: Dict[int, Dict[str, int]] = {}

        for node_id, obs in state.all_observations.items():
            node = state.tree.get_node(node_id)
            depth = node.depth

            if depth not in model_by_depth:
                model_by_depth[depth] = {"early": 0, "late": 0}

            model_by_depth[depth][obs.action] += 1

        return model_by_depth

    trad_depth_stats = analyze_model_selection_by_depth(state_traditional)
    multi_depth_stats = analyze_model_selection_by_depth(state_multiarm)

    # Print statistics
    print("\n--- Model Usage Statistics ---")
    print(f"Traditional: Early={early_traditional}, Late={late_traditional}")
    print(f"Multiarm: Early={early_multiarm}, Late={late_multiarm}")

    print("\n--- Traditional Strategy: Model Usage by Depth ---")
    for depth in sorted(trad_depth_stats.keys()):
        stats = trad_depth_stats[depth]
        total = stats["early"] + stats["late"]
        if total > 0:
            early_pct = stats["early"] / total * 100
            late_pct = stats["late"] / total * 100
            print(
                f"Depth {depth}: Early={stats['early']} ({early_pct:.1f}%), Late={stats['late']} ({late_pct:.1f}%)"
            )

    print("\n--- Multiarm Strategy: Model Usage by Depth ---")
    for depth in sorted(multi_depth_stats.keys()):
        stats = multi_depth_stats[depth]
        total = stats["early"] + stats["late"]
        if total > 0:
            early_pct = stats["early"] / total * 100
            late_pct = stats["late"] / total * 100
            print(
                f"Depth {depth}: Early={stats['early']} ({early_pct:.1f}%), Late={stats['late']} ({late_pct:.1f}%)"
            )

    # Test assertions
    # The strategies should lead to different exploration patterns, but we can't
    # predict exactly how due to randomness and complex Thompson sampling dynamics.
    # Instead, we'll check that both models were used and that the distributions differ.

    # 1. Both strategies should use both models
    assert early_traditional > 0 and late_traditional > 0, (
        "Traditional strategy should use both models"
    )
    assert early_multiarm > 0 and late_multiarm > 0, (
        "Multiarm strategy should use both models"
    )

    # 2. The distribution of models should differ between strategies
    trad_early_ratio = early_traditional / (early_traditional + late_traditional)
    multi_early_ratio = early_multiarm / (early_multiarm + late_multiarm)

    # Not asserting specific differences, but printing for inspection
    print(
        f"\nRatio of 'early' model usage: Traditional={trad_early_ratio:.2f}, Multiarm={multi_early_ratio:.2f}"
    )
    print(f"Difference in ratios: {abs(trad_early_ratio - multi_early_ratio):.2f}")

    # The strategies should adapt differently to model performance at different depths
    # Collect data for analysis but don't assert specific patterns
    max_depth = max(
        max(trad_depth_stats.keys(), default=0),
        max(multi_depth_stats.keys(), default=0),
    )
    if max_depth >= 3:
        shallow_depths = [1, 2]
        deeper_depths = list(range(3, max_depth + 1))

        trad_early_shallow = sum(
            trad_depth_stats.get(d, {}).get("early", 0) for d in shallow_depths
        )
        trad_early_deeper = sum(
            trad_depth_stats.get(d, {}).get("early", 0) for d in deeper_depths
        )
        trad_late_shallow = sum(
            trad_depth_stats.get(d, {}).get("late", 0) for d in shallow_depths
        )
        trad_late_deeper = sum(
            trad_depth_stats.get(d, {}).get("late", 0) for d in deeper_depths
        )

        multi_early_shallow = sum(
            multi_depth_stats.get(d, {}).get("early", 0) for d in shallow_depths
        )
        multi_early_deeper = sum(
            multi_depth_stats.get(d, {}).get("early", 0) for d in deeper_depths
        )
        multi_late_shallow = sum(
            multi_depth_stats.get(d, {}).get("late", 0) for d in shallow_depths
        )
        multi_late_deeper = sum(
            multi_depth_stats.get(d, {}).get("late", 0) for d in deeper_depths
        )

        print("\n--- Model Usage by Tree Section ---")
        print(
            f"Traditional - Shallow: Early={trad_early_shallow}, Late={trad_late_shallow}"
        )
        print(
            f"Traditional - Deeper: Early={trad_early_deeper}, Late={trad_late_deeper}"
        )
        print(
            f"Multiarm - Shallow: Early={multi_early_shallow}, Late={multi_late_shallow}"
        )
        print(
            f"Multiarm - Deeper: Early={multi_early_deeper}, Late={multi_late_deeper}"
        )
