import random
from typing import Optional, Tuple

from treequest import ABMCTSM
from treequest.visualization import visualize_tree_graphviz


def test_pymc_mixed_algo_basic():
    """Test the basic functionality of the ABMCTSM."""
    # Use a fixed seed for reproducibility
    random.seed(42)

    # Define two generate functions with different score distributions
    def generate_fn_high(state: Optional[str]) -> Tuple[str, float]:
        # Model that tends to produce higher scores but with variability
        parent_score = 0.5
        if state is not None and "score=" in state:
            try:
                parent_score = float(state.split("score=")[1].split(")")[0])
            except (IndexError, ValueError):
                pass

        # Scores tend to increase but with randomness
        score = min(max(parent_score + random.uniform(-0.1, 0.3), 0.0), 1.0)
        return f"High(score={score:.2f})", score

    def generate_fn_low(state: Optional[str]) -> Tuple[str, float]:
        # Model that tends to produce lower scores
        parent_score = 0.5
        if state is not None and "score=" in state:
            try:
                parent_score = float(state.split("score=")[1].split(")")[0])
            except (IndexError, ValueError):
                pass

        # Scores tend to decrease but with randomness
        score = min(max(parent_score + random.uniform(-0.3, 0.1), 0.0), 1.0)
        return f"Low(score={score:.2f})", score

    # Create the algorithm with default parameters
    algo = ABMCTSM(enable_pruning=True)
    state = algo.init_tree()

    # Create a mapping of model names to generate functions
    generate_fns = {"high": generate_fn_high, "low": generate_fn_low}

    # Run several steps to build the search tree
    n_steps = 30
    for step in range(n_steps):
        state = algo.step(state, generate_fns)

    # Check that we have generated some nodes
    state_score_pairs = algo.get_state_score_pairs(state)
    assert len(state_score_pairs) > 0, "Algorithm should generate nodes"

    # Check that observations are recorded
    assert len(state.all_observations) > 0, "Algorithm should record observations"

    # Visualize the tree
    visualize_tree_graphviz(
        state.tree,
        save_path="tests/pymc_mixed_algo_basic",
        title="PyMC Mixed Algorithm Basic Test",
        format="png",
    )

    # Check that both models were used at least once
    model_counts = dict()
    for obs in state.all_observations.values():
        model = obs.action
        model_counts[model] = model_counts.get(model, 0) + 1

    assert "high" in model_counts, "The 'high' model should be used at least once"
    assert "low" in model_counts, "The 'low' model should be used at least once"

    print(f"Model usage counts: {model_counts}")
