"""Tests for visualization snapshot generation."""

import random
from typing import Optional, Tuple

from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.vis import build_snapshot
from treequest.vis.errors import InvalidStateError


def test_build_snapshot_standard_mcts():
    """Test building a snapshot from StandardMCTS state."""
    random.seed(42)

    # Define generate functions
    def generate_fn_high(state: Optional[str]) -> Tuple[str, float]:
        parent_score = 0.5
        if state is not None and "score=" in state:
            try:
                parent_score = float(state.split("score=")[1].split(")")[0])
            except (IndexError, ValueError):
                pass  # Use default if parsing fails
        score = min(parent_score + random.uniform(-0.1, 0.3), 1.0)
        return f"High(score={score:.2f})", score

    def generate_fn_low(state: Optional[str]) -> Tuple[str, float]:
        parent_score = 0.5
        if state is not None and "score=" in state:
            try:
                parent_score = float(state.split("score=")[1].split(")")[0])
            except (IndexError, ValueError):
                pass  # Use default if parsing fails
        score = min(max(parent_score + random.uniform(-0.3, 0.1), 0.0), 1.0)
        return f"Low(score={score:.2f})", score

    # Create algorithm and run steps
    algo = StandardMCTS(samples_per_action=2, exploration_weight=1.0)
    state = algo.init_tree()

    generate_fns = {"high": generate_fn_high, "low": generate_fn_low}

    # Run several steps
    for _ in range(10):
        state = algo.step(state, generate_fns)

    # Build snapshot
    snapshot = build_snapshot(state)

    # Verify snapshot structure
    assert len(snapshot.nodes) > 0
    assert len(snapshot.edges) >= 0
    assert snapshot.metadata["algorithm"] == "StandardMCTS"
    assert snapshot.metadata["num_nodes"] == len(snapshot.nodes) - 1

    # Verify root node exists
    root_nodes = [n for n in snapshot.nodes if n.id == -1]
    assert len(root_nodes) == 1
    assert root_nodes[0].state_repr == "ROOT"
    assert root_nodes[0].parent_id is None

    # Verify non-root nodes have parents
    for node in snapshot.nodes:
        if node.id != -1:
            assert node.parent_id is not None
            assert node.score >= 0.0 and node.score <= 1.0

    # Verify edges match parent-child relationships
    assert len(snapshot.edges) == len(snapshot.nodes) - 1

    # Verify MCTS-specific metrics are present
    non_root_with_metrics = [
        n for n in snapshot.nodes if n.id != -1 and len(n.algo_metrics) > 0
    ]
    assert len(non_root_with_metrics) > 0


def test_build_snapshot_best_first_search():
    """Test building a snapshot from BestFirstSearch state."""
    random.seed(42)

    # Define generate function
    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        score = random.uniform(0.0, 1.0)
        return f"State(score={score:.2f})", score

    # Create algorithm and run steps
    algo = BestFirstSearchAlgo(num_samples=2)
    state = algo.init_tree()

    generate_fns = {"action": generate_fn}

    # Run several steps
    for _ in range(10):
        state = algo.step(state, generate_fns)

    # Build snapshot
    snapshot = build_snapshot(state)

    # Verify snapshot structure
    assert len(snapshot.nodes) > 0
    assert snapshot.metadata["algorithm"] == "BestFirstSearch"

    # Verify root node
    root_nodes = [n for n in snapshot.nodes if n.id == -1]
    assert len(root_nodes) == 1


def test_build_snapshot_with_custom_formatter():
    """Test building a snapshot with custom state formatter."""
    random.seed(42)

    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        score = random.uniform(0.0, 1.0)
        return "Custom state", score

    algo = StandardMCTS(samples_per_action=1)
    state = algo.init_tree()

    generate_fns = {"action": generate_fn}

    for _ in range(5):
        state = algo.step(state, generate_fns)

    # Build snapshot with custom formatter
    snapshot = build_snapshot(state, state_formatter=lambda s: f"Formatted: {s}")

    # Verify custom formatter was applied
    non_root_nodes = [n for n in snapshot.nodes if n.id != -1]
    for node in non_root_nodes:
        assert node.state_repr.startswith("Formatted:")


def test_build_snapshot_with_annotations():
    """Test building a snapshot with annotations."""
    algo = StandardMCTS()
    state = algo.init_tree()

    annotations = {"experiment": "test_run", "version": "1.0"}

    snapshot = build_snapshot(state, annotations=annotations)

    # Verify annotations are in metadata
    assert "experiment" in snapshot.metadata
    assert snapshot.metadata["experiment"] == "test_run"
    assert "version" in snapshot.metadata
    assert snapshot.metadata["version"] == "1.0"


def test_build_snapshot_invalid_state():
    """Test that building a snapshot with invalid state raises error."""
    # Object without tree attribute
    invalid_state = {"not": "a tree"}

    try:
        build_snapshot(invalid_state)
        assert False, "Should have raised InvalidStateError"
    except InvalidStateError:
        pass


def test_snapshot_trial_information():
    """Test that trial information is properly linked in snapshot."""
    random.seed(42)

    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        score = random.uniform(0.0, 1.0)
        return "State", score

    algo = StandardMCTS(samples_per_action=1)
    state = algo.init_tree()

    generate_fns = {"action1": generate_fn, "action2": generate_fn}

    # Run steps
    for _ in range(5):
        state = algo.step(state, generate_fns)

    # Build snapshot
    snapshot = build_snapshot(state)

    # Verify trials exist
    assert len(snapshot.trials) > 0

    # Verify nodes have trial_id (except root)
    non_root_nodes = [n for n in snapshot.nodes if n.id != -1]
    nodes_with_trial_id = [n for n in non_root_nodes if n.trial_id is not None]
    assert len(nodes_with_trial_id) > 0

    # Verify action_from_parent is populated
    nodes_with_action = [n for n in non_root_nodes if n.action_from_parent is not None]
    assert len(nodes_with_action) > 0
