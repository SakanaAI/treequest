import random
from typing import Optional, Tuple

import pytest

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.multi_armed_bandit_ucb import MultiArmedBanditUCBAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo


def _result_from_trial(action: str, parent_state: Optional[str]) -> Tuple[str, float]:
    # Deterministic and valid result tuple for tell
    # Keep score within [0,1]
    return f"Generated(action={action}, parent={parent_state})", 0.5


@pytest.mark.parametrize(
    "algo_factory,actions,batch_size,expect_length",
    [
        # All algos should return exactly batch_size trials
        (
            lambda: StandardMCTS(samples_per_action=2, exploration_weight=1.0),
            ["A", "B"],
            2,
            2,
        ),
        (lambda: BestFirstSearchAlgo(num_samples=2), ["A", "B"], 2, 2),
        (
            lambda: TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=3),
            ["A", "B"],
            1,
            1,
        ),
        (lambda: MultiArmedBanditUCBAlgo(exploration_weight=1.0), ["A", "B"], 3, 3),
        (lambda: ABMCTSA(), ["A", "B"], 3, 3),
    ],
)
def test_ask_batch_core_behaviour(algo_factory, actions, batch_size, expect_length):
    random.seed(0)
    algo = algo_factory()
    state = algo.init_tree()

    # Ask for a batch
    state, trials = algo.ask_batch(state, batch_size=batch_size, actions=actions)
    assert len(trials) == expect_length

    # Trials have valid fields
    for t in trials:
        assert t.action in actions
        # Node id should exist in the tree
        state.tree.get_node(t.node_to_expand)

    # Tell results for the first expect_length trials and check node growth
    before = len(state.tree.get_state_score_pairs())
    for t in trials[:expect_length]:
        result = _result_from_trial(t.action, t.parent_state)
        state = algo.tell(state, t.trial_id, result)

    after = len(state.tree.get_state_score_pairs())
    assert after - before == expect_length


def test_abmctsm_ask_batch_minimal():
    # Import ABMCTSM conditionally; skip if optional deps missing
    try:
        from treequest import ABMCTSM  # type: ignore

        _ = ABMCTSM  # silence linter
    except Exception as e:  # pragma: no cover - skip in environments without deps
        pytest.skip(f"ABMCTSM unavailable: {e}")

    # Instantiate and run with batch_size=1 to avoid parallel execution path
    # Use light config to minimize work during selection
    try:
        algo = ABMCTSM(enable_pruning=False, max_process_workers=1)  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ABMCTSM instantiation failed: {e}")

    state = algo.init_tree()
    actions = ["A", "B"]

    state, trials = algo.ask_batch(state, batch_size=1, actions=actions)
    assert len(trials) == 1
    t = trials[0]
    assert t.action in actions
    state.tree.get_node(t.node_to_expand)

    # tell and verify one node added
    before = len(state.tree.get_state_score_pairs())
    result = _result_from_trial(t.action, t.parent_state)
    state = algo.tell(state, t.trial_id, result)  # type: ignore
    after = len(state.tree.get_state_score_pairs())
    assert after - before == 1


def test_abmctsm_ask_batch_invalid_size():
    # Import ABMCTSM conditionally; skip if optional deps missing
    try:
        from treequest import ABMCTSM  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ABMCTSM unavailable: {e}")

    try:
        algo = ABMCTSM(enable_pruning=False, max_process_workers=1)  # type: ignore
    except Exception as e:  # pragma: no cover
        pytest.skip(f"ABMCTSM instantiation failed: {e}")
    state = algo.init_tree()
    with pytest.raises(ValueError):
        algo.ask_batch(state, batch_size=0, actions=["A"])  # type: ignore
