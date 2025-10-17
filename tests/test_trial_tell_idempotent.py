import random
from typing import Optional, Tuple

import pytest

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.multi_armed_bandit_ucb import MultiArmedBanditUCBAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo


def _gen(state: Optional[str], val: float) -> Tuple[str, float]:
    return f"S_from={state}", val


@pytest.mark.parametrize(
    "algo_factory",
    [
        lambda: StandardMCTS(samples_per_action=1, exploration_weight=1.0),
        lambda: BestFirstSearchAlgo(num_samples=1),
        lambda: TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=2),
        lambda: MultiArmedBanditUCBAlgo(exploration_weight=1.0),
        lambda: ABMCTSA(),
    ],
)
def test_tell_idempotent_same_trial_all_algos(algo_factory):
    random.seed(0)
    algo = algo_factory()
    state = algo.init_tree()
    actions = ["A"]

    # Ask a single trial and tell twice with different scores
    state, trials = algo.ask_batch(state, batch_size=1, actions=actions)
    t = trials[0]

    before = len(state.tree.get_state_score_pairs())
    state = algo.tell(
        state, t.trial_id, _gen(state.tree.get_node(t.node_to_expand).state, 0.7)
    )
    mid = len(state.tree.get_state_score_pairs())
    assert mid == before + 1

    # Second tell on the same trial_id should be ignored (no additional node)
    state = algo.tell(
        state, t.trial_id, _gen(state.tree.get_node(t.node_to_expand).state, 0.2)
    )
    after = len(state.tree.get_state_score_pairs())
    assert after == mid


def test_tell_idempotent_same_trial_abmctsm():
    try:
        from treequest import ABMCTSM  # type: ignore
    except Exception as e:
        pytest.skip(f"ABMCTSM unavailable: {e}")

    random.seed(0)
    algo = ABMCTSM(enable_pruning=False, max_process_workers=1)  # type: ignore
    state = algo.init_tree()
    actions = ["A"]

    state, trials = algo.ask_batch(state, batch_size=1, actions=actions)
    t = trials[0]

    before = len(state.tree.get_state_score_pairs())
    state = algo.tell(
        state, t.trial_id, _gen(state.tree.get_node(t.node_to_expand).state, 0.7)
    )  # type: ignore
    mid = len(state.tree.get_state_score_pairs())
    assert mid == before + 1

    state = algo.tell(
        state, t.trial_id, _gen(state.tree.get_node(t.node_to_expand).state, 0.2)
    )  # type: ignore
    after = len(state.tree.get_state_score_pairs())
    assert after == mid
