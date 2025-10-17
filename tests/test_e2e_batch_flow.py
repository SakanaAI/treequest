import random
from typing import Optional, Tuple

import pytest

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.multi_armed_bandit_ucb import MultiArmedBanditUCBAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo


def _gen_a(state: Optional[str]) -> Tuple[str, float]:
    # Simple deterministic generation with a safe score
    return f"A_from={state}", 0.6


def _gen_b(state: Optional[str]) -> Tuple[str, float]:
    # Slightly different but same score range
    return f"B_from={state}", 0.55


@pytest.mark.parametrize(
    "algo_factory,actions,rounds,batch_size",
    [
        (
            lambda: StandardMCTS(samples_per_action=2, exploration_weight=1.0),
            ["A", "B"],
            3,
            2,
        ),
        (lambda: BestFirstSearchAlgo(num_samples=2), ["A", "B"], 3, 2),
        (
            lambda: TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=3),
            ["A", "B"],
            3,
            2,
        ),
        (lambda: MultiArmedBanditUCBAlgo(exploration_weight=1.0), ["A", "B"], 3, 2),
        (lambda: ABMCTSA(), ["A", "B"], 3, 2),
    ],
)
def test_e2e_batch_flow_light(algo_factory, actions, rounds, batch_size):
    random.seed(0)
    algo = algo_factory()
    state = algo.init_tree()

    generate_fns = {"A": _gen_a, "B": _gen_b}

    total_reflected = 0
    for _ in range(rounds):
        state, trials = algo.ask_batch(state, batch_size=batch_size, actions=actions)

        # Perform tell for all trials returned this round (order no longer matters)
        for t in trials:
            result = generate_fns[t.action](state.tree.get_node(t.node_to_expand).state)
            before = len(state.tree.get_state_score_pairs())
            state = algo.tell(state, t.trial_id, result)
            after = len(state.tree.get_state_score_pairs())
            if after == before + 1:
                total_reflected += 1

        # After processing, ensure no running trials remain this round if the store is visible
        store = getattr(state, "trial_store", None)
        if store is not None:
            running = getattr(store, "running_trials", None)
            if running is not None:
                assert len(running) == 0

    # Verify that exactly reflected nodes (excluding root) were added
    assert len(state.tree.get_state_score_pairs()) == total_reflected
