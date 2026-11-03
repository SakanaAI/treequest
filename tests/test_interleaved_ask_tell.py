import random
from typing import Optional, Tuple

import numpy as np
import pytest

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.multi_armed_bandit_ucb import MultiArmedBanditUCBAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo


def _gen_a(state: Optional[str]) -> Tuple[str, float]:
    return f"A_from={state}", 0.61


def _gen_b(state: Optional[str]) -> Tuple[str, float]:
    return f"B_from={state}", 0.57


@pytest.mark.parametrize(
    "algo_factory,actions,first_batch,second_batch",
    [
        (
            lambda: StandardMCTS(samples_per_action=2, exploration_weight=1.0),
            ["A", "B"],
            3,
            3,
        ),
        (lambda: BestFirstSearchAlgo(num_samples=2), ["A", "B"], 3, 3),
        (
            lambda: TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=2),
            ["A", "B"],
            2,
            2,
        ),
        (lambda: MultiArmedBanditUCBAlgo(exploration_weight=1.0), ["A", "B"], 3, 2),
        (lambda: ABMCTSA(), ["A", "B"], 3, 2),
    ],
)
def test_interleaved_ask_tell_order_independent(
    algo_factory, actions, first_batch, second_batch
):
    """
    Verify ask and tell can interleave and tells can be out of order as long as trial_id matches.
    This primarily targets algorithms that rely on TrialStoreWithNodeQueue.
    """
    random.seed(1)
    np.random.seed(1)
    algo = algo_factory()
    state = algo.init_tree()

    gen = {"A": _gen_a, "B": _gen_b}

    # Ask twice before telling anything
    state, trials1 = algo.ask_batch(state, batch_size=first_batch, actions=actions)
    state, trials2 = algo.ask_batch(state, batch_size=second_batch, actions=actions)
    all_trials = list(trials1) + list(trials2)

    # Shuffle to break any LIFO/FIFO assumptions and mix across batches/actions
    random.shuffle(all_trials)

    # Tell in shuffled order, count successful reflections
    reflected = 0
    for t in all_trials:
        parent_state = state.tree.get_node(t.node_to_expand).state
        before = len(state.tree.get_state_score_pairs())
        state = algo.tell(state, t.trial_id, gen[t.action](parent_state))
        after = len(state.tree.get_state_score_pairs())
        if after == before + 1:
            reflected += 1

    # All trial_ids we asked should now be finished (COMPLETE or INVALID)
    store = getattr(state, "trial_store", None)
    assert store is not None
    assert all(t.trial_id in store.finished_trials for t in all_trials)
    # No running trials should remain
    assert len(store.running_trials) == 0

    # Tree should have grown exactly by the number of reflected tells
    assert len(state.tree.get_state_score_pairs()) == reflected

    # Additional ask still returns the requested number of trials
    state, trials3 = algo.ask_batch(state, batch_size=2, actions=actions)
    assert len(trials3) == 2


def test_interleaved_ask_tell_abmctsm():
    try:
        from treequest import ABMCTSM  # type: ignore
    except Exception as e:
        pytest.skip(f"ABMCTSM unavailable: {e}")

    # Keep batch_size=1 per ask to avoid parallel path
    algo = ABMCTSM(enable_pruning=False, max_process_workers=1)  # type: ignore
    state = algo.init_tree()
    actions = ["A", "B"]
    gen = {"A": _gen_a, "B": _gen_b}

    # Ask twice without telling
    state, trials1 = algo.ask_batch(state, batch_size=1, actions=actions)
    state, trials2 = algo.ask_batch(state, batch_size=1, actions=actions)
    all_trials = list(trials1) + list(trials2)
    random.shuffle(all_trials)

    reflected = 0
    for t in all_trials:
        before = len(state.tree.get_state_score_pairs())
        state = algo.tell(
            state,
            t.trial_id,
            gen[t.action](state.tree.get_node(t.node_to_expand).state),
        )  # type: ignore
        after = len(state.tree.get_state_score_pairs())
        if after == before + 1:
            reflected += 1

    store = getattr(state, "trial_store", None)
    assert store is not None
    assert all(t.trial_id in store.finished_trials for t in all_trials)
    assert len(store.running_trials) == 0

    state, trials3 = algo.ask_batch(state, batch_size=1, actions=actions)
    assert len(trials3) == 1


def test_tot_bfs_extra_tells_past_root_are_discarded():
    """
    Below the root, ToT-BFS queues several nodes per action. Parallel asks can hand out
    more trials for the first queued node than it has expansions; the extra tells must be
    discarded like any other over-ask instead of failing in advance_queue.
    """
    algo = TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=2)
    state = algo.init_tree()
    actions = ["A"]

    # Grow the tree until the deepest level holds two nodes to expand.
    for _ in range(3):
        state, trial = algo.ask(state, actions)
        state = algo.tell(state, trial.trial_id, _gen_a(None))

    # Three workers ask before any of them tells.
    trials = []
    for _ in range(3):
        state, trial = algo.ask(state, actions)
        trials.append(trial)
    assert len({t.node_to_expand for t in trials}) == 1

    reflected = 0
    for t in trials:
        before = len(state.tree.get_state_score_pairs())
        parent_state = state.tree.get_node(t.node_to_expand).state
        state = algo.tell(state, t.trial_id, _gen_a(parent_state))
        if len(state.tree.get_state_score_pairs()) == before + 1:
            reflected += 1

    # The node had two queued expansions, so only two of the three tells count.
    assert reflected == 2
    store = state.trial_store
    assert [store.finished_trials[t.trial_id].trial_status for t in trials].count(
        "INVALID"
    ) == 1
    assert len(store.running_trials) == 0

    # The other queued node is still expanded afterwards.
    state, next_trials = algo.ask_batch(state, batch_size=1, actions=actions)
    assert next_trials[0].node_to_expand != trials[0].node_to_expand
