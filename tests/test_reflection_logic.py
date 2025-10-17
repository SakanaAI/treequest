import random
from typing import Optional, Tuple

import pytest

from treequest.algos.ab_mcts_a.algo import ABMCTSA
from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.multi_armed_bandit_ucb import MultiArmedBanditUCBAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo


def _gen(state: Optional[str]) -> Tuple[str, float]:
    return f"S_from={state}", 0.5


def _reflect_many(algo, state, trials, gen):
    reflected = 0
    for t in trials:
        before = len(state.tree.get_state_score_pairs())
        state = algo.tell(
            state, t.trial_id, gen(state.tree.get_node(t.node_to_expand).state)
        )
        after = len(state.tree.get_state_score_pairs())
        if after == before + 1:
            reflected += 1
    return state, reflected


@pytest.mark.parametrize(
    "algo_factory,actions",
    [
        (
            lambda: StandardMCTS(samples_per_action=2, exploration_weight=1.0),
            ["A", "B"],
        ),
        (lambda: BestFirstSearchAlgo(num_samples=2), ["A", "B"]),
    ],
)
def test_reflection_logic_queue_algorithms(algo_factory, actions):
    random.seed(0)
    algo = algo_factory()
    state = algo.init_tree()

    # First ask: request a large batch; the queue will still contain the real candidates
    state, t1 = algo.ask_batch(state, batch_size=100, actions=actions)
    # The number of true reflections equals queued candidates, not number of trials returned
    store = getattr(state, "trial_store", None)
    assert store is not None
    queued_total = sum(len(nodes) for nodes in store.next_nodes.values())  # type: ignore[attr-defined]
    assert queued_total > 0
    expected_reflections = queued_total

    # Second ask before any tell to create duplicate trials referencing same queued nodes
    state, t2 = algo.ask_batch(state, batch_size=expected_reflections, actions=actions)
    assert len(t2) > 0

    all_trials = list(t1) + list(t2)
    random.shuffle(all_trials)

    before_nodes = len(state.tree.get_state_score_pairs())
    state, reflected = _reflect_many(algo, state, all_trials, _gen)
    after_nodes = len(state.tree.get_state_score_pairs())

    # Only the initially queued items should reflect exactly once
    assert reflected == expected_reflections
    assert after_nodes - before_nodes == expected_reflections

    # Remaining trials should have been invalidated
    store = getattr(state, "trial_store", None)
    assert store is not None
    invalid = sum(
        1 for tr in store.finished_trials.values() if tr.trial_status == "INVALID"
    )
    assert invalid == len(all_trials) - expected_reflections
    assert len(store.running_trials) == 0


def test_reflection_logic_tot_bfs():
    random.seed(0)
    algo = TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=2)
    state = algo.init_tree()
    actions = ["A", "B"]

    # First ask returns exactly batch_size trials (duplicates of the single queued root expansion)
    state, t1 = algo.ask_batch(state, batch_size=10, actions=actions)
    assert len(t1) == 10
    # Create more duplicates by asking again without telling
    state, t2 = algo.ask_batch(state, batch_size=3, actions=actions)
    assert len(t2) == 3
    all_trials = list(t1) + list(t2)
    random.shuffle(all_trials)

    before_nodes = len(state.tree.get_state_score_pairs())
    state, reflected = _reflect_many(algo, state, all_trials, _gen)
    after_nodes = len(state.tree.get_state_score_pairs())

    assert reflected == 1
    assert after_nodes - before_nodes == 1

    store = getattr(state, "trial_store", None)
    assert store is not None
    invalid = sum(
        1 for tr in store.finished_trials.values() if tr.trial_status == "INVALID"
    )
    assert invalid == len(all_trials) - 1
    assert len(store.running_trials) == 0


@pytest.mark.parametrize(
    "algo_factory,actions",
    [
        (lambda: MultiArmedBanditUCBAlgo(exploration_weight=1.0), ["A", "B"]),
        (lambda: ABMCTSA(), ["A", "B"]),
    ],
)
def test_reflection_logic_nonqueue_algorithms(algo_factory, actions):
    random.seed(0)
    algo = algo_factory()
    state = algo.init_tree()

    # Ask twice before telling
    state, t1 = algo.ask_batch(state, batch_size=3, actions=actions)
    state, t2 = algo.ask_batch(state, batch_size=2, actions=actions)
    all_trials = list(t1) + list(t2)
    random.shuffle(all_trials)

    before_nodes = len(state.tree.get_state_score_pairs())
    state, reflected = _reflect_many(algo, state, all_trials, _gen)
    after_nodes = len(state.tree.get_state_score_pairs())

    # No invalidation; all tells reflect exactly once
    assert reflected == len(all_trials)
    assert after_nodes - before_nodes == len(all_trials)

    store = getattr(state, "trial_store", None)
    assert store is not None
    invalid = sum(
        1 for tr in store.finished_trials.values() if tr.trial_status == "INVALID"
    )
    assert invalid == 0
    assert len(store.running_trials) == 0


def test_reflection_logic_abmctsm_nonqueue():
    try:
        from treequest import ABMCTSM  # type: ignore
    except Exception as e:
        pytest.skip(f"ABMCTSM unavailable: {e}")

    algo = ABMCTSM(enable_pruning=False, max_process_workers=1)  # type: ignore
    state = algo.init_tree()
    actions = ["A", "B"]

    # Ask twice before telling
    state, t1 = algo.ask_batch(state, batch_size=1, actions=actions)
    state, t2 = algo.ask_batch(state, batch_size=1, actions=actions)
    all_trials = list(t1) + list(t2)
    random.shuffle(all_trials)

    before_nodes = len(state.tree.get_state_score_pairs())
    state, reflected = _reflect_many(algo, state, all_trials, _gen)  # type: ignore
    after_nodes = len(state.tree.get_state_score_pairs())

    assert reflected == len(all_trials)
    assert after_nodes - before_nodes == len(all_trials)
