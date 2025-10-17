import random
from typing import Optional, Tuple

from treequest.algos.best_first_search import BestFirstSearchAlgo
from treequest.algos.standard_mcts import StandardMCTS
from treequest.algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo


def _gen(state: Optional[str]) -> Tuple[str, float]:
    return f"S_from={state}", 0.42


def _tell_and_count(algo, state, trials):
    reflected = 0
    for t in trials:
        before = len(state.tree.get_state_score_pairs())
        parent_state = state.tree.get_node(t.node_to_expand).state
        state = algo.tell(state, t.trial_id, _gen(parent_state))
        after = len(state.tree.get_state_score_pairs())
        if after == before + 1:
            reflected += 1
    return state, reflected


def test_next_nodes_and_invalidation_standard_mcts():
    random.seed(0)
    actions = ["A", "B"]
    algo = StandardMCTS(samples_per_action=2, exploration_weight=1.0)
    state = algo.init_tree()

    # First ask: request full queue across actions (2 per action = 4 total)
    state, trials_first_ask = algo.ask_batch(state, batch_size=4, actions=actions)
    next_nodes = state.trial_store.next_nodes  # type: ignore[attr-defined]
    first_action, second_action = actions[0], actions[1]
    assert len(next_nodes[first_action]) == 2
    assert len(next_nodes[second_action]) == 2
    # First ask contains 2 trials for each action
    a_trials_1 = [t for t in trials_first_ask if t.action == first_action]
    b_trials_1 = [t for t in trials_first_ask if t.action == second_action]
    assert len(a_trials_1) == 2 and len(b_trials_1) == 2

    # Second ask: duplicate some trials; due to ordering it will produce 2 more for first_action
    state, trials_second_ask = algo.ask_batch(state, batch_size=2, actions=actions)
    a_trials_2 = [t for t in trials_second_ask if t.action == first_action]
    assert len(a_trials_2) == 2

    # Tell all first_action trials (4 total): only 2 should reflect, 2 become INVALID
    state, reflected_a = _tell_and_count(algo, state, a_trials_1 + a_trials_2)
    assert reflected_a == 2
    assert len(state.trial_store.next_nodes[first_action]) == 0  # type: ignore

    store = state.trial_store  # type: ignore
    invalid_a = sum(
        1
        for tr in store.finished_trials.values()
        if tr.action == first_action and tr.trial_status == "INVALID"
    )
    assert invalid_a == 2

    # Now tell the second_action trials from the first ask: both should reflect
    state, reflected_b = _tell_and_count(algo, state, b_trials_1)
    assert reflected_b == 2
    assert len(state.trial_store.next_nodes[second_action]) == 0  # type: ignore

    # No INVALID for second_action since we did not over-ask beyond its queue
    invalid_b = sum(
        1
        for tr in store.finished_trials.values()
        if tr.action == second_action and tr.trial_status == "INVALID"
    )
    assert invalid_b == 0

    # After consuming queues, new asks will repopulate for a new selection
    state, trials_new = algo.ask_batch(state, batch_size=2, actions=actions)
    assert len(trials_new) == 2


def test_next_nodes_and_invalidation_bfs():
    random.seed(0)
    actions = ["A", "B"]
    algo = BestFirstSearchAlgo(num_samples=2)
    state = algo.init_tree()

    # First ask the full queue across actions
    state, trials_first_ask = algo.ask_batch(state, batch_size=4, actions=actions)
    next_nodes = state.trial_store.next_nodes  # type: ignore[attr-defined]
    first_action, second_action = actions[0], actions[1]
    assert len(next_nodes[first_action]) == 2
    assert len(next_nodes[second_action]) == 2
    a_trials_1 = [t for t in trials_first_ask if t.action == first_action]
    b_trials_1 = [t for t in trials_first_ask if t.action == second_action]
    assert len(a_trials_1) == 2 and len(b_trials_1) == 2

    # Duplicate for first_action
    state, trials_second_ask = algo.ask_batch(state, batch_size=2, actions=actions)
    a_trials_2 = [t for t in trials_second_ask if t.action == first_action]
    assert len(a_trials_2) == 2

    state, reflected_first = _tell_and_count(algo, state, a_trials_1 + a_trials_2)
    assert reflected_first == 2
    assert len(state.trial_store.next_nodes[first_action]) == 0  # type: ignore

    store = state.trial_store  # type: ignore
    invalid_first = sum(
        1
        for tr in store.finished_trials.values()
        if tr.action == first_action and tr.trial_status == "INVALID"
    )
    assert invalid_first == 2

    # Tell second action trials from first ask
    state, reflected_second = _tell_and_count(algo, state, b_trials_1)
    assert reflected_second == 2
    assert len(state.trial_store.next_nodes[second_action]) == 0  # type: ignore


def test_next_nodes_tot_bfs_root_case():
    random.seed(0)
    actions = ["A", "B"]
    algo = TreeOfThoughtsBFSAlgo(breadth_limit=2, size_limit=2)
    state = algo.init_tree()

    # First ask for many; ToT-BFS queues only one candidate but returns batch_size duplicates
    state, t1 = algo.ask_batch(state, batch_size=5, actions=actions)
    next_nodes = state.trial_store.next_nodes  # type: ignore[attr-defined]
    keys = list(next_nodes.keys())
    assert keys == [actions[0]]
    assert len(next_nodes[actions[0]]) == 1
    assert len(t1) == 5

    # Duplicate ask without tell; still single queued node, but returns duplicates to fill batch
    state, t2 = algo.ask_batch(state, batch_size=3, actions=actions)
    assert len(t2) == 3
    assert len(state.trial_store.next_nodes[actions[0]]) == 1  # type: ignore

    # Tell both trials; only one should reflect; queue becomes empty and duplicates invalidated
    state, reflected = _tell_and_count(algo, state, list(t1) + list(t2))
    assert reflected == 1
    assert state.trial_store.is_queue_empty()  # type: ignore
