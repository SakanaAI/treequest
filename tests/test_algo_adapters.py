"""Tests for visualization algorithm adapters."""

from typing import Dict

from treequest.algos.ab_mcts_a.algo import ABMCTSA, ABMCTSAAlgoState
from treequest.algos.ab_mcts_m.algo import ABMCTSMState
from treequest.algos.ab_mcts_m.pymc_interface import Observation
from treequest.algos.best_first_search import BFSHeapItem, BFSState
from treequest.algos.multi_armed_bandit_ucb import UCBState
from treequest.algos.standard_mcts import MCTSState
from treequest.algos.tree import Tree
from treequest.algos.tree_of_thought_bfs import ToTBFSState
from treequest.vis.algo_adapters.ab_mcts_a import ABMCTSAAdapter
from treequest.vis.algo_adapters.ab_mcts_m import ABMCTSMAdapter
from treequest.vis.algo_adapters.best_first_search import BestFirstSearchAdapter
from treequest.vis.algo_adapters.multi_armed_bandit_ucb import (
    MultiArmedBanditUCBAdapter,
)
from treequest.vis.algo_adapters.standard_mcts import StandardMCTSAdapter
from treequest.vis.algo_adapters.tree_of_thoughts_bfs import (
    TreeOfThoughtsBFSAdapter,
)
from treequest.types import GenerateFnType

StateType = str


def test_ab_mcts_a_adapter_reports_action_probas() -> None:
    algo: ABMCTSA[StateType] = ABMCTSA()
    state: ABMCTSAAlgoState[StateType] = algo.init_tree()

    generate_fns: Dict[str, GenerateFnType[StateType]] = {
        "explore": lambda parent: ("explore", 0.8),
        "exploit": lambda parent: ("exploit", 0.3),
    }
    for _ in range(4):
        state = algo.step(state, generate_fns)

    adapter = ABMCTSAAdapter()
    root_metrics = adapter.extract_node_metrics(state, state.tree.root)
    assert "action_probas" in root_metrics
    assert "display_value" in root_metrics["action_probas"]
    assert isinstance(root_metrics["action_probas"]["display_value"], str)


def test_ab_mcts_m_adapter_aggregates_descendant_observations() -> None:
    tree: Tree[StateType] = Tree.with_root_node()
    child = tree.add_node(("s", 0.6), tree.root)
    grandchild = tree.add_node(("s2", 0.4), child)

    state: ABMCTSMState[StateType] = ABMCTSMState(tree=tree)
    state.all_observations[child.expand_idx] = Observation(
        reward=0.6, action="grow", node_expand_idx=child.expand_idx
    )
    state.all_observations[grandchild.expand_idx] = Observation(
        reward=0.4, action="refine", node_expand_idx=grandchild.expand_idx
    )

    adapter = ABMCTSMAdapter()
    metrics = adapter.extract_node_metrics(state, child)
    # Adapter reports human-readable lists and flags under display_value
    assert "rewards_by_action" in metrics
    assert "rewards_by_child" in metrics
    assert metrics["rewards_by_action"]["display_value"].startswith("<ul>")


def test_multi_armed_bandit_adapter_reports_global_stats() -> None:
    tree: Tree[StateType] = Tree.with_root_node()
    state: UCBState[StateType] = UCBState(tree=tree)
    state.scores_by_action["a"] = [0.4, 0.6]
    state.scores_by_action["b"] = [0.2]

    adapter = MultiArmedBanditUCBAdapter()
    metrics = adapter.extract_node_metrics(state, tree.root)
    assert "total_len" in metrics
    assert metrics["total_len"]["display_value"].isdigit()
    assert "action_stats" in metrics
    assert metrics["action_stats"]["display_value"].startswith("<ul>")


def test_tree_of_thoughts_adapter_currently_no_specific_metrics() -> None:
    tree: Tree[str] = Tree.with_root_node()
    state: ToTBFSState[str] = ToTBFSState(tree=tree)
    adapter = TreeOfThoughtsBFSAdapter()
    metrics = adapter.extract_node_metrics(state, state.tree.root)
    assert metrics == {}


def test_best_first_search_adapter_reports_queue_details() -> None:
    tree: Tree[StateType] = Tree.with_root_node()
    child = tree.add_node(("s", 0.7), tree.root)
    state: BFSState[StateType] = BFSState(tree=tree)
    state.leaves.append(BFSHeapItem(node=child, score=0.7))
    state.trial_store.fill_nodes_queue([(child, "expand")])

    adapter = BestFirstSearchAdapter()
    metrics = adapter.extract_node_metrics(state, child)
    assert metrics["is_leaf"]["display_value"] in {"Yes", "No"}
    # If the leaf is enqueued it should have rank/score values
    assert "leaf_rank" in metrics and "display_value" in metrics["leaf_rank"]
    assert "leaf_score" in metrics and "display_value" in metrics["leaf_score"]


def test_standard_mcts_adapter_defaults_and_queue() -> None:
    tree: Tree[StateType] = Tree.with_root_node()
    child = tree.add_node(("s", 0.5), tree.root)
    state: MCTSState[StateType] = MCTSState(tree=tree)
    state.visit_counts[child.expand_idx] = 2
    state.value_sums[child.expand_idx] = 1.0
    state.priors[child.expand_idx] = 0.25
    state.trial_store.fill_nodes_queue([(child, "expand")])

    adapter = StandardMCTSAdapter()
    metrics = adapter.extract_node_metrics(state, child)
    assert metrics["visits"]["display_value"] == "2"
    assert (
        metrics["mean"]["display_value"].startswith("0.")
        or metrics["mean"]["display_value"] == "N/A"
    )
    assert "prior" in metrics and "uct_score" in metrics
