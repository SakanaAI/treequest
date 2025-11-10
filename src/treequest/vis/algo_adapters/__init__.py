"""Visualization adapters for tree search algorithms."""

import warnings
from typing import Dict, Optional, TypeVar

from treequest.algos.ab_mcts_m._ab_mcts_m_imports import _import as _ab_mcts_m_import
from treequest.vis.algo_adapters.ab_mcts_a import ABMCTSAAdapter
from treequest.vis.algo_adapters.base import VisualizerAdapter
from treequest.vis.algo_adapters.best_first_search import BestFirstSearchAdapter
from treequest.vis.algo_adapters.multi_armed_bandit_ucb import (
    MultiArmedBanditUCBAdapter,
)
from treequest.vis.algo_adapters.standard_mcts import StandardMCTSAdapter
from treequest.vis.algo_adapters.tree_of_thoughts_bfs import TreeOfThoughtsBFSAdapter

StateT = TypeVar("StateT")

_ADAPTER_REGISTRY: Dict[str, VisualizerAdapter] = {
    "ABMCTSAAlgoState": ABMCTSAAdapter(),
    "BFSState": BestFirstSearchAdapter(),
    "MCTSState": StandardMCTSAdapter(),
    "ToTBFSState": TreeOfThoughtsBFSAdapter(),
    "UCBState": MultiArmedBanditUCBAdapter(),
}

if _ab_mcts_m_import.is_successful():
    from treequest.vis.algo_adapters.ab_mcts_m import ABMCTSMAdapter

    _ADAPTER_REGISTRY["ABMCTSMState"] = ABMCTSMAdapter()


def register_adapter(state_type_name: str, adapter: VisualizerAdapter) -> None:
    """
    Register an adapter for a given algorithm state type.

    Args:
        state_type_name: Name of the state type (e.g., "MCTSState")
        adapter: Adapter instance
    """
    if state_type_name in _ADAPTER_REGISTRY:
        warnings.warn(
            f"Adapter for state type '{state_type_name}' is already registered. Overwriting."
        )
    _ADAPTER_REGISTRY[state_type_name] = adapter


def get_adapter(algo_state: StateT) -> Optional[VisualizerAdapter]:
    """
    Get the appropriate adapter for a given algorithm state.

    Args:
        algo_state: The algorithm state

    Returns:
        Adapter instance or None if no adapter is registered
    """
    state_type_name = type(algo_state).__name__

    # Check registry first
    if state_type_name in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[state_type_name]

    # No adapter found - return a default that returns empty metrics
    warnings.warn(
        f"No adapter found for state type '{state_type_name}'. Visualization may be limited.\n"
        "Consider implementing VisualizerAdapter protocol for this state type and registering it via register_adapter()."
    )
    return None
