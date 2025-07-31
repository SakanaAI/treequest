from .algos.ab_mcts_m._ab_mcts_m_imports import _import as _ab_mcts_m_import

if _ab_mcts_m_import.is_successful():
    from .algos.ab_mcts_m.algo import ABMCTSM
else:
    # Create a placeholder that raises an informative error when accessed
    class _ABMCTSMPlaceholder:
        def __getattr__(self, name):
            _ab_mcts_m_import.check()
            raise ImportError("ABMCTSM import failed.")

        def __call__(self, *args, **kwargs):
            _ab_mcts_m_import.check()
            raise ImportError("ABMCTSM import failed.")

    ABMCTSM = _ABMCTSMPlaceholder()

from .algos.ab_mcts_a.algo import ABMCTSA
from .algos.base import Algorithm
from .algos.standard_mcts import StandardMCTS
from .algos.tree_of_thought_bfs import TreeOfThoughtsBFSAlgo
from .ranker import top_k

__all__ = [
    "StandardMCTS",
    "top_k",
    "TreeOfThoughtsBFSAlgo",
    "ABMCTSA",
    "ABMCTSM",
    "Algorithm",
]
