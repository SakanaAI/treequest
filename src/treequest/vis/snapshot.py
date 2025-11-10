"""Data structures for visualization snapshots."""

import dataclasses
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata as importlib_metadata
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from treequest.types import NodeId, TrialId

StateT = TypeVar("StateT")


@dataclasses.dataclass
class AlgoMetrics:
    """Algorithm-specific metrics for a node."""

    display_name: str
    display_value: str


@dataclasses.dataclass
class NodeSnapshot:
    """Snapshot of a single node in the tree."""

    # Node identification
    id: NodeId  # expand_idx
    trial_id: Optional[TrialId]
    parent_id: Optional[NodeId]

    # Node properties
    depth: int
    score: float
    state_repr: str

    # Trial information (joined from TrialStore)
    action_from_parent: Optional[str]
    created_at: Optional[str]
    completed_at: Optional[str]
    status: Optional[str]  # "RUNNING", "COMPLETE", "INVALID", "ROOT", etc.

    # Extensible data
    annotations: Dict[str, Any] = dataclasses.field(default_factory=dict)
    algo_metrics: Dict[str, Union[bool, int, float, str, AlgoMetrics]] = (
        dataclasses.field(default_factory=dict)
    )


@dataclasses.dataclass
class EdgeSnapshot:
    """Snapshot of an edge in the tree."""

    source: NodeId
    target: NodeId
    action: Optional[str] = None


@dataclasses.dataclass
class TrialSnapshot:
    """Snapshot of a trial."""

    trial_id: TrialId
    node_to_expand: NodeId
    action: str
    score: Optional[float]
    created_at: str
    completed_at: Optional[str]
    trial_status: str


@dataclasses.dataclass
class VisualizationSnapshot(Generic[StateT]):
    """Complete snapshot of tree state for visualization."""

    nodes: List[NodeSnapshot]
    edges: List[EdgeSnapshot]
    trials: List[TrialSnapshot]
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate snapshot after initialization."""
        if not self.nodes:
            raise ValueError("Snapshot must contain at least one node (root)")

        # Verify root exists (id = -1)
        if not any(node.id == -1 for node in self.nodes):
            raise ValueError("Snapshot must contain a root node (id=-1)")

    @classmethod
    def create_with_metadata(
        cls,
        nodes: List[NodeSnapshot],
        edges: List[EdgeSnapshot],
        trials: List[TrialSnapshot],
        algorithm_name: str = "unknown",
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> "VisualizationSnapshot[StateT]":
        """Create a snapshot with auto-generated metadata."""
        non_root_count = sum(1 for node in nodes if node.id != -1)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": algorithm_name,
            "treequest_version": _get_treequest_version(),
            "num_nodes": non_root_count,
            "num_edges": len(edges),
            "num_trials": len(trials),
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        return cls(nodes=nodes, edges=edges, trials=trials, metadata=metadata)


@lru_cache(maxsize=1)
def _get_treequest_version() -> str:
    """Return the installed TreeQuest version or 'unknown'."""
    try:
        return importlib_metadata.version("treequest")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"
