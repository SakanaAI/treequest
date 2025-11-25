"""Functions for building visualization snapshots from algorithm states."""

import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from treequest.algos.tree import Tree
from treequest.trial import Trial, TrialStore, TrialStoreWithNodeQueue
from treequest.vis.algo_adapters import get_adapter
from treequest.vis.errors import InvalidStateError
from treequest.vis.snapshot import (
    EdgeSnapshot,
    NodeSnapshot,
    TrialSnapshot,
    VisualizationSnapshot,
)

StateT = TypeVar("StateT")
AlgoStateT = TypeVar("AlgoStateT")

try:  # Optional dependency – used only if available
    from pydantic import BaseModel as PydanticBaseModel  # type: ignore
except Exception:  # pragma: no cover - pydantic is optional
    PydanticBaseModel = None


def _default_state_formatter(state: Any) -> str:
    """Default formatter for node states.

    - If the state is a pydantic BaseModel, prefer JSON.
    - If the state is a dataclass instance, serialize via asdict() to JSON.
    - Fallback to repr()/str() otherwise.

    """
    # NOTE: We use separators=(",", ":") workaround to get the consistent json string representation.
    # See https://github.com/pydantic/pydantic/issues/6606

    # Pydantic BaseModel → JSON
    if PydanticBaseModel is not None and isinstance(state, PydanticBaseModel):
        try:
            if hasattr(state, "model_dump_json"):
                # Pydantic v2 preferred API
                return state.model_dump_json()
            if hasattr(state, "model_dump"):
                # Pydantic v2 Python object → JSON string
                return json.dumps(
                    state.model_dump(), default=str, separators=(",", ":")
                )
            if hasattr(state, "json"):
                # Pydantic v1 API
                return state.json(separators=(",", ":"))
        except Exception:
            # Fall through to generic formatting
            pass

    # Dataclass instance → JSON
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        try:
            return json.dumps(
                dataclasses.asdict(state), default=str, separators=(",", ":")
            )
        except Exception:
            # Fall through to generic formatting
            pass

    # Generic fallback
    try:
        return repr(state)
    except Exception:
        try:
            return str(state)
        except Exception:
            return "<unrepresentable state>"


def build_snapshot(
    search_tree: AlgoStateT,
    state_formatter: Optional[Callable[[StateT], str]] = None,
    annotations: Optional[Dict[str, Any]] = None,
) -> VisualizationSnapshot:
    """
    Build a visualization snapshot from an algorithm state.

    Args:
        search_tree: Search tree state (return value of algo.init_tree, algo.step, algo.ask, or algo.tell).
        state_formatter: Optional function to format node states. Defaults to repr().
        annotations: Optional global annotations to add to metadata.

    Returns:
        VisualizationSnapshot ready for rendering

    Raises:
        InvalidStateError: If the state is invalid or missing required attributes
    """
    # Validate state has required attributes
    if not hasattr(search_tree, "tree"):
        raise InvalidStateError(
            f"State must have a 'tree' attribute, got {type(search_tree)}"
        )

    tree: Tree = search_tree.tree

    # Get trial store if available
    trial_store: Optional[Union[TrialStore, TrialStoreWithNodeQueue]] = None
    finished_trials: Optional[Dict[str, Trial]] = None
    running_trials: Optional[Dict[str, Trial]] = None

    if hasattr(search_tree, "trial_store"):
        trial_store = search_tree.trial_store
        finished_trials = getattr(trial_store, "finished_trials", {}) or {}
        running_trials = getattr(trial_store, "running_trials", {}) or {}

    # Get adapter for this algorithm
    adapter = get_adapter(search_tree)
    algorithm_name = adapter.get_algorithm_name(search_tree) if adapter else "Unknown"

    # Default state formatter
    if state_formatter is None:
        state_formatter = _default_state_formatter

    # Build node snapshots
    nodes = tree.get_nodes()
    node_snapshots: List[NodeSnapshot] = []
    edges: List[EdgeSnapshot] = []

    for node in nodes:
        # Get trial information if available
        trial_id = node.trial_id
        trial: Optional[Trial] = None
        action_from_parent: Optional[str] = None
        created_at: Optional[str] = None
        completed_at: Optional[str] = None
        status: Optional[str] = "ROOT" if node.is_root() else None

        if trial_id and trial_store:
            # Try to find trial in finished trials
            if finished_trials and trial_id in finished_trials:
                trial = finished_trials[trial_id]
                action_from_parent = trial.action
                created_at = trial.created_at
                completed_at = trial.completed_at
                status = trial.trial_status
            # Try to find in running trials
            elif running_trials and trial_id in running_trials:
                trial = running_trials[trial_id]
                action_from_parent = trial.action
                created_at = trial.created_at
                status = "RUNNING"

        # Fill in default status for completed nodes without trial record
        if status is None:
            if trial_id is not None:
                # Node exists but trial record cleaned up – treat as completed.
                status = "COMPLETE"
            elif node.is_root():
                status = "ROOT"

        # Get state representation
        if node.is_root():
            state_repr = "ROOT"
        else:
            node_state = node.state
            if node_state is None:
                raise InvalidStateError(
                    f"Non-root node (ID: {node.expand_idx}) must have an associated state."
                )
            try:
                state_repr = state_formatter(node_state)
            except Exception:
                state_repr = str(node_state)

        # Extract algorithm-specific metrics
        algo_metrics: Dict[str, Any] = {}
        if adapter:
            try:
                algo_metrics = adapter.extract_node_metrics(search_tree, node)
            except Exception:
                # Ignore errors in metric extraction
                pass

        # Create node snapshot
        node_snapshot = NodeSnapshot(
            id=node.expand_idx,
            trial_id=trial_id,
            parent_id=node.parent.expand_idx if node.parent else None,
            depth=node.depth,
            score=node.score,
            state_repr=state_repr,
            action_from_parent=action_from_parent,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            annotations={},
            algo_metrics=algo_metrics,
        )
        node_snapshots.append(node_snapshot)

        # Create edge snapshot
        if node.parent:
            edge = EdgeSnapshot(
                source=node.parent.expand_idx,
                target=node.expand_idx,
                action=action_from_parent,
            )
            edges.append(edge)

    # Build trial snapshots
    trial_snapshots: List[TrialSnapshot] = []
    if trial_store and finished_trials is not None and running_trials is not None:
        # Add finished trials
        for trial in finished_trials.values():
            trial_snapshot = TrialSnapshot(
                trial_id=trial.trial_id,
                node_to_expand=trial.node_to_expand,
                action=trial.action,
                score=trial.score,
                created_at=trial.created_at,
                completed_at=trial.completed_at,
                trial_status=trial.trial_status,
            )
            trial_snapshots.append(trial_snapshot)

        # Add running trials
        for trial in running_trials.values():
            trial_snapshot = TrialSnapshot(
                trial_id=trial.trial_id,
                node_to_expand=trial.node_to_expand,
                action=trial.action,
                score=None,
                created_at=trial.created_at,
                completed_at=None,
                trial_status="RUNNING",
            )
            trial_snapshots.append(trial_snapshot)

    # Create snapshot with metadata
    additional_metadata = annotations or {}

    snapshot: VisualizationSnapshot[StateT] = (
        VisualizationSnapshot.create_with_metadata(
            nodes=node_snapshots,
            edges=edges,
            trials=trial_snapshots,
            algorithm_name=algorithm_name,
            additional_metadata=additional_metadata,
        )
    )

    return snapshot
