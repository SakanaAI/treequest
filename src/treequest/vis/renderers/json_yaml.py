"""JSON and YAML output for visualization snapshots."""

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from treequest.vis.errors import (
    DependencyNotFoundError,
    RenderError,
    VisualizationError,
)
from treequest.vis.snapshot import VisualizationSnapshot


def _snapshot_to_dict(snapshot: VisualizationSnapshot) -> Dict[str, Any]:
    """
    Convert a snapshot to a dictionary for serialization.

    Args:
        snapshot: Visualization snapshot

    Returns:
        Dictionary representation
    """
    return {
        "nodes": [dataclasses.asdict(node) for node in snapshot.nodes],
        "edges": [dataclasses.asdict(edge) for edge in snapshot.edges],
        "trials": [dataclasses.asdict(trial) for trial in snapshot.trials],
        "metadata": snapshot.metadata,
    }


def dump_snapshot(
    snapshot: VisualizationSnapshot,
    output_basename: str,
    *,
    format: str,
    include_fields: Optional[List[str]] = None,
    include_algo_metrics: bool = True,
    include_annotations: bool = True,
    indent: int = 2,
) -> None:
    """
    Dump a visualization snapshot to JSON or YAML format.

    Args:
        snapshot: Visualization snapshot to dump
        output_basename: Output file path without extension
        format: Output format ("json" or "yaml")
        include_fields: Optional list of node fields to include.
                       If None, all fields are included.
        include_algo_metrics: Whether to include algorithm metrics
        include_annotations: Whether to include annotations
        indent: Indentation level for output

    Raises:
        DependencyNotFoundError: If YAML support is requested but pyyaml is not installed
        RenderError: If serialization fails
        ValueError: If format is not supported
    """
    # Normalize format
    format = format.lower()

    if format not in ["json", "yaml"]:
        raise VisualizationError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

    # Convert snapshot to dict
    try:
        snapshot_dict = _snapshot_to_dict(snapshot)

        # Filter node fields if requested
        if include_fields is not None:
            filtered_nodes = []
            for node in snapshot_dict["nodes"]:
                filtered_node = {k: v for k, v in node.items() if k in include_fields}
                filtered_nodes.append(filtered_node)
            snapshot_dict["nodes"] = filtered_nodes
        else:
            # Apply include flags
            if not include_algo_metrics:
                for node in snapshot_dict["nodes"]:
                    node.pop("algo_metrics", None)
            if not include_annotations:
                for node in snapshot_dict["nodes"]:
                    node.pop("annotations", None)

    except Exception as e:
        raise RenderError(f"Failed to convert snapshot to dictionary: {e}")

    # Determine output path
    output_path = Path(output_basename)
    if output_path.is_dir():
        # Generate filename with timestamp
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = output_path / f"treequest_{timestamp}.{format}"
    else:
        # Add extension if not present
        if not str(output_path).endswith(f".{format}"):
            output_path = Path(str(output_path) + f".{format}")

    # Serialize
    try:
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(snapshot_dict, f, indent=indent)
        elif format == "yaml":
            try:
                import yaml  # type: ignore
            except ImportError:
                raise DependencyNotFoundError(
                    "pyyaml is not installed. Install it with: pip install treequest[vis-basic]"
                )
            with open(output_path, "w") as f:
                yaml.dump(snapshot_dict, f, indent=indent, sort_keys=False)
    except DependencyNotFoundError:
        raise
    except Exception as e:
        raise RenderError(f"Failed to write {format.upper()} file: {e}")


def snapshot_to_json_string(
    snapshot: VisualizationSnapshot,
    include_fields: Optional[List[str]] = None,
    include_algo_metrics: bool = True,
    include_annotations: bool = True,
    indent: int = 2,
) -> str:
    """
    Convert a snapshot to a JSON string.

    Args:
        snapshot: Visualization snapshot
        include_fields: Optional list of node fields to include
        include_algo_metrics: Whether to include algorithm metrics
        include_annotations: Whether to include annotations
        indent: Indentation level

    Returns:
        JSON string

    Raises:
        RenderError: If serialization fails
    """
    try:
        snapshot_dict = _snapshot_to_dict(snapshot)

        # Filter node fields if requested
        if include_fields is not None:
            filtered_nodes = []
            for node in snapshot_dict["nodes"]:
                filtered_node = {k: v for k, v in node.items() if k in include_fields}
                filtered_nodes.append(filtered_node)
            snapshot_dict["nodes"] = filtered_nodes
        else:
            # Apply include flags
            if not include_algo_metrics:
                for node in snapshot_dict["nodes"]:
                    node.pop("algo_metrics", None)
            if not include_annotations:
                for node in snapshot_dict["nodes"]:
                    node.pop("annotations", None)

        return json.dumps(snapshot_dict, indent=indent)
    except Exception as e:
        raise RenderError(f"Failed to convert snapshot to JSON string: {e}")
