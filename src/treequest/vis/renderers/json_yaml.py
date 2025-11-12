"""JSON and YAML output for visualization snapshots."""

import dataclasses
import json
from typing import Any, Dict, List, Optional

import yaml

from treequest.vis.errors import RenderError
from treequest.vis.snapshot import VisualizationSnapshot


def snapshot_to_dict(
    snapshot: VisualizationSnapshot,
    include_fields: Optional[List[str]] = None,
    include_algo_metrics: bool = True,
    include_annotations: bool = True,
) -> Dict[str, Any]:
    """
    Convert a snapshot to a dictionary for serialization.

    Args:
        snapshot: Visualization snapshot
        include_fields: Optional list of node fields to include
        include_algo_metrics: Whether to include algorithm metrics
        include_annotations: Whether to include annotations

    Returns:
        Dictionary representation of the snapshot
    """
    try:
        # Filter node fields if requested
        filtered_nodes = []
        for node in snapshot.nodes:
            node_dict = dataclasses.asdict(node)
            if include_fields is not None:
                node_dict = {k: v for k, v in node_dict.items() if k in include_fields}
            if not include_algo_metrics:
                node_dict.pop("algo_metrics", None)
            if not include_annotations:
                node_dict.pop("annotations", None)
            filtered_nodes.append(node_dict)

        return {
            "nodes": filtered_nodes,
            "edges": [dataclasses.asdict(edge) for edge in snapshot.edges],
            "trials": [dataclasses.asdict(trial) for trial in snapshot.trials],
            "metadata": snapshot.metadata,
        }
    except Exception as e:
        raise RenderError(f"Failed to convert snapshot to dictionary: {e}")


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
        RenderError: If serialization fails
        ValueError: If format is not supported
    """
    # Normalize format
    format = format.lower()
    if format not in ["json", "yaml"]:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

    # Convert snapshot to dict
    try:
        snapshot_dict = snapshot_to_dict(
            snapshot,
            include_fields=include_fields,
            include_algo_metrics=include_algo_metrics,
            include_annotations=include_annotations,
        )
    except Exception as e:
        raise RenderError(f"Failed to convert snapshot to dictionary: {e}")

    # Serialize
    output_path = f"{output_basename}.{format}"
    try:
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(snapshot_dict, f, indent=indent)
        elif format == "yaml":
            with open(output_path, "w") as f:
                yaml.dump(snapshot_dict, f, indent=indent, sort_keys=False)
    except Exception as e:
        raise RenderError(f"Failed to write {format.upper()} file: {e}")
