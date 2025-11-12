"""Graphviz renderer for tree visualization."""

from typing import Callable, Optional, Union

from treequest.vis.errors import DependencyNotFoundError, RenderError
from treequest.vis.snapshot import VisualizationSnapshot
from treequest.vis.renderers.color_utils import (
    ROOT_COLOR,
    ColorMap,
    apply_status_color,
    resolve_colormap,
)


def render_graphviz(
    snapshot: VisualizationSnapshot,
    output_basename: str,
    *,
    format: str,
    title: Optional[str] = None,
    show_scores: bool = True,
    max_label_length: int = 20,
    color_map: Optional[Union[str, ColorMap, Callable[[float], str]]] = None,
) -> None:
    """
    Render a visualization snapshot using Graphviz.

    Args:
        snapshot: Visualization snapshot to render
        output_basename: Output file path without extension
        format: Output format (e.g., "pdf", "png", "svg")
        title: Optional title for the graph
        show_scores: Whether to show scores in node labels
        max_label_length: Maximum length for state representation
        color_map: Color mapping for nodes. Can be:
            - None: Use default colormap
            - str: Colormap name (e.g., 'viridis', 'coolwarm')
            - ColorMap instance: Custom colormap
            - Callable[[float], str]: Custom function mapping score to hex color

    Raises:
        DependencyNotFoundError: If graphviz is not installed
        RenderError: If rendering fails
    """
    try:
        import graphviz  # type: ignore
    except ImportError:
        raise DependencyNotFoundError(
            "graphviz Python package is not installed. Install it with: pip install treequest[vis]"
        )

    # Normalize format
    format = format.lower()

    # Calculate score range for colormap
    scores = [node.score for node in snapshot.nodes if node.score >= 0]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    if min_score == max_score:  # Expand range to avoid division by zero
        max_score = max_score + 0.5
        min_score = min_score - 0.5

    # Resolve color_map to a callable
    color_fn = resolve_colormap(color_map, min_score, max_score)

    # Create directed graph
    dot = graphviz.Digraph(comment=title or "Tree Visualization")

    if title:
        dot.attr(label=title, labelloc="t", fontsize="16")

    # Add nodes
    for node in snapshot.nodes:
        node_id = str(node.id)

        # Create label
        if node.id == -1:  # Root node
            label = "ROOT"
            color = apply_status_color(node.status, ROOT_COLOR)
        else:
            # Truncate state representation if needed
            state_str = node.state_repr
            if len(state_str) > max_label_length:
                state_str = state_str[:max_label_length] + "..."

            # Build label with score and other info
            label_parts = [f"ID: {node.id}"]
            if show_scores:
                label_parts.append(f"Score: {node.score:.2f}")
            label_parts.append(state_str)

            label = "\\n".join(label_parts)

            # Get color from score
            color = color_fn(node.score)
            color = apply_status_color(node.status, color)

        # Create tooltip with metrics
        tooltip_parts = [f"Node ID: {node.id}", f"Score: {node.score:.4f}"]
        if node.status:
            tooltip_parts.append(f"Status: {node.status}")
        if node.action_from_parent:
            tooltip_parts.append(f"Action: {node.action_from_parent}")
        if node.algo_metrics:
            tooltip_parts.append("Metrics:")
            for key, value in node.algo_metrics.items():
                if isinstance(value, dict) and "value" in value:
                    display_value = value["value"]
                else:
                    display_value = value
                tooltip_parts.append(f"  {key}: {display_value}")
        if node.annotations:
            tooltip_parts.append("Annotations:")
            for key, value in node.annotations.items():
                tooltip_parts.append(f"  {key}: {value}")

        tooltip = "\\n".join(tooltip_parts)

        # Add node
        dot.node(node_id, label=label, style="filled", fillcolor=color, tooltip=tooltip)

    # Add edges
    for edge in snapshot.edges:
        source_id = str(edge.source)
        target_id = str(edge.target)

        # Create edge label
        edge_label = edge.action if edge.action else ""
        dot.edge(source_id, target_id, label=edge_label)

    # Render
    try:
        dot.render(filename=output_basename, format=format, cleanup=True)
    except graphviz.backend.execute.ExecutableNotFound:
        raise DependencyNotFoundError(
            "Graphviz executable is not in system PATH. "
            "Please install Graphviz: https://graphviz.org/download/"
        )
    except Exception as e:
        raise RenderError(f"Failed to render Graphviz output: {e}")
