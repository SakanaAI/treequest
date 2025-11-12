"""Mermaid diagram renderer for tree visualization."""

from typing import Callable, Optional, Union

from treequest.vis.errors import RenderError
from treequest.vis.snapshot import VisualizationSnapshot
from treequest.vis.renderers.color_utils import ROOT_COLOR, ColorMap, resolve_colormap


def render_mermaid(
    snapshot: VisualizationSnapshot,
    output_basename: str,
    *,
    format: str = "mermaid",
    theme: str = "default",
    max_nodes: Optional[int] = None,
    color_map: Optional[Union[str, ColorMap, Callable[[float], str]]] = None,
) -> None:
    """
    Render a visualization snapshot as a Mermaid diagram.

    Args:
        snapshot: Visualization snapshot to render
        output_basename: Output file path without extension.
        format: Output format ("mermaid" or "md" for markdown)
        theme: Mermaid theme ("default", "dark", "forest", etc.)
        max_nodes: Maximum number of nodes to include. If exceeded,
                  only the highest-scoring nodes will be included.
        color_map: Color mapping for nodes. Can be:
            - None: Use default colormap
            - str: Colormap name (e.g., 'viridis', 'coolwarm')
            - ColorMap instance: Custom colormap
            - Callable[[float], str]: Custom function mapping score to hex color

    Raises:
        RenderError: If rendering fails
        ValueError: If format is not supported
    """
    # Normalize format
    format = format.lower()
    if format not in ["mermaid", "md", "markdown"]:
        raise ValueError(
            f"Unsupported format: {format}. Use 'mermaid', 'md', or 'markdown'."
        )

    try:
        # Filter nodes if max_nodes is specified
        nodes = snapshot.nodes
        if max_nodes is not None and len(nodes) > max_nodes:
            # Sort by score and take top nodes, always include root
            root = [n for n in nodes if n.id == -1]
            non_root = [n for n in nodes if n.id != -1]
            non_root_sorted = sorted(non_root, key=lambda n: n.score, reverse=True)
            nodes = root + non_root_sorted[: max_nodes - 1]

        # Build node ID mapping
        node_ids = {node.id for node in nodes}

        # Calculate score range for colormap
        scores = [node.score for node in nodes if node.score >= 0]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        if min_score == max_score:  # Expand range to avoid division by zero
            max_score = max_score + 0.5
            min_score = min_score - 0.5

        # Resolve color_map to a callable
        color_fn = resolve_colormap(color_map, min_score, max_score)

        # Start building Mermaid diagram
        lines = ["%%{init: {'theme':'" + theme + "'}}%%", "graph TD"]

        # Add nodes with direct color specification
        for node in nodes:
            node_id = f"node{node.id}"

            # Create label
            if node.id == -1:
                label = "ROOT"
            else:
                label = f"ID: {node.id}"
                label += f"<br/>Score: {node.score:.2f}"
                label += f"<br/>{node.state_repr[:20]}"
                if len(node.state_repr) > 20:
                    label += "..."

            # Get color from colormap
            if node.id == -1 or node.score < 0:
                fill_color = ROOT_COLOR
            else:
                fill_color = color_fn(node.score)

            # Add node
            lines.append(f'    {node_id}["{label}"]')
            # Add style with direct color specification
            lines.append(
                f"    style {node_id} fill:{fill_color},stroke:#333,stroke-width:2px"
            )

        # Add edges (only if both nodes are in the filtered set)
        for edge in snapshot.edges:
            if edge.source in node_ids and edge.target in node_ids:
                source_id = f"node{edge.source}"
                target_id = f"node{edge.target}"

                # Create edge label
                edge_label = edge.action if edge.action else ""
                if edge_label:
                    lines.append(f'    {source_id} -->|"{edge_label}"| {target_id}')
                else:
                    lines.append(f"    {source_id} --> {target_id}")

        mermaid_str = "\n".join(lines)

        # Wrap in markdown code block if format is markdown
        if format in ["md", "markdown"]:
            mermaid_str = f"```mermaid\n{mermaid_str}\n```"

        # Write to file if output_basename is provided
        ext = ".md" if format in ["md", "markdown"] else ".mermaid"
        with open(output_basename + ext, "w") as f:
            f.write(mermaid_str)
    except Exception as e:
        raise RenderError(f"Failed to render Mermaid diagram: {e}")
