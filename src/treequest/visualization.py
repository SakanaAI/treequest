import warnings
from typing import Optional

import graphviz  # type: ignore

from treequest.algos.tree import Tree


def visualize_tree_graphviz(
    tree: Tree,
    save_path: Optional[str] = None,
    show_scores: bool = True,
    max_label_length: int = 20,
    title: Optional[str] = None,
    format: str = "png",
):
    """
    Visualize a Tree using Graphviz.

    .. deprecated:: 0.2.0
        This is a legacy API maintained for backward compatibility.
        For new code, use the `treequest.vis` module which provides:
        - Multiple output formats (HTML, Mermaid, JSON, YAML)
        - Interactive HTML visualization with time-based exploration
        - Algorithm-specific metrics display
        - Better integration with trial information

        Example::

            from treequest import vis
            vis.render(state, output_basename="output", format="html")

    Args:
        tree: The Tree object to visualize
        save_path: Optional path to save the visualization (without extension)
        show_scores: Whether to show scores in node labels
        max_label_length: Maximum length for node labels
        title: Optional title for the visualization
        format: Output format (png, pdf, svg, etc.)

    Returns:
        Graphviz Digraph object
    """
    # Emit deprecation warning
    warnings.warn(
        "visualize_tree_graphviz() is deprecated and will be removed in a future version. "
        "Please use treequest.vis.render() instead for more features and better integration. "
        "See: https://docs.treequest.dev/visualization",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        # Create a directed graph
        dot = graphviz.Digraph(comment=title or "Tree Visualization")

        if title:
            dot.attr(label=title)

        # Add nodes and edges
        nodes = tree.get_nodes()

        for node in nodes:
            # Create a unique identifier for the node
            node_id = str(node.expand_idx)

            # Create label based on state and score
            if node.is_root():
                label = "ROOT"
                color = "lightgray"
            else:
                # Truncate state representation if needed
                state_str = str(node.state)
                if len(state_str) > max_label_length:
                    state_str = state_str[:max_label_length] + "..."

                if show_scores:
                    label = f"{state_str}\\nScore: {node.score:.2f}\\nExpandIdx: {node.expand_idx}"
                else:
                    label = state_str

                # Color based on score (green for high scores, red for low)
                if node.score >= 0.7:
                    color = "lightgreen"
                elif node.score >= 0.4:
                    color = "lightyellow"
                else:
                    color = "lightcoral"

            # Add node with label
            dot.node(node_id, label=label, style="filled", fillcolor=color)

            # Add edge from parent to this node
            if node.parent:
                dot.edge(str(node.parent.expand_idx), node_id)

        # Render and save if path is provided
        if save_path:
            dot.render(save_path, format=format, cleanup=True)

        return dot

    except graphviz.backend.execute.ExecutableNotFound:
        print("graphviz executable is not in system Path, visualization skipped...")
        return None
