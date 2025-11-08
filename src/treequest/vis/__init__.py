"""Tree visualization module for TreeQuest.

This module provides comprehensive visualization capabilities for tree search algorithms,
including:

- Multiple output formats: Graphviz, Mermaid, JSON, YAML, HTML
- Interactive HTML visualization with time-based exploration
- Algorithm-specific metrics extraction and display
- High-level API for easy visualization

Example usage:

    >>> import treequest as tq
    >>> from treequest import vis
    >>>
    >>> algo = tq.StandardMCTS()
    >>> state = algo.init_tree()
    >>> # ... run algorithm ...
    >>>
    >>> # Simple high-level API
    >>> tq.render(state, output_basename="output", format="html")
    >>> tq.render(state, output_basename="output", format="png")
    >>>
    >>> # Get Mermaid diagram as string
    >>> diagram = tq.render(state, format="mermaid")
    >>>
    >>> # Low-level API with custom formatting
    >>> snapshot = vis.build_snapshot(
    ...     state,
    ...     state_formatter=lambda s: str(s)[:10],
    ...     annotations={"experiment": "run42"}
    ... )
    >>> vis.render_graphviz(snapshot, "output", format="pdf")
"""

from treequest.vis.algo_adapters import register_adapter
from treequest.vis.build_snapshot import build_snapshot
from treequest.vis.render import render
from treequest.vis.errors import (
    DependencyNotFoundError,
    InvalidStateError,
    RenderError,
    SecurityWarning,
    VisualizationError,
)
from treequest.vis.renderers import (
    dump_snapshot,
    render_graphviz,
    render_html,
    render_mermaid,
)
from treequest.vis.snapshot import (
    EdgeSnapshot,
    NodeSnapshot,
    TrialSnapshot,
    VisualizationSnapshot,
)

__all__ = [
    # High-level API
    "render",
    "build_snapshot",
    # Low-level renderers
    "render_graphviz",
    "render_html",
    "render_mermaid",
    "dump_snapshot",
    # Data structures
    "VisualizationSnapshot",
    "NodeSnapshot",
    "EdgeSnapshot",
    "TrialSnapshot",
    # Errors
    "VisualizationError",
    "DependencyNotFoundError",
    "InvalidStateError",
    "RenderError",
    "SecurityWarning",
    # Adapter registration
    "register_adapter",
]
