"""High-level API for tree visualization."""

import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from treequest.vis.build_snapshot import build_snapshot
from treequest.vis.errors import VisualizationError
from treequest.vis.renderers.graphviz_renderer import render_graphviz
from treequest.vis.renderers.html import render_html
from treequest.vis.renderers.json_yaml import dump_snapshot
from treequest.vis.renderers.mermaid import render_mermaid
from treequest.vis.snapshot import VisualizationSnapshot

StateT = TypeVar("StateT")
AlgoStateT = TypeVar("AlgoStateT")


def render(
    algo_state_or_snapshot: Union[AlgoStateT, VisualizationSnapshot[StateT]],
    output_basename: Union[str, Path],
    *,
    format: str,
    state_formatter: Optional[Callable[[StateT], str]] = None,
    annotations: Optional[Dict[str, Any]] = None,
    **renderer_kwargs,
) -> None:
    """
    High-level API to render a tree visualization.

    This function accepts either an algorithm state or a pre-built snapshot,
    and renders it to the specified format.

    Args:
        algo_state_or_snapshot: Algorithm state (e.g., MCTSState, BFSState) or a VisualizationSnapshot.
             Provide either of these.
        output_basename: Output file path without extension. If an existing directory is provided,
                         a timestamped filename (treequest_YYYYMMDD_HHMMSS) will be generated inside it.
        format: Output format. Supported values:
               - "png", "pdf", "svg", "jpg", "jpeg": Graphviz formats
               - "json", "yaml": Data export formats
               - "mermaid", "md": Mermaid diagram
               - "html": Interactive HTML (requires jinja2)
        state_formatter: Optional function to format node states
        annotations: Optional annotations to add to snapshot metadata
        **renderer_kwargs: Additional keyword arguments passed to the renderer

    Raises:
        VisualizationError: If inputs are invalid or rendering fails

    Examples:
        >>> import treequest as tq
        >>>
        >>> algo = tq.StandardMCTS()
        >>> state = algo.init_tree()
        >>> # ... run algorithm ...
        >>>
        >>> # Render to HTML
        >>> tq.render(state, "logs/run42", format="html")
        >>>
        >>> # Render to PNG
        >>> tq.render(state, "logs/tree", format="png")
        >>>
        >>> # Render a Mermaid diagram (Markdown format)
        >>> tq.render(state, "logs", format="md")
    """
    # Validate and resolve input object → snapshot
    if isinstance(algo_state_or_snapshot, VisualizationSnapshot):
        snapshot = algo_state_or_snapshot
    else:
        snapshot = build_snapshot(
            algo_state_or_snapshot,
            state_formatter=state_formatter,
            annotations=annotations,
        )

    output_path = Path(output_basename).resolve()
    if output_path.is_dir():  # Generate filename with timestamp
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_basename = str(output_path / f"treequest_{timestamp}")
    else:
        output_basename = str(output_path)

    # Route to appropriate renderer
    format = format.lower()
    if format in ["png", "pdf", "svg", "jpg", "jpeg"]:  # Graphviz formats
        render_graphviz(snapshot, output_basename, format=format, **renderer_kwargs)
    elif format in ["json", "yaml"]:  # Data export formats
        dump_snapshot(snapshot, output_basename, format=format, **renderer_kwargs)
    elif format in ["mermaid", "md", "markdown"]:  # Mermaid diagram
        render_mermaid(snapshot, output_basename, format=format, **renderer_kwargs)
    elif format == "html":  # HTML renderer
        try:
            render_html(snapshot, output_basename, format=format, **renderer_kwargs)
        except ImportError:
            raise VisualizationError(
                "HTML rendering requires jinja2. Install it with: pip install treequest[vis]"
            )
    else:
        raise VisualizationError(
            f"Unsupported format: {format}. "
            f"Supported formats: png, pdf, svg, jpg, jpeg, json, yaml, mermaid, md, html"
        )
