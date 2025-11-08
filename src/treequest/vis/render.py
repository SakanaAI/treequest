"""High-level API for tree visualization."""

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
    obj: Union[AlgoStateT, VisualizationSnapshot[StateT]],
    output_basename: Optional[str] = None,
    *,
    format: str,
    state_formatter: Optional[Callable[[StateT], str]] = None,
    annotations: Optional[Dict[str, Any]] = None,
    **renderer_kwargs,
) -> Optional[str]:
    """
    High-level API to render a tree visualization.

    This function accepts either an algorithm state or a pre-built snapshot,
    and renders it to the specified format.

    Args:
        obj: Algorithm state (e.g., MCTSState, BFSState) or a VisualizationSnapshot.
             Provide either of these.
        output_basename: Output file path without extension. If None and format
                        supports it (e.g., mermaid), returns a string.
        format: Output format. Supported values:
               - "png", "pdf", "svg", "jpg", "jpeg": Graphviz formats
               - "json", "yaml": Data export formats
               - "mermaid", "md": Mermaid diagram
               - "html": Interactive HTML (requires jinja2)
        state_formatter: Optional function to format node states
        annotations: Optional annotations to add to snapshot metadata
        **renderer_kwargs: Additional keyword arguments passed to the renderer

    Returns:
        For formats that support string output (mermaid), returns the string.
        For file-based formats, returns None.

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
        >>> # Get Mermaid diagram as string
        >>> diagram = tq.render(state, format="mermaid")
    """
    # Validate and resolve input object → snapshot
    if isinstance(obj, VisualizationSnapshot):
        snapshot = obj
    else:
        snapshot = build_snapshot(
            obj, state_formatter=state_formatter, annotations=annotations
        )

    # Normalize format
    format = format.lower()

    # Route to appropriate renderer
    if format in ["png", "pdf", "svg", "jpg", "jpeg"]:
        # Graphviz formats
        if output_basename is None:
            raise VisualizationError(
                f"output_basename is required for format '{format}' (Graphviz)"
            )
        render_graphviz(snapshot, output_basename, format=format, **renderer_kwargs)
        return None
    elif format in ["json", "yaml"]:
        # Data export formats
        if output_basename is None:
            raise VisualizationError(
                f"output_basename is required for format '{format}'"
            )
        dump_snapshot(snapshot, output_basename, format=format, **renderer_kwargs)
        return None
    elif format in ["mermaid", "md", "markdown"]:
        # Mermaid diagram
        result = render_mermaid(
            snapshot, output_basename, format=format, **renderer_kwargs
        )
        return result
    elif format == "html":
        # HTML renderer
        try:
            if output_basename is None:
                raise VisualizationError(
                    "output_basename is required for format 'html'"
                )
            render_html(snapshot, output_basename, format=format, **renderer_kwargs)
            return None
        except ImportError:
            raise VisualizationError(
                "HTML rendering requires jinja2. "
                "Install it with: pip install treequest[vis-interactive]"
            )
    else:
        raise VisualizationError(
            f"Unsupported format: {format}. "
            f"Supported formats: png, pdf, svg, jpg, jpeg, json, yaml, mermaid, md, html"
        )
