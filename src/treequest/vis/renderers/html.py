"""D3.js-based interactive HTML renderer for tree visualization."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from treequest.vis.errors import DependencyNotFoundError, RenderError
from treequest.vis.snapshot import VisualizationSnapshot
from treequest.vis.renderers.json_yaml import snapshot_to_dict
from treequest.vis.renderers.color_utils import (
    ROOT_COLOR,
    ColorMap,
    apply_status_color,
    resolve_colormap,
)


def _get_d3_js() -> str:
    """Load d3.js from bundled assets."""
    d3_path = Path(__file__).parents[1] / "assets" / "d3.v7.min.js"
    if not d3_path.exists():
        raise RenderError(f"d3.js not found at {d3_path}")
    with open(d3_path, "r") as f:
        return f.read()


def _get_template() -> str:
    """Load HTML template from assets directory."""
    template_path = Path(__file__).parents[1] / "assets" / "d3_tree.html.jinja2"
    if not template_path.exists():
        raise RenderError(f"HTML template not found at {template_path}")
    with open(template_path, "r") as f:
        return f.read()


def render_html(
    snapshot: VisualizationSnapshot,
    output_basename: str,
    *,
    format: str = "html",
    theme: str = "light",
    color_map: Optional[Union[str, ColorMap, Callable[[float], str]]] = None,
    include_fields: Optional[List[str]] = None,
    include_algo_metrics: bool = True,
    include_annotations: bool = True,
) -> None:
    """
    Render a visualization snapshot as an interactive HTML page using D3.js.

    IMPORTANT: When using HTML format, ensure that the HTML file is securely handled,
               especially if the state formatter includes raw HTML content.
               Avoid opening untrusted HTML files in your browser.
               For example, XSS (cross site scripting) attacks can occur
               if the state includes malicious HTML/JavaScript code.

    Args:
        snapshot: Visualization snapshot to render
        output_basename: Output file path without extension
        format: Output format (should be "html")
        theme: Theme for the visualization ("light" or "dark")
        color_map: Color mapping for nodes. Can be:
            - None: Use default colormap
            - str: Colormap name (e.g., 'viridis', 'coolwarm')
            - ColorMap instance: Custom colormap
            - Callable[[float], str]: Custom function mapping score to hex color
        include_fields: Optional list of node fields to include
        include_algo_metrics: Whether to include algorithm metrics
        include_annotations: Whether to include annotations

    Raises:
        DependencyNotFoundError: If jinja2 is not installed
        RenderError: If rendering fails
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise DependencyNotFoundError(
            "jinja2 is not installed. Install it with: pip install treequest[vis]"
        )

    # Normalize format
    format = format.lower()
    if format not in ["html"]:
        raise ValueError(f"Unsupported format: {format}. Use 'html'.")

    try:
        # Convert snapshot to JSON string (no pretty-printing for compact HTML)
        snapshot_dict = snapshot_to_dict(
            snapshot,
            include_fields=include_fields,
            include_algo_metrics=include_algo_metrics,
            include_annotations=include_annotations,
        )
    except Exception as e:
        raise RenderError(f"Failed to convert snapshot to dictionary: {e}")

    # Load d3.js and Jinja2 template
    d3_js = _get_d3_js()
    template_str = _get_template()

    # Calculate score range for colormap
    scores = [node.score for node in snapshot.nodes if node.score >= 0]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    score_all_same = False
    if min_score == max_score:  # Expand range to avoid division by zero
        score_all_same = True
        max_score = max_score + 0.5
        min_score = min_score - 0.5

    # Resolve color_map to a callable
    color_fn = resolve_colormap(color_map, min_score, max_score)

    try:
        # Pre-compute node colors for client-side rendering
        node_colors: Dict[int, str] = {}
        for node in snapshot.nodes:
            if node.id == -1 or node.score < 0:
                base_color = ROOT_COLOR
            else:
                base_color = color_fn(node.score)
            node_colors[node.id] = apply_status_color(node.status, base_color)

        sample_count = 100
        legend_samples: List[Dict[str, Union[float, str]]] = []

        if score_all_same:
            color_value = color_fn(min_score)
            legend_samples = [
                {"value": float(min_score), "color": color_value}
                for _ in range(sample_count)
            ]
        else:
            for i in range(sample_count):
                position = i / (sample_count - 1)
                value = min_score + (max_score - min_score) * position
                legend_samples.append({"value": float(value), "color": color_fn(value)})

        # Render template
        template = Template(template_str, autoescape=True)
        html_content = template.render(
            snapshot_dict=snapshot_dict,
            metadata=snapshot.metadata,
            theme=theme,
            d3_js=d3_js,
            node_colors=node_colors,
            color_legend_samples=legend_samples,
            colormap_stats={"minScore": float(min_score), "maxScore": float(max_score)},
        )

        # Write to file
        with open(output_basename + ".html", "w") as f:
            f.write(html_content)

    except DependencyNotFoundError:
        raise
    except Exception as e:
        raise RenderError(f"Failed to render D3.js HTML: {e}")
