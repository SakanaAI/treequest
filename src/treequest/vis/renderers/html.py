"""D3.js-based interactive HTML renderer for tree visualization."""

import json
from pathlib import Path
from typing import Callable, Optional, Union

from treequest.vis.errors import DependencyNotFoundError, RenderError
from treequest.vis.renderers.json_yaml import snapshot_to_json_string
from treequest.vis.snapshot import VisualizationSnapshot
from treequest.vis.renderers.color_utils import (
    ROOT_COLOR,
    ColorMap,
    apply_status_color,
    expand_score_range,
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
    embed_snapshot: bool = True,
    color_map: Optional[Union[str, ColorMap, Callable[[float], str]]] = None,
) -> None:
    """
    Render a visualization snapshot as an interactive HTML page using D3.js.

    Args:
        snapshot: Visualization snapshot to render
        output_basename: Output file path without extension
        format: Output format (should be "html")
        theme: Theme for the visualization ("light" or "dark")
        embed_snapshot: Whether to embed the snapshot data in the HTML
        color_map: Color mapping for nodes. Can be:
            - None: Use default colormap
            - str: Colormap name (e.g., 'viridis', 'coolwarm')
            - ColorMap instance: Custom colormap
            - Callable[[float], str]: Custom function mapping score to hex color
            Note: This parameter prepares colormap data for JavaScript,
                  but full dynamic colormap support in D3 visualization
                  will be implemented in a future update.

    Raises:
        DependencyNotFoundError: If jinja2 is not installed
        RenderError: If rendering fails
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise DependencyNotFoundError(
            "jinja2 is not installed. Install it with: pip install treequest[vis-interactive]"
        )

    if not embed_snapshot:
        raise NotImplementedError(
            "External JSON loading is not yet implemented. Use embed_snapshot=True."
        )

    try:
        # Load d3.js
        d3_js = _get_d3_js()

        # Load template
        template_str = _get_template()

        # Convert snapshot to JSON string
        snapshot_json = snapshot_to_json_string(snapshot, indent=None)

        # Calculate score range for colormap
        scores = [node.score for node in snapshot.nodes if node.score >= 0]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 1.0
        min_score, max_score = expand_score_range(min_score, max_score)

        # Resolve color_map to a callable (for potential future use)
        color_fn = resolve_colormap(color_map, min_score, max_score)

        # Pre-compute node colors for client-side rendering
        node_colors: dict[int, str] = {}
        for node in snapshot.nodes:
            if node.id == -1 or node.score < 0:
                base_color = ROOT_COLOR
            else:
                base_color = color_fn(node.score)
            node_colors[node.id] = apply_status_color(node.status, base_color)

        node_colors_json = json.dumps(node_colors)

        sample_count = 100
        legend_samples: list[dict[str, float | str]] = []

        if max_score <= min_score:
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

        legend_samples_json = json.dumps(legend_samples)
        colormap_stats_json = json.dumps(
            {"minScore": float(min_score), "maxScore": float(max_score)}
        )

        # Render template
        template = Template(template_str, autoescape=True)
        html_content = template.render(
            snapshot_json=snapshot_json,
            metadata=snapshot.metadata,
            theme=theme,
            d3_js=d3_js,
            node_colors=node_colors_json,
            color_legend_samples=legend_samples_json,
            colormap_stats=colormap_stats_json,
        )

        # Write to file
        output_path = Path(output_basename)
        if not str(output_path).endswith(".html"):
            output_path = Path(str(output_path) + ".html")

        with open(output_path, "w") as f:
            f.write(html_content)

    except DependencyNotFoundError:
        raise
    except Exception as e:
        raise RenderError(f"Failed to render D3.js HTML: {e}")
