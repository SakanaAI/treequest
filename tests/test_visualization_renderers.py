"""Tests for visualization renderers."""

import importlib.util
import json
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pytest

from treequest.algos.standard_mcts import StandardMCTS
from treequest.vis import build_snapshot, render
from treequest.vis.errors import DependencyNotFoundError, VisualizationError


def create_test_state():
    """Create a simple test state for rendering tests."""
    random.seed(42)

    def generate_fn(state: Optional[str]) -> Tuple[str, float]:
        score = random.uniform(0.0, 1.0)
        return f"State(score={score:.2f})", score

    algo = StandardMCTS(samples_per_action=1)
    state = algo.init_tree()

    generate_fns = {"action": generate_fn}

    # Run a few steps
    for _ in range(5):
        state = algo.step(state, generate_fns)

    return state


def test_render_json():
    """Test rendering to JSON format."""
    state = create_test_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test"
        render(state, output_basename=str(output_path), format="json")

        # Verify JSON file was created
        json_file = Path(str(output_path) + ".json")
        assert json_file.exists()

        # Verify JSON is valid and contains expected structure
        with open(json_file) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
        assert "trials" in data
        assert "metadata" in data
        assert len(data["nodes"]) > 0


def test_render_mermaid():
    """Test rendering to Mermaid format."""
    state = create_test_state()

    # Render to file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test"
        render(state, output_basename=str(output_path), format="mermaid")
        mermaid_file = Path(str(output_path) + ".mermaid")
        assert mermaid_file.exists()


def test_render_mermaid_with_max_nodes():
    """Test rendering Mermaid with node limit."""
    state = create_test_state()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test"
        render(state, output_basename=str(output_path), format="mermaid", max_nodes=3)
        mermaid_file = Path(str(output_path) + ".mermaid")
        assert mermaid_file.exists()

        # Count node definition lines only (avoid counting edges/style directives)
        with open(mermaid_file) as f:
            result = f.read()
        defs = re.findall(r"^\s*node-?\d+\[", result, flags=re.MULTILINE)
        assert len(defs) <= 3


def test_render_graphviz_honors_dot_availability():
    """Graphviz rendering should succeed only when both module and `dot` binary exist."""

    state = create_test_state()
    graphviz_spec = importlib.util.find_spec("graphviz")
    dot_path = shutil.which("dot")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test"
        if graphviz_spec is None or dot_path is None:
            with pytest.raises(DependencyNotFoundError):
                render(state, output_basename=str(output_path), format="png")
        else:
            render(state, output_basename=str(output_path), format="png")
            png_file = Path(str(output_path) + ".png")
            assert png_file.is_file(), "PNG file was not created"


def test_render_html():
    """Test rendering to HTML format."""
    state = create_test_state()

    jinja2_spec = importlib.util.find_spec("jinja2")
    if jinja2_spec is None:
        pytest.skip("jinja2 not installed")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test"
            render(state, output_basename=str(output_path), format="html")

            # Verify HTML file was created
            html_file = Path(str(output_path) + ".html")
            assert html_file.exists()

            # Verify HTML contains expected elements
            with open(html_file) as f:
                content = f.read()

            assert "TreeQuest Visualization" in content
            assert "snapshotData" in content
            assert "StandardMCTS" in content


def test_render_with_snapshot():
    """Test rendering with pre-built snapshot."""
    state = create_test_state()
    snapshot = build_snapshot(state)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test"
        render(snapshot, str(output_path), format="json")

        json_file = Path(str(output_path) + ".json")
        assert json_file.exists()


def test_render_invalid_format():
    """Test that rendering with invalid format raises error."""
    state = create_test_state()

    with pytest.raises(VisualizationError):
        render(state, output_basename="test", format="invalid_format")
