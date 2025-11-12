"""Tests for color_utils module."""

import pytest

from treequest.vis.renderers.color_utils import (
    color_tuple_to_hex,
    get_colormap,
    hex_to_color_tuple,
    GrayscaleColorMap,
    InterpolatedColorMap,
    list_colormap_names,
)


class TestColorConversion:
    """Test color conversion utilities."""

    def test_color_tuple_to_hex(self):
        assert color_tuple_to_hex((255, 0, 0)) == "#ff0000"
        assert color_tuple_to_hex((0, 255, 0)) == "#00ff00"
        assert color_tuple_to_hex((0, 0, 255)) == "#0000ff"
        assert color_tuple_to_hex((128, 128, 128)) == "#808080"
        assert color_tuple_to_hex((0, 0, 0)) == "#000000"
        assert color_tuple_to_hex((255, 255, 255)) == "#ffffff"

    def test_hex_to_color_tuple(self):
        assert hex_to_color_tuple("#ff0000") == (255, 0, 0)
        assert hex_to_color_tuple("#00ff00") == (0, 255, 0)
        assert hex_to_color_tuple("#0000ff") == (0, 0, 255)
        assert hex_to_color_tuple("#808080") == (128, 128, 128)
        assert hex_to_color_tuple("000000") == (0, 0, 0)
        assert hex_to_color_tuple("ffffff") == (255, 255, 255)

    def test_roundtrip_conversion(self):
        original = (123, 45, 67)
        hex_color = color_tuple_to_hex(original)
        converted_back = hex_to_color_tuple(hex_color)
        assert original == converted_back


class TestGrayscaleColorMap:
    """Test GrayscaleColorMap."""

    def test_basic_mapping(self):
        cmap = GrayscaleColorMap(0.0, 100.0)
        assert cmap.get_color_tuple(0.0) == (0, 0, 0)
        assert cmap.get_color_tuple(100.0) == (255, 255, 255)

        gray = cmap.get_color_tuple(50.0)
        assert gray[0] == gray[1] == gray[2]
        assert 120 < gray[0] < 135  # Approximately 127.5

    def test_clamping(self):
        cmap = GrayscaleColorMap(0.0, 100.0)
        assert cmap.get_color_tuple(-10.0) == (0, 0, 0)
        assert cmap.get_color_tuple(200.0) == (255, 255, 255)

    def test_get_color_hex(self):
        cmap = GrayscaleColorMap(0.0, 100.0)
        assert cmap.get_color_hex(0.0) == "#000000"
        assert cmap.get_color_hex(100.0) == "#ffffff"


class TestInterpolatedColorMap:
    """Test InterpolatedColorMap."""

    def test_simple_gradient(self):
        colors = [(255, 0, 127), (0, 0, 255)]
        cmap = InterpolatedColorMap(colors)
        assert cmap.get_color_tuple(0.0) == (255, 0, 127)
        assert cmap.get_color_tuple(1.0) == (0, 0, 255)

        middle = cmap.get_color_tuple(0.5)
        assert middle == (127, 0, 191)

    def test_reversed_gradient(self):
        colors = [(255, 0, 127), (0, 0, 255)]
        cmap = InterpolatedColorMap(colors, reverse=True)
        assert cmap.get_color_tuple(0.0) == (0, 0, 255)
        assert cmap.get_color_tuple(1.0) == (255, 0, 127)

    def test_multiple_colors(self):
        colors = [(255, 0, 127), (0, 255, 191), (0, 0, 255)]
        cmap = InterpolatedColorMap(colors, max_value=100.0)
        assert cmap.get_color_tuple(0.0) == (255, 0, 127)
        assert cmap.get_color_tuple(100.0) == (0, 0, 255)

        middle = cmap.get_color_tuple(50.0)
        assert middle == (0, 255, 191)

    def test_clamping(self):
        colors = [(255, 0, 0), (0, 0, 255)]
        cmap = InterpolatedColorMap(colors, 0.0, 1.0)
        assert cmap.get_color_tuple(-1.0) == (255, 0, 0)
        assert cmap.get_color_tuple(2.0) == (0, 0, 255)

    def test_empty_colors_raises_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            InterpolatedColorMap([], 0.0, 1.0)

    def test_invalid_range_raises_error(self):
        colors = [(255, 0, 0), (0, 0, 255)]
        with pytest.raises(ValueError, match="must be less than"):
            InterpolatedColorMap(colors, 1.0, 0.0)

    def test_invalid_color_values_rgba_raise_error(self):
        colors = [(255, 0, 0, 255), (0, 0, 255, 255)]
        with pytest.raises(
            ValueError, match="Each color in color_data must be a tuple"
        ):
            InterpolatedColorMap(colors, 0.0, 1.0)

    def test_invalid_color_values_not_integers_raise_error(self):
        colors = [(1.0, 0, 0), (0, 0, 1.0)]
        with pytest.raises(
            ValueError, match="Each color in color_data must be a tuple"
        ):
            InterpolatedColorMap(colors, 0.0, 1.0)

    def test_invalid_color_values_out_of_range_raise_error(self):
        colors = [(-1, 0, 0), (0, 0, 256)]
        with pytest.raises(
            ValueError, match="Each color in color_data must be a tuple"
        ):
            InterpolatedColorMap(colors, 0.0, 1.0)


class TestColormapLoading:
    """Test colormap data loading and factory functions."""

    def test_list_colormap_names(self):
        colormaps = list_colormap_names()
        assert isinstance(colormaps, list)
        assert len(colormaps) > 0
        assert "viridis" in colormaps
        assert "coolwarm" in colormaps
        assert colormaps == sorted(colormaps)

    def test_get_colormap_basic(self):
        cmap = get_colormap("viridis", 0.0, 1.0)
        assert isinstance(cmap, InterpolatedColorMap)
        assert cmap.min_value == 0.0
        assert cmap.max_value == 1.0

        color_min = cmap.get_color_tuple(0.0)
        color_max = cmap.get_color_tuple(1.0)
        assert len(color_min) == 3
        assert len(color_max) == 3
        assert all(0 <= c <= 255 for c in color_min)
        assert all(0 <= c <= 255 for c in color_max)

    def test_get_colormap_reversed(self):
        cmap_normal = get_colormap("viridis", 0.0, 1.0, reverse=False)
        cmap_reversed = get_colormap("viridis", 0.0, 1.0, reverse=True)
        color_min_normal = cmap_normal.get_color_tuple(0.0)
        color_max_normal = cmap_normal.get_color_tuple(1.0)
        color_min_reversed = cmap_reversed.get_color_tuple(0.0)
        color_max_reversed = cmap_reversed.get_color_tuple(1.0)
        assert color_min_normal == color_max_reversed
        assert color_max_normal == color_min_reversed

    def test_get_colormap_invalid_name(self):
        with pytest.raises(ValueError, match="not found"):
            get_colormap("nonexistent_colormap", 0.0, 1.0)

    def test_get_colormap_hex_output(self):
        cmap = get_colormap("coolwarm", 0.0, 100.0)
        hex_color = cmap.get_color_hex(50.0)
        assert isinstance(hex_color, str)
        assert hex_color.startswith("#")
        assert len(hex_color) == 7

    def test_multiple_colormaps(self):
        """Test that multiple common colormaps can be loaded."""
        common_colormaps = ["viridis", "plasma", "coolwarm", "Greys"]

        for name in common_colormaps:
            cmap = get_colormap(name, 0.0, 1.0)
            assert isinstance(cmap, InterpolatedColorMap)
            for val in [0.0, 0.5, 1.0]:
                color = cmap.get_color_tuple(val)
                assert len(color) == 3
                assert all(0 <= c <= 255 for c in color)


class TestColormapIntegration:
    """Integration tests for complete colormap workflow."""

    def test_full_workflow(self):
        """Test complete workflow from listing to using colormaps."""
        available = list_colormap_names()
        assert len(available) > 0

        cmap_name = available[0]

        cmap = get_colormap(cmap_name, 0.0, 100.0)
        colors = [cmap.get_color_tuple(i) for i in range(0, 101, 10)]
        assert len(colors) == 11

        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_colormap_smoothness(self):
        """Test that colormap interpolation is smooth."""
        cmap = get_colormap("viridis", 0.0, 100.0)
        colors = [cmap.get_color_tuple(i) for i in range(101)]

        # Check that adjacent colors don't differ too much
        for i in range(len(colors) - 1):
            c1, c2 = colors[i], colors[i + 1]
            for channel in range(3):
                diff = abs(c1[channel] - c2[channel])
                assert diff < 10  # No huge jumps between adjacent values


def test_original_colormap_is_available() -> None:
    cmap = get_colormap("original", -1.0, 1.0)
    for probe in (-1.0, 0.0, 1.0):
        color = cmap.get_color_tuple(probe)
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)
