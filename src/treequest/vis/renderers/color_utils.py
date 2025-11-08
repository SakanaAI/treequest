import abc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional


def color_tuple_to_hex(color: Tuple[int, int, int]) -> str:
    """Convert an (R, G, B) tuple to a hex color string."""
    return "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])


def hex_to_color_tuple(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class ColorMap(abc.ABC):
    @abc.abstractmethod
    def get_color_tuple(self, value: float) -> Tuple[int, int, int]:
        pass

    def get_color_hex(self, value: float) -> str:
        """Get the hex color string for a given value."""
        color_tuple = self.get_color_tuple(value)
        return color_tuple_to_hex(color_tuple)


class GrayscaleColorMap(ColorMap):
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def get_color_tuple(self, value: float) -> Tuple[int, int, int]:
        """Map a value to a grayscale color."""
        if value < self.min_value:
            value = self.min_value
        if value > self.max_value:
            value = self.max_value
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        gray_level = int(normalized * 255)
        return (gray_level, gray_level, gray_level)


class InterpolatedColorMap(ColorMap):
    """Colormap using pre-computed color lookup table with linear interpolation.

    This colormap uses a list of RGB color tuples sampled from standard colormaps
    (e.g., matplotlib, seaborn) and performs linear interpolation between them.

    Args:
        color_data: List of RGB tuples (each value 0-255) representing the colormap
        min_value: Minimum value to map
        max_value: Maximum value to map
        reverse: If True, reverse the color order (equivalent to matplotlib's _r suffix)
    """

    def __init__(
        self,
        color_data: List[Tuple[int, int, int]],
        min_value: float = 0.0,
        max_value: float = 1.0,
        reverse: bool = False,
    ):
        if not color_data:
            raise ValueError("color_data must not be empty")
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")

        self.color_data = list(reversed(color_data)) if reverse else list(color_data)
        self.min_value = min_value
        self.max_value = max_value

    def get_color_tuple(self, value: float) -> Tuple[int, int, int]:
        """Map a value to an RGB color using linear interpolation."""
        # Clamp value to range
        if value <= self.min_value:
            return self.color_data[0]
        if value >= self.max_value:
            return self.color_data[-1]

        # Normalize to [0, 1]
        normalized = (value - self.min_value) / (self.max_value - self.min_value)

        # Calculate position in color_data
        position = normalized * (len(self.color_data) - 1)
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, len(self.color_data) - 1)

        # Linear interpolation between adjacent colors
        t = position - lower_idx
        lower_color = self.color_data[lower_idx]
        upper_color = self.color_data[upper_idx]

        r = int(lower_color[0] * (1 - t) + upper_color[0] * t)
        g = int(lower_color[1] * (1 - t) + upper_color[1] * t)
        b = int(lower_color[2] * (1 - t) + upper_color[2] * t)

        return (r, g, b)


# Colormap data loading and factory functions
_COLORMAP_DATA_CACHE: Optional[Dict[str, Any]] = None


def _load_colormap_data() -> Dict[str, Any]:
    """Load colormap data from JSON file (cached)."""
    global _COLORMAP_DATA_CACHE

    if _COLORMAP_DATA_CACHE is not None:
        return _COLORMAP_DATA_CACHE

    # Find the JSON file relative to this module
    data_path = Path(__file__).parents[1] / "assets" / "colormaps.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Colormap data file not found at {data_path}.")

    with open(data_path, "r") as f:
        _COLORMAP_DATA_CACHE = json.load(f)

    return _COLORMAP_DATA_CACHE


def list_colormaps() -> List[str]:
    """List all available colormap names.

    Returns:
        List of colormap names that can be used with get_colormap()
    """
    data = _load_colormap_data()
    return sorted(data["colormaps"].keys())


def get_colormap(
    name: str, min_value: float, max_value: float, reverse: bool = False
) -> InterpolatedColorMap:
    """Create a colormap instance by name.

    Args:
        name: Name of the colormap (e.g., 'viridis', 'coolwarm', 'rocket')
        min_value: Minimum value to map
        max_value: Maximum value to map
        reverse: If True, reverse the color order (equivalent to matplotlib's _r suffix)

    Returns:
        InterpolatedColorMap instance ready to use

    Raises:
        ValueError: If colormap name is not found

    Example:
        >>> cmap = get_colormap('coolwarm', 0.0, 100.0)
        >>> color = cmap.get_color_hex(50.0)
        >>> cmap_r = get_colormap('coolwarm', 0.0, 100.0, reverse=True)
    """
    data = _load_colormap_data()

    if name not in data["colormaps"]:
        available = sorted(data["colormaps"].keys())
        raise ValueError(
            f"Colormap '{name}' not found. "
            f"Available colormaps: {', '.join(available[:5])}... "
            f"(total {len(available)}). Use list_colormaps() for full list."
        )

    color_data = data["colormaps"][name]["colors"]

    # Convert list of lists to list of tuples
    color_tuples = [tuple(c) for c in color_data]

    return InterpolatedColorMap(
        color_data=color_tuples,
        min_value=min_value,
        max_value=max_value,
        reverse=reverse,
    )


def resolve_colormap(
    color_map_input: Optional[Any] = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
    default_colormap: str = "original",
) -> Callable[[float], str]:
    """Resolve a color_map input to a callable that returns hex color strings.

    Args:
        color_map_input: Can be:
            - None: Use default colormap
            - str: Colormap name (e.g., 'viridis', 'coolwarm')
            - ColorMap instance: Use its get_color_hex method
            - Callable[[float], str]: Use directly
        min_value: Minimum value for the colormap range
        max_value: Maximum value for the colormap range
        default_colormap: Default colormap name to use when color_map_input is None

    Returns:
        A callable that takes a float and returns a hex color string

    Raises:
        ValueError: If the input type is not supported or colormap name is invalid
        TypeError: If the input is not a valid type

    Example:
        >>> # Using a colormap name
        >>> color_fn = resolve_colormap('coolwarm', 0.0, 1.0)
        >>> color_fn(0.5)
        '#f7f7f7'

        >>> # Using a ColorMap instance
        >>> cmap = get_colormap('viridis', 0.0, 100.0)
        >>> color_fn = resolve_colormap(cmap, 0.0, 100.0)
        >>> color_fn(50.0)
        '#21918c'

        >>> # Using a custom callable
        >>> custom_fn = lambda x: '#ff0000' if x > 0.5 else '#0000ff'
        >>> color_fn = resolve_colormap(custom_fn, 0.0, 1.0)
        >>> color_fn(0.7)
        '#ff0000'
    """
    if color_map_input is None:
        # Use default colormap
        cmap = get_colormap(default_colormap, min_value, max_value)
        return cmap.get_color_hex

    elif isinstance(color_map_input, str):
        # Colormap name
        cmap = get_colormap(color_map_input, min_value, max_value)
        return cmap.get_color_hex

    elif isinstance(color_map_input, ColorMap):
        # ColorMap instance
        return color_map_input.get_color_hex

    elif callable(color_map_input):
        # Already a callable
        return color_map_input

    else:
        raise TypeError(
            f"color_map must be None, str, ColorMap, or Callable[[float], str], "
            f"got {type(color_map_input).__name__}"
        )


ROOT_COLOR = "#AAAAAA"  # light gray


def apply_status_color(status: Optional[str], default_color: str = ROOT_COLOR) -> str:
    """Adjust the node color based on execution status."""
    if status == "RUNNING":
        return "#C277DC"  # purple
    if status == "INVALID":
        return "#76502E"  # brown
    if status == "ROOT":
        return ROOT_COLOR
    return default_color


def expand_score_range(
    min_value: float,
    max_value: float,
    buffer: float = 0.5,
) -> Tuple[float, float]:
    """Guarantee a non-zero color range, even for uniform scores."""

    if max_value <= min_value:
        return min_value - buffer, max_value + buffer
    return min_value, max_value
