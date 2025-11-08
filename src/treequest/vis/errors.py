"""Error and warning classes for visualization module."""


class VisualizationError(Exception):
    """Base class for visualization-related errors."""

    pass


class DependencyNotFoundError(VisualizationError):
    """Raised when a required dependency or system binary is not found."""

    pass


class InvalidStateError(VisualizationError):
    """Raised when state data is invalid or malformed."""

    pass


class RenderError(VisualizationError):
    """Raised when rendering process fails."""

    pass


class SecurityWarning(UserWarning):
    """Warning raised when potentially unsafe operations are performed."""

    pass
