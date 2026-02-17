"""Base utilities for interactive visualizations using Plotly.

Provides theme configuration, image encoding for tooltips,
and chart export functionality.
"""

import base64
from io import BytesIO
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio


def configure_theme() -> None:
    """Configure Plotly theme for clean, publication-quality figures."""
    pio.templates["viz2psy"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Arial, sans-serif", size=12),
            title=dict(font=dict(size=14), x=0.0, xanchor="left"),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=False,
            ),
            colorway=[
                "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
            ],
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
    )
    pio.templates.default = "viz2psy"


def encode_image_base64(path: str | Path, max_size: int | None = None) -> str:
    """Encode an image as a base64 string for embedding.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    max_size : int, optional
        Maximum dimension for resizing (preserves aspect ratio).

    Returns
    -------
    str
        Base64-encoded string (without data URL prefix).
    """
    from PIL import Image

    img = Image.open(path)

    if max_size:
        img.thumbnail((max_size, max_size))

    # Convert to RGB if necessary
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image_data_url(path: str | Path, max_size: int | None = None) -> str:
    """Encode an image as a data URL for use in HTML/tooltips.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    max_size : int, optional
        Maximum dimension for resizing.

    Returns
    -------
    str
        Data URL string (e.g., "data:image/png;base64,...").
    """
    b64 = encode_image_base64(path, max_size)
    return f"data:image/png;base64,{b64}"


def save_figure(
    fig: go.Figure,
    path: str | Path,
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
) -> None:
    """Save a Plotly figure to file (HTML, PNG, SVG, PDF).

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to save.
    path : str or Path
        Output path. Extension determines format:
        - .html: Interactive HTML
        - .png: High-DPI PNG via kaleido
        - .svg: Vector SVG
        - .pdf: PDF document
    width : int, optional
        Width in pixels for static exports.
    height : int, optional
        Height in pixels for static exports.
    scale : float
        Scale factor for static exports (default: 2.0 for retina).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".html":
        fig.write_html(str(path), include_plotlyjs="cdn")
    elif suffix in (".png", ".svg", ".pdf", ".jpeg", ".jpg", ".webp"):
        fig.write_image(
            str(path),
            width=width,
            height=height,
            scale=scale,
        )
    else:
        # Default to HTML
        fig.write_html(str(path), include_plotlyjs="cdn")


def display_or_save(
    fig: go.Figure,
    output: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
) -> go.Figure | None:
    """Display figure in Jupyter or save to file.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure.
    output : str or Path, optional
        If provided, save to this path. Otherwise, return figure for display.
    width : int, optional
        Width for static exports.
    height : int, optional
        Height for static exports.

    Returns
    -------
    go.Figure or None
        Returns the figure if no output path provided (for Jupyter display),
        otherwise returns None after saving.
    """
    if output is None:
        return fig

    save_figure(fig, output, width=width, height=height)
    print(f"Saved to {output}")
    return None
