import math 
import matplotlib.pyplot as plt
from shapely import LineString
import numpy as np



def plot_shapes(
    poly,
    figsize=(5,5),
    ncols = 2,
):
    """
    Plot shape

    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    if poly.geom_type == 'Polygon':
        plt.figure(1, figsize=figsize)
        plt.plot(*poly.boundary.xy, c='red')
        plt.scatter(*poly.centroid.xy, c="blue", s=5)
        plt.title(f'threshold : {130}')
        plt.show()

    elif poly.geom_type == 'MultiPolygon':
        n = len(poly.geom_type)
        nrows = math.ceil(n / ncols)

        plt.figure(figsize=(13, 5 * nrows))
        plt.subplots_adjust(hspace =0.9, wspace=0.5)

        for n, pol in enumerate(poly.geoms):
            ax = plt.subplot(nrows, ncols, n+1)
            ax.plot(*pol.boundary.xy, c='red')
            ax.scatter(*pol.centroid.xy, c="blue", s=5)
            ax.set_title(f'Polygon {n+1}')
            ax.set_aspect('equal')
          
        plt.tight_layout()
        plt.show()
    
    else: 
        print('Shape is not a polygon or a multipolygon')


# def plot_genes_expr_in_shapes(
#   sdata,
# #   along|orthogonal
#     genes: list,
#     group_by,
#     cell_type,

# ):
#     sdata




def centerlines_orientation(
    centerlines: dict[str, LineString],
    ncols: int = 4,
    figsize_per_ax: tuple[float, float] = (4,4),
    point_size: int = 60,
) -> plt.Figure:
    """Plot each centerline in its own subplot, coloring start/end points to show orientation.

    Start point is shown in green, end point in red, so a reversed centerline
    is immediately visible when comparing subplots across samples.

    Parameters
    ----------
    centerlines
        Mapping of sample id -> centerline LineString.
    ncols
        Number of subplot columns.
    figsize_per_ax
        Figure size (width, height) allocated per subplot.
    point_size
        Marker size for start/end points.

    Returns
    -------
    The matplotlib Figure.
    """
    keys = list(centerlines.keys())
    n = len(keys)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, keys):
        line = centerlines[key]
        coords = np.array(line.coords)

        # full path colored by progression along the line (start -> end)
        ax.plot(coords[:, 0], coords[:, 1], color="gray", lw=1.5, zorder=1)
        sc = ax.scatter(
            coords[1:-1, 0], coords[1:-1, 1],
            c=np.arange(1, len(coords) - 1),
            cmap="viridis", s=point_size * 0.4, zorder=2,
        ) if len(coords) > 2 else None

        # start = green, end = red
        ax.scatter(*coords[0], color="green", s=point_size, zorder=3, label="start", edgecolor="k")
        ax.scatter(*coords[-1], color="red", s=point_size, zorder=3, label="end", edgecolor="k")

        ax.set_title(str(key), fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused axes
    for ax in axes_flat[n:]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="green", markeredgecolor="k", markersize=8, label="start"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="red", markeredgecolor="k", markersize=8, label="end"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    plt.show()
    # return fig