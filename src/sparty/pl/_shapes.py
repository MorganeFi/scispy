import matplotlib.pyplot as plt
from shapely import Polygon, LineString
# from shapely.plotting import plot_polygon
import math
import numpy as np
# from squidpy._docs import d
import pandas as pd
import matplotlib.patches as mpatches

def spatial_distance(
    dfs: dict,
    column: str,
    ncols: int = 4,
    figsize_per_plot: tuple = (5, 5),
    markersize: float = 1.0,
    cmap: str = "viridis",
    shared_scale: bool = True,
    show_axis: bool = False,
    show_colorbar: bool = True,
    show: bool = True,
    return_y_axis: bool = True,
    return_fig: bool = False,
    save: str | None = None,
    dpi: int = 300,
    alpha: float = 0.8,
    **plot_kwargs,
):
    """
    Plot a given column of (Geo)DataFrames stored in a dictionary, using the
    centroid of each geometry as a scatter point (works for both Point and
    Polygon geometries). Supports both numeric and categorical columns.

    Parameters
    ----------
    dfs : dict
        Dictionary {sample_name: GeoDataFrame}. Each GeoDataFrame must
        contain the column given by `column` and a `geometry` column.
    column : str
        Name of the column to visualize. Can be numeric (continuous colormap)
        or categorical/object/bool (discrete colors + legend).
    ncols : int
        Number of columns for the subplot grid (ignored if there is only one sample).
    figsize_per_plot : tuple
        Size (width, height) allocated per subplot.
    markersize : float
        Marker size (passed as `s` to ax.scatter).
    cmap : str
        Colormap name.
    shared_scale : bool
        If True, uses a common vmin/vmax (numeric) or a common category->color
        mapping (categorical) across all samples, and a single shared
        colorbar/legend is shown instead of one per subplot.
    show_axis : bool
        If False, hides the axes (ticks, labels, frame).
    show_colorbar : bool
        If True, adds a colorbar/legend (shared if `shared_scale`, else per subplot).
    show : bool
        If True, displays the figure with plt.show().
    return_fig : bool
        If True, returns the matplotlib Figure object.
    save : str, optional
        File path to save the figure (e.g. "out.png"). None = no saving.
    dpi : int
        Resolution for saving (ignored if save=None).
    alpha : float
        Point transparency, useful for dense scatter clouds.
    **plot_kwargs
        Extra keyword arguments forwarded to ax.scatter (e.g. edgecolors, linewidths).

    Returns
    -------
    fig or None
        The matplotlib figure if return_fig=True, otherwise None.
    """
    n_samples = len(dfs)
    
    # detect categorical / non-numeric columns
    sample_col = next(iter(dfs.values()))[column]
    is_categorical = (
        isinstance(sample_col.dtype, pd.CategoricalDtype)
        or sample_col.dtype == object
        or sample_col.dtype == bool
    )

    vmin = vmax = None
    categories = None
    color_map = None

    if is_categorical:
        if shared_scale:
            if isinstance(sample_col.dtype, pd.CategoricalDtype) and sample_col.cat.ordered:
                categories = list(sample_col.cat.categories)
            else:
                cats = set()
                for df in dfs.values():
                    cats.update(df[column].dropna().unique().tolist())
                categories = sorted(cats, key=str)
        cmap_obj = plt.get_cmap(cmap, max(len(categories), 1) if categories else None)
    else:
        if shared_scale:
            vmin = min(df[column].min() for df in dfs.values())
            vmax = max(df[column].max() for df in dfs.values())

    # legend per-subplot only makes sense when scales/categories differ per sample
    legend_per_ax = show_colorbar and not shared_scale

    axes = []
    legend_handles = None
    def _scatter(ax, df, name):
        nonlocal color_map, legend_handles
        centroids = df.geometry.centroid
        vals = df[column]

        if is_categorical:
            local_categories = categories if categories is not None else sorted(
                vals.dropna().unique().tolist(), key=str
            )
            local_cmap = plt.get_cmap(cmap, max(len(local_categories), 1))
            local_color_map = {cat: local_cmap(i) for i, cat in enumerate(local_categories)}
            default_color = (0.8, 0.8, 0.8, 1.0)  # NaN -> light grey

            # avoid Categorical.map (breaks on tuple-valued mappers); map on plain
            # object values instead
            colors = np.array([
                local_color_map.get(v, default_color) if pd.notna(v) else default_color
                for v in vals.astype(object)
            ])

            sc = ax.scatter(
                centroids.x, centroids.y,
                c=colors, s=markersize, alpha=alpha, linewidths=0,
                **plot_kwargs,
            )
            if shared_scale:
                color_map = local_color_map
            if legend_per_ax:
                handles = [
                    mpatches.Patch(color=local_color_map[cat], label=str(cat))
                    for cat in local_categories
                ]
                ax.legend(handles=handles, fontsize=7, loc="best", frameon=False)
            else:
                legend_handles = [
                    mpatches.Patch(color=local_color_map[cat], label=str(cat))
                    for cat in local_categories
                ]
        else:
            sc = ax.scatter(
                centroids.x, centroids.y,
                c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                s=markersize, alpha=alpha, linewidths=0,
                **plot_kwargs,
            )
            if legend_per_ax:
                plt.colorbar(sc, ax=ax, shrink=0.6, label=column)

        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")

        if return_y_axis:
            ax.invert_yaxis()

        if not show_axis:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        return sc


    if n_samples == 1:
        name, df = next(iter(dfs.items()))
        fig, ax = plt.subplots(figsize=figsize_per_plot)
        _scatter(ax, df, name)
        axes.append(ax)

    else:
        nrows = n_samples // ncols + (n_samples % ncols > 0)

        fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
        plt.subplots_adjust(hspace=0.5, wspace=0.35)

        for i, (name, df) in enumerate(dfs.items(), start=1):
            ax = plt.subplot(nrows, ncols, i)
            _scatter(ax, df, name)
            axes.append(ax)

    if shared_scale and show_colorbar:
        if is_categorical:
            fig.legend(
                handles=legend_handles, title=column,
                loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
            )
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            fig.colorbar(sm, ax=axes, shrink=0.6, label=column)

    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    if return_fig:
        return fig
    

def _plot_single_shape(
    ax, polygon, centerline, name, linewidth, show_axis,
    plot_interiors, color_shape, color_line, origin,
):
    """Helper interne : trace une polygon + centerline optionnelle sur un ax donné."""
    x, y = polygon.exterior.xy
    ax.plot(x, y, c=color_shape, linewidth=linewidth)

    if plot_interiors:
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, c=color_shape, linewidth=linewidth)

    if centerline is not None:
        x, y = centerline.xy
        ax.plot(
            x, y,
            "-", c=color_line, linewidth=linewidth,
        )

    ax.set_title(name, fontsize=20)
    ax.set_aspect("equal")

    if origin == "upper":
        ax.invert_yaxis()

    if not show_axis:
        ax.axis("off")


def outlines( # or shapes
    shapes: dict,
    centerlines: dict = None,
    ncols: int = 4,
    figsize_per_plot: tuple = (5, 5),
    linewidth: float = 1.5,
    show_axis: bool = False,
    plot_interiors: bool = False,
    color_shape: str = "black",
    color_line: str = "red",
    origin: str = "upper",  # "upper" ou "lower"
    show: bool = True,
    return_fig: bool = False,
    save: str | None = None,
    dpi: int = 300
):
    """
    Plot shapely polygons stored in a dictionary, with optional centerlines.

    Parameters
    ----------
    shapes : dict
        Dictionary {shape_name: shapely.Polygon}.
    centerlines : dict, optional
        Dictionary {shape_name: shapely.LineString}. Keys should match
        (fully or partially) those of `shapes`. If a shape has no associated
        centerline, it is simply plotted without a line.
    ncols : int
        Number of columns for the subplot grid (ignored if there is only one shape).
    figsize_per_plot : tuple
        Size (width, height) allocated per subplot.
    linewidth : float
        Line width.
    show_axis : bool
        If False, hides the axes (ticks, labels, frame).
    plot_interiors : bool
        If True, also plots the holes (interiors) of the polygons, separately
        from the exterior to avoid spurious connecting lines. False by default.
    color_shape : str
        Color of the shape outline (exterior + interiors).
    color_line : str
        Color of the centerline.
    origin : str
        "upper" (défaut) place l'origine en haut à gauche (convention image,
        comme imshow avec origin='upper'). "lower" garde le repère cartésien
        standard (origine en bas à gauche).
    show : bool
        If True, displays the figure with plt.show().
    return_fig : bool
        If True, returns the matplotlib Figure object.
    save : str, optional
        File path to save the figure (e.g. "out.png"). None = no saving.
    dpi : int
        Resolution for saving (ignored if save=None).

    Returns
    -------
    fig or None
        The matplotlib figure if return_fig=True, otherwise None.
    """
    if centerlines is None:
        centerlines = {}

    if origin not in ("upper", "lower"):
        raise ValueError(f"origin must be 'upper' or 'lower', got {origin!r}")
    
    n_shapes = len(shapes)

    if n_shapes == 1:
        name, polygon = next(iter(shapes.items()))
        fig, ax = plt.subplots(figsize=figsize_per_plot)
        _plot_single_shape(
            ax, polygon, centerlines.get(name), name, linewidth, show_axis,
            plot_interiors, color_shape, color_line, origin,
        )
       
    else:
        nrows = n_shapes // ncols + (n_shapes % ncols > 0)

        fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
        plt.subplots_adjust(hspace=0.5, wspace=0.25)

        for i, (name, polygon) in enumerate(shapes.items(), start=1):
            ax = plt.subplot(nrows, ncols, i)
            _plot_single_shape(
                ax, polygon, centerlines.get(name), name, linewidth, show_axis,
                plot_interiors, color_shape, color_line, origin,
            )

    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    if return_fig:
        return fig


def check_centerlines_orientation(
    centerlines: dict[str, LineString],
    ncols: int = 4,
    figsize_per_ax: tuple[float, float] = (3.5, 3.5),
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



def plot_shapes(shapes: dict, ncols: int = 4):
    """
    Plot shapely polygons stored in a dictionary.

    Parameters
    ----------
    shapes : dict
        Dictionary {shape_name: shapely.Polygon}
    ncols : int
        Number of columns for subplot grid.

    Returns
    -------
        to be completed  
    """
    n_shapes = len(shapes)
    nrows = n_shapes // ncols + (n_shapes % ncols > 0)

    plt.figure(figsize=(5 * ncols, 5 * nrows))
    plt.subplots_adjust(hspace=0.5, wspace=0.25)

    for i, (name, polygon) in enumerate(shapes.items(), start=1):
        x, y = polygon.exterior.xy
        ax = plt.subplot(nrows, ncols, i)
        ax.plot(x, y, linewidth=2)
        ax.set_title(name, fontsize=20)
        # ax.set_aspect('equal')
        # ax.axis('off')          # Turn off axes ticks and labels
        # ax.set_frame_on(True)  # Remove the box around the plot
    plt.show()


def plot_shape(
    shape, 
    title = "",
    only_max = False,
    plot_points = False,
    return_y_axis = True,
    figsize=(5,5),
) -> None:  
    """
    Plot one shape
    """
    _, ax = plt.subplots(figsize=figsize)
    
    if (shape.geom_type == 'MultiPolygon') &  (only_max):
        shape = max(shape.geoms, key=lambda g: g.area)
    
    if shape.geom_type == 'Polygon':
        x, y = shape.exterior.xy
        ax.plot(x, y, color="blue")
        ax.fill(x, y, alpha=0.3, color="lightblue")

        if plot_points:
            ax.scatter(x, y, c="blue", s=5)
    elif shape.geom_type == 'MultiPolygon':
        for pol in shape.geoms:
            x, y = pol.exterior.xy
            ax.plot(x, y, color="blue")
            ax.fill(x, y, alpha=0.3, color="lightblue")

            if plot_points:
                ax.scatter(x, y, c="blue", s=5)

    if return_y_axis:
        ax.invert_yaxis()
    
    plt.title(title)
    plt.show()



# def plot_shapes(
#     sdata: sd.SpatialData,
#     group_lst: tuple = None,  # the cell types to consider
#     shapes_lst: tuple = None,  # the shapes to plot
#     color_key: str = "celltype_spatial",
#     shape_key: str = "arteries",
#     target_coordinates: str = "microns",
#     figsize: tuple = (12, 6),
#     palette: tuple = None,
#     save: bool = False,
# ):
#     """Plot list of shapes

#     Parameters
#     ----------
#     sdata
#         SpatialData object obtained by tl.get_sdata_polygon()
#     group_lst
#         group list to consider (related to label_obs_key)
#     shapes_lst
#         shapes list to plot
#     color_key
#         label_key in sdata['table'].obs to consider
#     shape_key
#         SpatialData shape element to consider
#     palette
#         dictionary of colors to use
#     target_coordinates
#         target_coordinates system of sdata object
#     figsize
#         figure size
#     save
#         wether or not to save the figure

#     """
#     region_key = sdata['table'].uns["spatialdata_attrs"]["region"]
#     my_shapes = {region_key: sdata[region_key], shape_key: sdata[shape_key]}
#     my_tables = {"table": sdata["table"]}
#     sdata2 = SpatialData(shapes=my_shapes, tables=my_tables)

#     fig, axs = plt.subplots(ncols=len(shapes_lst), nrows=1, figsize=figsize)
#     for i in range(0, len(shapes_lst)):
#         poly = sdata2[shape_key][sdata2[shape_key].name == shapes_lst[i]].geometry.item()
#         sdata3 = sd.polygon_query(
#             sdata2,
#             poly,
#             target_coordinate_system=target_coordinates,
#             filter_table=True,
#         )

#         # sdata3.pl.render_images().pl.show(ax=axs[i])
#         if group_lst is None:
#             group_lst = sdata2['table'].obs[color_key].unique().tolist()

#         if palette is not None:
#             mypal = [palette[x] for x in group_lst]
#             sdata3.pl.render_shapes(elements=region_key, color=color_key, groups=group_lst, palette=mypal).pl.show(
#                 ax=axs[i]
#             )
#         else:
#             sdata3.pl.render_shapes(elements=region_key, color=color_key, groups=group_lst).pl.show(ax=axs[i])

#         axs[i].set_title(shapes_lst[i])
#         if i < len(shapes_lst) - 1:
#             axs[i].get_legend().remove()

#     plt.tight_layout()

