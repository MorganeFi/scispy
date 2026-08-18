"""Generic geometry helpers used across Sparty's spatial-niche reconstruction
tools (alpha-shape smoothing/bridging, point-cloud sampling, local scale
estimation).

These functions are intentionally decoupled from AnnData/SpatialData objects
where possible, so they can be unit-tested and reused independently.
"""

from __future__ import annotations

import numpy as np
import shapely
from anndata import AnnData


# --------------------------------------------------------------------------
# Local scale estimation
# --------------------------------------------------------------------------

def local_mean_nn_distance(
    adata: AnnData,
    mask: np.ndarray | None = None,
    distances_key: str = "spatial_distances",
) -> float:
    """Estimate the mean nearest-neighbor distance for a (sub)population of cells.

    This is used as the natural length scale for smoothing/bridging operations,
    instead of an arbitrary fixed buffer. Computing it on the relevant subset
    (e.g. cortex cells of a single sample) rather than the full ``adata``
    avoids mixing densities across regions/samples.

    Parameters
    ----------
    adata
        AnnData object with a spatial neighbors graph already computed
        (e.g. via ``squidpy.gr.spatial_neighbors``).
    mask
        Optional boolean mask (over ``adata.obs_names``) to restrict the
        distance matrix to a subset of cells (e.g. a single sample or region).
        If None, uses all cells in ``adata``.
    distances_key
        Key in ``adata.obsp`` holding the sparse spatial distances matrix.

    Returns
    -------
    float
        Mean nonzero nearest-neighbor distance for the subset.
    """
    dists = adata.obsp[distances_key]
    if mask is not None:
        dists = dists[mask][:, mask]
    data = dists.data
    if data.size == 0:
        raise ValueError(
            "No spatial distances found for this subset. "
            "Check that spatial_neighbors was computed and that the mask is not empty."
        )
    return float(data.mean())


def equivalent_radius(polygon: shapely.Polygon) -> float:
    """Radius of a disk with the same area as ``polygon``.

    Useful as a scale-free reference for tolerances (simplify, extend distance)
    that should scale with the overall size of the shape.
    """
    return float(np.sqrt(polygon.area / np.pi))


# --------------------------------------------------------------------------
# Smoothing / bridging
# --------------------------------------------------------------------------

def smooth_polygon(
    polygon: shapely.Polygon,
    buffer_dist: float,
    simplify_tol: float | None = None,
    simplify_frac: float = 0.5,
) -> shapely.Polygon:
    """Smooth a polygon's boundary via morphological closing + simplification.

    Parameters
    ----------
    polygon
        Input polygon (typically the output of an alpha-shape).
    buffer_dist
        Distance used for the closing operation (``buffer(+d).buffer(-d)``).
        Should reflect the local point-cloud resolution
        (see :func:`local_mean_nn_distance`), not the overall shape size.
    simplify_tol
        Tolerance passed to ``shapely.simplify``. If None, defaults to
        ``buffer_dist * simplify_frac``.
    simplify_frac
        Fraction of ``buffer_dist`` used as the simplify tolerance when
        ``simplify_tol`` is not given.

    Returns
    -------
    shapely.Polygon
        Smoothed polygon. If the closing operation yields a MultiPolygon
        (disjoint parts), the largest part is returned with a warning-free
        fallback -- callers who need bridging should call
        :func:`bridge_multipolygon` beforehand.
    """
    if simplify_tol is None:
        simplify_tol = buffer_dist * simplify_frac

    smoothed = polygon.buffer(buffer_dist).buffer(-buffer_dist)

    if isinstance(smoothed, shapely.MultiPolygon):
        smoothed = max(smoothed.geoms, key=lambda g: g.area)

    smoothed = smoothed.simplify(tolerance=simplify_tol, preserve_topology=True)
    return smoothed


def bridge_multipolygon(
    multipolygon: shapely.MultiPolygon | shapely.Polygon,
    margin_frac: float = 1.1,
    max_bridge_dist: float | None = None,
) -> shapely.Polygon | shapely.MultiPolygon:
    """Reconnect disjoint parts of a MultiPolygon (e.g. CA/DG separated by a
    small segmentation gap) using a closing sized to the actual gap distance.

    Parameters
    ----------
    multipolygon
        Polygon or MultiPolygon to process. If a Polygon is passed, it is
        returned unchanged.
    margin_frac
        Multiplicative margin applied to the minimal gap distance found
        between parts, to make sure the bridge fully closes the gap.
    max_bridge_dist
        Optional safety cap on the bridging buffer distance. If the minimal
        gap between parts exceeds this value, bridging is skipped and the
        MultiPolygon is returned as-is (parts are assumed genuinely separate,
        not a segmentation artifact).

    Returns
    -------
    shapely.Polygon | shapely.MultiPolygon
        A single Polygon if bridging succeeded (or input was already a
        Polygon), otherwise the original MultiPolygon.
    """
    if isinstance(multipolygon, shapely.Polygon):
        return multipolygon

    parts = list(multipolygon.geoms)
    if len(parts) < 2:
        return parts[0] if parts else multipolygon

    min_gap = float("inf")
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            d = parts[i].distance(parts[j])
            if d < min_gap:
                min_gap = d

    if max_bridge_dist is not None and min_gap > max_bridge_dist:
        return multipolygon

    bridge_dist = (min_gap / 2) * margin_frac
    merged = multipolygon.buffer(bridge_dist).buffer(-bridge_dist)
    return merged


def auto_process_shape(
    polygon: shapely.Polygon | shapely.MultiPolygon,
    adata: AnnData,
    mask: np.ndarray | None = None,
    distances_key: str = "spatial_distances",
    buffer_factor: float = 3.0,
    simplify_frac: float = 0.5,
    bridge_margin: float = 1.1,
    max_bridge_dist: float | None = None,
) -> shapely.Polygon:
    """Full shape post-processing pipeline: bridge disjoint parts, then smooth.

    Buffer/tolerance are derived from the local mean nearest-neighbor
    distance of the cells that produced the shape, so the same function
    generalizes across samples of different density/scale without manual
    per-sample tuning.

    Parameters
    ----------
    polygon
        Alpha-shape output (Polygon or MultiPolygon).
    adata
        AnnData used to estimate the local point-cloud resolution.
    mask
        Boolean mask restricting ``adata`` to the cells used for this shape
        (e.g. a single sample's cortex cells). Strongly recommended when
        ``adata`` contains multiple samples/regions.
    distances_key
        Key in ``adata.obsp`` for the spatial distances matrix.
    buffer_factor
        Multiplier applied to the mean NN distance to get the closing buffer.
    simplify_frac
        Fraction of the buffer distance used for boundary simplification.
    bridge_margin
        Margin factor for :func:`bridge_multipolygon`.
    max_bridge_dist
        Safety cap for bridging distance, see :func:`bridge_multipolygon`.

    Returns
    -------
    shapely.Polygon
        Smoothed, single-part polygon ready for centerline extraction.
    """
    mean_nn_dist = local_mean_nn_distance(adata, mask=mask, distances_key=distances_key)
    buffer_dist = buffer_factor * mean_nn_dist

    if isinstance(polygon, shapely.MultiPolygon):
        polygon = bridge_multipolygon(
            polygon, margin_frac=bridge_margin, max_bridge_dist=max_bridge_dist
        )
        if isinstance(polygon, shapely.MultiPolygon):
            # Bridging did not fully merge the parts (gap too large / capped):
            # fall back to the largest part so downstream code always gets a Polygon.
            polygon = max(polygon.geoms, key=lambda g: g.area)

    polygon = smooth_polygon(
        polygon, buffer_dist=buffer_dist, simplify_frac=simplify_frac
    )
    return polygon


# --------------------------------------------------------------------------
# Point cloud sampling (faster alternative to full rasterization)
# --------------------------------------------------------------------------

def sample_points_in_polygon(
    polygon: shapely.Polygon,
    n_points: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Vectorized rejection sampling of ``n_points`` uniform points inside a polygon.

    Much faster than rasterizing the full polygon (:func:`shapeToImg`) when
    only a representative point cloud is needed (e.g. as KMeans input).

    Parameters
    ----------
    polygon
        Polygon to sample from.
    n_points
        Number of points to return.
    random_state
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 2)``.
    """
    rng = np.random.default_rng(random_state)
    minx, miny, maxx, maxy = polygon.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    area_ratio = bbox_area / polygon.area

    points = np.empty((0, 2))
    while points.shape[0] < n_points:
        remaining = n_points - points.shape[0]
        batch_size = int(remaining * area_ratio * 1.3) + 20

        xs = rng.uniform(minx, maxx, batch_size)
        ys = rng.uniform(miny, maxy, batch_size)

        mask = shapely.contains_xy(polygon, xs, ys)
        points = np.vstack([points, np.column_stack((xs[mask], ys[mask]))])

    return points[:n_points]

    