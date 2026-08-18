from anndata import AnnData
from scipy.sparse import csr_matrix
import networkx as nx 
import squidpy as sq
import numpy as np 
import shapely
from scipy.spatial import Delaunay
import spatialdata as sd

import math
import warnings

from ._geometry_utils import auto_process_shape


#  https://web.archive.org/web/20201013181320/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
#     ; https://gist.github.com/dwyerk/10561690 ; https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889#file-concave_hulls-ipynb 
#     ; https://github.com/mlichter2/concavity
# https://web.archive.org/web/20201013181320/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/


def _delaunay_triangulation(points: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Delaunay triangulation once, for reuse across alpha values.
 
    Parameters
    ----------
    points
        List of cell centroids.
 
    Returns
    -------
    coords
        (n, 2) array of point coordinates.
    simplices
        (m, 3) array of triangle vertex indices.
    circum_r
        (m,) array of circumcircle radii, one per triangle.
    """
    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
 
    circum_r = np.empty(len(tri.simplices))
    for k, (ia, ib, ic) in enumerate(tri.simplices):
        # ia, ib, ic = indices of corner points of the triangle
        pa, pb, pc = coords[ia], coords[ib], coords[ic]
 
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
 
        s = (a + b + c) / 2.0                               # Semiperimeter of triangle
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))   # Area of triangle by Heron's formula
        circum_r[k] = a * b * c / (4.0 * area)              # radius of circumcircle
         
    return coords, tri.simplices, circum_r
 
 
def _polygon_from_triangulation(
    coords: np.ndarray,
    simplices: np.ndarray,
    circum_r: np.ndarray,
    alpha: float,
) -> shapely.Polygon | shapely.MultiPolygon:
    """Union of all triangles with circumcircle radius < alpha (the alpha shape)."""
    mask = circum_r < alpha
    tri_idx = simplices[mask]
    if len(tri_idx) == 0:
        return shapely.Polygon()
 
    tri_coords = coords[tri_idx]  # (m, 3, 2)
    closed_rings = np.concatenate([tri_coords, tri_coords[:, :1, :]], axis=1)  # (m, 4, 2) 
    # concatenate 3 points of triangle + first point triangle A-B-C-A => closed_rings
    triangle_polys = shapely.polygons(closed_rings)  # vectorized, no Python-level loop
    return shapely.unary_union(triangle_polys)

 
def find_optimal_alpha(points, target_count: int, upper_bound: int = 1000):
    """Find the smallest alpha (assuming f is increasing) such that the
    alpha shape covers exactly ``target_count`` points, via dichotomic search.
 
    The Delaunay triangulation of ``points`` is computed only once and
    reused for every alpha tested during the search, instead of being
    recomputed from scratch at each step -- the triangulation is by far the
    most expensive part and does not depend on alpha. Point-coverage
    counting also builds the point geometries once up front rather than on
    every iteration. To inspect a single alpha value manually (e.g. for
    exploration/debugging), use :func:`alpha_shape` directly.
 
    Parameters
    ----------
    points
        Point cloud to fit.
    target_count
        Target cell count.
    upper_bound
        Upper bound for the search range.
 
    Returns
    -------
    tuple | None
        ``(polygon, alpha, cell_count)`` for the best alpha found, or None
        if the search range ``[0, upper_bound]`` was empty.
    """
    if len(points) < 4:
        warnings.warn("Warning Message: Less than 4 points, simply compute the convex hull")
        pol = shapely.MultiPoint(points).convex_hull
        return pol, 0, len(points)
 
    coords, simplices, circum_r = _delaunay_triangulation(points)
    point_geoms = shapely.points(coords[:, 0], coords[:, 1])
 
    left, right = 1, upper_bound
    best = None
 
    while left <= right:
        mid = (left + right) // 2
 
        shape = _polygon_from_triangulation(coords, simplices, circum_r, mid)
        pol = max(shape.geoms, key=lambda g: g.area) if shape.geom_type == "MultiPolygon" else shape
        val = int(shapely.covers(pol, point_geoms).sum())
 
        if val == target_count:
            best = (pol, mid, val)
            right = mid - 1
        elif val < target_count:
            best = (pol, mid, val)
            left = mid + 1
        else:
            right = mid - 1
    return best
 

def remove_long_links(
    adata: AnnData,
    distance_percentile: float = 99.0,
    connectivity_key: str | None = None,
    distances_key: str | None = None,
    neighs_key: str | None = None,
    copy: bool = False,
) -> tuple[csr_matrix, csr_matrix] | None:
    """Remove links between cells at a distance greater than a percentile of
    all positive distances. Designed for data with generic coordinates.

    Parameters
    ----------
    adata
        Annotated data object.
    distance_percentile
        Percentile of the distances between cells above which links are trimmed.
    connectivity_key
        Key in ``adata.obsp`` for spatial connectivities.
    distances_key
        Key in ``adata.obsp`` for spatial distances.
    neighs_key
        Key in ``adata.uns`` for the spatial_neighbors parameters.
    copy
        If True, return the trimmed matrices instead of modifying ``adata`` in place.

    Returns
    -------
    tuple[csr_matrix, csr_matrix] | None
        If ``copy=True``, returns ``(connectivities, distances)``.
        Otherwise modifies ``adata`` in place and returns None.
    """
    conns, dists = adata.obsp[connectivity_key], adata.obsp[distances_key]

    if copy:
        conns, dists = conns.copy(), dists.copy()

    threshold = np.percentile(np.array(dists[dists != 0]).squeeze(), distance_percentile)
    conns[dists > threshold] = 0
    dists[dists > threshold] = 0

    conns.eliminate_zeros()
    dists.eliminate_zeros()

    if copy:
        return conns, dists

    adata.uns[neighs_key]["params"]["radius"] = threshold
    return None


def alpha_shape_optimal(
    sdata: sd.SpatialData,
    group_by: str,
    groups: int | str | list,
    table_key: str = "table",
    cell_id: str = "cell_id",
    convex_hull: bool = False,
    only_shape: bool = True,
    percentile: float = 99.0,
    region: str = "region",
    # shape_key: str = "cell_boundaries",
    connectivity_key: str = "spatial_connectivities",
    distances_key: str = "spatial_distances",
    neighs_key: str = "spatial_neighbors",
    option: int = 1,
    #option 1 = remove long link et largest_cc
    #  option 2 = len(list_points) * percentile / 100
    smooth: bool = True,
    buffer_factor: float = 3.0,
    simplify_frac: float = 0.5,
    bridge_margin: float = 1.1,
    max_bridge_dist: float | None = None,
) -> shapely.Polygon | tuple:
    """Create a shape from a point cloud using the alpha-shape algorithm.

    Parameters
    ----------
    sdata
        SpatialData object containing the table and shapes.
    group_by
        Column in ``adata.obs`` used to select the group of cells
        (e.g. a spatial domain / niche label).
    groups
        Value(s) of ``group_by`` to include (e.g. ``"cortex"``).
    table_key
        Key of the table in ``sdata``.
    cell_id
        Column in ``adata.obs`` holding the cell id matching
        ``sdata[shape_key]`` index.
    convex_hull
        If True, return the convex hull instead of the alpha-shape.
    only_shape
        If True, return only the polygon. Otherwise also return
        ``(alpha, alpha_cells)``.
    percentile
        Percentile threshold used by :func:`remove_long_links` to trim the
        Delaunay graph before taking the largest connected component.
    region
        Key in ``adata.uns['spatialdata_attrs']`` holding the shape element name.
    connectivity_key, distances_key, neighs_key
        Keys in ``adata.obsp``/``adata.uns`` for the spatial neighbors graph.
    option
        1: target cell count = size of the largest connected component after
           trimming long Delaunay links (recommended, default).
        2: target cell count = percentile fraction of all group cells.
    smooth
        If True (default), automatically bridge disjoint parts and smooth
        the resulting polygon's boundary using the local point-cloud
        resolution (see :func:`_geometry_utils.auto_process_shape`).
        Ignored when ``convex_hull=True``, since a convex hull has no
        notches, branches, or disjoint parts to correct.
    buffer_factor, simplify_frac, bridge_margin, max_bridge_dist
        Passed through to :func:`_geometry_utils.auto_process_shape`.

    Returns
    -------
    shapely.Polygon | tuple
        The (optionally smoothed) polygon, or ``(polygon, alpha, alpha_cells)``
        if ``only_shape=False``.
    """
    if not isinstance(groups, list):
        groups = [groups]

    adata = sdata[table_key][sdata[table_key].obs[group_by].isin(groups)].copy()
    shape_key = adata.uns["spatialdata_attrs"][region]
    if isinstance(shape_key, list):
        shape_key = shape_key[0]

    largest_cc = None
    if option == 1 or convex_hull:
        print(f"Remove long links > {percentile} percentile...")
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
        remove_long_links(
            adata,
            distance_percentile=percentile,
            connectivity_key=connectivity_key,
            distances_key=distances_key,
            neighs_key=neighs_key,
        )
        G = nx.from_numpy_array(adata.obsp[connectivity_key].todense())
        largest_cc = max(nx.connected_components(G), key=len)

    if convex_hull:
        print("Convex hull...")
        sub_cells = adata[list(largest_cc), :].obs[cell_id].values
        list_points = [
            poly.centroid for poly in sdata[shape_key].loc[sub_cells, :].geometry.values
        ]
        pol = shapely.convex_hull(shapely.MultiPoint(list_points))
        # A convex hull has no notches/branches/disjoint parts by construction,
        # so bridging and boundary smoothing are unnecessary here regardless
        # of the `smooth` flag -- applying them would only slightly distort
        # an already-clean shape for no benefit.
        return pol

    # --- Non-convex-hull path -------------------------------------------
    # Both the target cell count (nb_cells) and the point cloud searched
    # (list_points) must come from the SAME cell population, otherwise the
    # dichotomic search in find_optimal_alpha optimizes against an
    # inconsistent reference.
    if option == 1:
        sub_cells = adata[list(largest_cc), :].obs[cell_id].values
        nb_cells = len(largest_cc)
        print(f"Target cell count (largest connected component): {nb_cells}")
    elif option == 2:
        sub_cells = adata.obs[cell_id].values
        nb_cells = int(len(sub_cells) * percentile / 100)
        print(f"Target cell count (percentile of group cells): {nb_cells}")
    else:
        raise ValueError(f"Unknown option={option!r}, expected 1 or 2.")

    list_points = [
        poly.centroid for poly in sdata[shape_key].loc[sub_cells, :].geometry.values
    ]

    pol, alpha, alpha_cells = find_optimal_alpha(
        points=list_points, target_count=nb_cells, upper_bound=1000
    )
    print(f"alpha={alpha}: {alpha_cells} cells")

    if smooth:
        mask = adata.obs[cell_id].isin(sub_cells).values
        pol = auto_process_shape(
            pol,
            adata=adata,
            mask=mask,
            distances_key=distances_key,
            buffer_factor=buffer_factor,
            simplify_frac=simplify_frac,
            bridge_margin=bridge_margin,
            max_bridge_dist=max_bridge_dist,
        )

    if only_shape:
        return pol
    return pol, alpha, alpha_cells




def alpha_shape(
    points: list,
    alpha: float,
    only_shape: bool = True,
) -> tuple | shapely.Polygon | shapely.MultiPolygon:
    """Compute the alpha shape of a set of points.
 
    https://web.archive.org/web/20201013181320/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
 
    Parameters
    ----------
    points
        List of cell centroids.
    alpha
        Value to influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large, and you
        lose everything.
    only_shape
        By default return only the shape. If False, also return the
        edge_points (lines) and all_circum_r (radii of the circumcircles).
 
    Returns
    -------
    By default return only the shape (Polygon or MultiPolygon).
    If ``only_shape=False``, return a tuple with the shape, all the lines
    used to compute it, and all circumcircle radii.
 
    Notes
    -----
    For repeated calls on the same point cloud with different alpha values
    (e.g. inside a search loop), prefer computing the triangulation once
    with :func:`_delaunay_triangulation` and reusing
    :func:`_polygon_from_triangulation` directly -- see
    :func:`find_optimal_alpha` for an example that avoids recomputing the
    triangulation at every step.
    """
    if len(points) < 4:
        warnings.warn("Warning Message: Less than 4 points, simply compute the convex hull")
        return shapely.MultiPoint(points).convex_hull
 
    coords, simplices, circum_r = _delaunay_triangulation(points)
    shape = _polygon_from_triangulation(coords, simplices, circum_r, alpha)
 
    if only_shape:
        return shape
 
    mask = circum_r < alpha
    edge_points = []
    seen = set()
    for ia, ib, ic in simplices[mask]:
        for i, j in ((ia, ib), (ib, ic), (ic, ia)):
            if (i, j) not in seen and (j, i) not in seen:
                seen.add((i, j))
                edge_points.append(coords[[i, j]])
 
    return shape, edge_points, circum_r.tolist()


# def function_alpha_counts_cells(points: list, alpha: float):
#     """Compute the alpha shape for a given alpha and count how many points it covers."""
#     shapes = alpha_shape(points=points, alpha=alpha)
#     if shapes.geom_type == "MultiPolygon":
#         pol = max(shapes.geoms, key=lambda g: g.area)
#     else:
#         pol = shapes
#     count = shapely.covers(pol, points).sum()
#     return pol, count



# def find_optimal_alpha(points, target_count: int, upper_bound: int = 1000):
#     """Find the smallest alpha (assuming f is increasing) such that
#     ``function_alpha_counts_cells(points, alpha) == target_count``, via dichotomic search.

#     Parameters
#     ----------
#     points
#         Point cloud to fit.
#     threshold
#         Target cell count.
#     upper_bound
#         Upper bound for the search range.

#     Returns
#     -------
#     tuple | None
#         ``(polygon, alpha, cell_count)`` for the best alpha found, or None
#         if the search range ``[0, upper_bound]`` was empty.
#     """
#     left, right = 1, upper_bound
#     best = None

#     while left <= right:
#         mid = (left + right) // 2
#         pol, val = function_alpha_counts_cells(points=points, alpha=mid)

#         if val == target_count:
#             best = (pol, mid, val)
#             right = mid - 1
#         elif val < target_count:
#             best = (pol, mid, val)
#             left = mid + 1
#         else:
#             right = mid - 1
#     return best


### OLD

# def alpha_shape_optimal_old(
#     sdata: sd.SpatialData,
#     group_by: str,
#     groups: int|str|list,
#     table_key: str = 'table',
#     cell_id: str = 'cell_id',
#     convex_hull: bool = False,
#     only_shape: bool = True,
#     percentile: float = 99.0,
#     region: str = 'region',
#     connectivity_key: str ='spatial_connectivities', 
#     distances_key: str ='spatial_distances',
#     neighs_key: str ='spatial_neighbors',
#     option = 1, #option 1 = remove long link et largest_cc
#     #  option 2 = len(list_points) * percentile / 100
# ) -> shapely.Polygon | tuple:  

#     if type(groups) != list:
#         groups = [groups]
#     adata = sdata[table_key][sdata[table_key].obs[group_by].isin(groups)].copy()
#     shape_key = adata.uns['spatialdata_attrs'][region]
    
#     if type(shape_key) == list:
#         # print(len(shape_key))
#         shape_key = shape_key[0]
    
#     if (option == 1) | (convex_hull):
#         print(f'Remove long links > {percentile} percentile...')
#         sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)
#         remove_long_links(
#             adata,
#             distance_percentile = percentile,
#             connectivity_key=connectivity_key, 
#             distances_key=distances_key,
#             neighs_key=neighs_key)
#         G = nx.from_numpy_array(adata.obsp[connectivity_key].todense())
#         largest_cc = max(nx.connected_components(G), key=len)
    

#     if convex_hull:
#         print('Convexe hull...')
#         sub_cells = adata[list(largest_cc),].obs[cell_id].values
#         list_points = [poly.centroid for poly in sdata[shape_key].loc[sub_cells,].geometry.values]
#         # print(len(list_points))
#         pol = shapely.convex_hull(shapely.MultiPoint(list_points))
#     else:
#         if option == 1:
#             nb_cells = len(largest_cc)
#             print(nb_cells)
#         ### MAYBE TO REMOVE OPTION 2 
#         ### recall and precision lower than option 1 ????
#         elif option == 2:    
#             print('Option 2 avec percent of list_points')
#             nb_cells = int(len(list_points) * percentile / 100)
#             print(nb_cells)
#         #######################################
#         sub_cells = adata.obs[cell_id].values
#         list_points = [poly.centroid for poly in sdata[shape_key].loc[sub_cells].geometry.values]

#         pol, alpha, alpha_cells = find_optimal_alpha(
    #     points=list_points, target_count=nb_cells, upper_bound=1000
    # )
#         print(f'{alpha}: {alpha_cells} cells')

#     if convex_hull or only_shape:
#         return pol
#     else: 
#         return pol, alpha, alpha_cells


# def alpha_shape(
#     points: list,
#     alpha: float,
#     only_shape: bool = True,
# ) -> tuple | shapely.Polygon | shapely.MultiPolygon:
#     """Compute the alpha shape of a set of points.

#     https://web.archive.org/web/20201013181320/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

#     Parameters
#     ----------
#     points
#         List of cell centroids.
#     alpha
#         Value to influence the gooeyness of the border. Smaller numbers
#         don't fall inward as much as larger numbers. Too large, and you
#         lose everything.
#     only_shape
#         By default return only the shape. If False, also return the
#         edge_points (lines) and all_circum_r (radii of the circumcircles).

#     Returns
#     -------
#     By default return only the shape (Polygon or MultiPolygon).
#     If ``only_shape=False``, return a tuple with the shape, all the lines
#     used to compute it, and all circumcircle radii.
#     """
#     if len(points) < 4:
#         warnings.warn("Warning Message: Less than 4 points, simply compute the convex hull")
#         return shapely.MultiPoint(points).convex_hull

#     def add_edge(edges, edge_points, coords, i, j):
#         if (i, j) in edges or (j, i) in edges:
#             return             # already added
#         edges.add((i, j))
#         edge_points.append(coords[[i, j]])

#     coords = np.array([point.coords[0] for point in points])
#     tri = Delaunay(coords)

#     edges = set()
#     edge_points = []
#     all_circum_r = []

#     for ia, ib, ic in tri.simplices:
#         # ia, ib, ic = indices of corner points of the triangle
#         pa, pb, pc = coords[ia], coords[ib], coords[ic]

#         # Lengths of sides of triangle
#         a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
#         b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
#         c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

#         s = (a + b + c) / 2.0                               # Semiperimeter of triangle
#         area = math.sqrt(s * (s - a) * (s - b) * (s - c))   # Area of triangle by Heron's formula
#         circum_r = a * b * c / (4.0 * area)                 # radius of circumcircle
#         all_circum_r.append(circum_r)

#         if circum_r < alpha:
#             add_edge(edges, edge_points, coords, ia, ib)
#             add_edge(edges, edge_points, coords, ib, ic)
#             add_edge(edges, edge_points, coords, ic, ia)

#     m = shapely.MultiLineString(edge_points)
#     triangles = list(polygonize(m))

#     if only_shape:
#         return unary_union(triangles)
#     return unary_union(triangles), edge_points, all_circum_r



# def remove_long_links(
#     adata: AnnData,
#     distance_percentile: float = 99.0,
#     connectivity_key: str | None = None,
#     distances_key: str | None = None,
#     neighs_key: str | None = None,
#     copy: bool = False,
# ) -> tuple[csr_matrix, csr_matrix] | None:
#     """
#     Remove links between cells at a distance bigger than a certain percentile of all positive distances.

#     It is designed for data with generic coordinates.

#     Parameters
#     ----------
#     %(adata)s

#     distance_percentile
#         Percentile of the distances between cells over which links are trimmed after the network is built.
#     %(conn_key)s

#     distances_key
#         Key in :attr:`anndata.AnnData.obsp` where spatial distances are stored.
#         Default is: :attr:`anndata.AnnData.obsp` ``['{{Key.obsp.spatial_dist()}}']``.
#     neighs_key
#         Key in :attr:`anndata.AnnData.uns` where the parameters from gr.spatial_neighbors are stored.
#         Default is: :attr:`anndata.AnnData.uns` ``['{{Key.uns.spatial_neighs()}}']``.

#     %(copy)s

#     Returns
#     -------
#     If ``copy = True``, returns a :class:`tuple` with the new spatial connectivities and distances matrices.

#     Otherwise, modifies the ``adata`` with the following keys:
#         - :attr:`anndata.AnnData.obsp` ``['{{connectivity_key}}']`` - the new spatial connectivities.
#         - :attr:`anndata.AnnData.obsp` ``['{{distances_key}}']`` - the new spatial distances.
#         - :attr:`anndata.AnnData.uns`  ``['{{neighs_key}}']`` - :class:`dict` containing parameters.
#     """



