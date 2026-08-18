import scipy
import shapely
import numpy as np
import skimage as ski
from sklearn.cluster import KMeans 
import warnings

from ._geometry_utils import equivalent_radius, sample_points_in_polygon


# 1) shapeToImg
# 2) sortedCentroidToLine
# order_points
# 3) get_angle 
# extendLine 
# 4) addPoints 
# extendLine 
# 5) centerline 

# Centerline extraction for elongated spatial domains (e.g. cortex ribbon),
# via KMeans clustering of the shape's interior + ordered path reconstruction.

# Fixes applied vs. the original implementation:
# - ``centerline``: the search loop used to keep incrementing ``n_clusters``
#   even when :func:`sortedCentroidToLine` failed (returned None), then
#   compared angle patterns between non-consecutive k values, occasionally
#   reporting an arbitrary "best k" far from the true elbow. The loop now
#   tracks the last *valid* k, only compares consecutive k's, and stops after
#   a bounded number of consecutive failures (``max_consecutive_failures``)
#   or once ``max_clusters`` is reached.
# - ``addPoints``: when the extended ray crosses the polygon boundary more
#   than once (e.g. a concave notch near a branch), the closest intersection
#   used to be kept. This can stop the line at a local notch instead of the
#   true tip of the shape. Now the intersection furthest along the extension
#   direction is kept.
# - ``length_max`` (in :func:`sortedCentroidToLine`) and ``distance`` (in
#   :func:`extendLine`/:func:`addPoints`) can now be passed as ``None`` to
#   :func:`centerline`, in which case they are derived automatically from the
#   polygon's own scale (equivalent radius) instead of being fixed constants
#   that don't generalize across samples of different size.
# - Added an optional faster point-cloud input via
#   ``_geometry_utils.sample_points_in_polygon`` instead of full rasterization
#   (:func:`shapeToImg`), controlled by ``use_sampling``/``n_sample_points``.



# --------------------------------------------------------------------------
# Point cloud generation
# --------------------------------------------------------------------------

def shapeToImg(
    polygon: shapely.Polygon,
    micron_to_pixel: float = 1,
    only_position: bool = True,
) -> tuple | np.ndarray:
    """Create a binary image representing the polygon.

    Parameters
    ----------
    polygon
        Polygon to transform.
    micron_to_pixel
        Size of one micron in pixels.
    only_position
        If True, return only the (x, y) positions of pixels inside the shape.
        If False, also return the binary image.

    Returns
    -------
    posImg, or (img, posImg)
        ``posImg``: (x, y) positions of all True pixels.
        ``img``: binary image (only if ``only_position=False``).
    """
    boundary_pixels = np.rint(shapely.get_coordinates(polygon.boundary) / micron_to_pixel)

    x = int(boundary_pixels[:, 0].max()) + 1
    y = int(boundary_pixels[:, 1].max()) + 1

    img = np.zeros((y, x), dtype=bool)
    row, col = ski.draw.polygon(*boundary_pixels.T)
    img[col, row] = 1
    # row = y
    # col = x
    yy, xx = np.where(img)
    posImg = np.column_stack((xx, yy))
    if only_position:
        return posImg
    return img, posImg


# --------------------------------------------------------------------------
# Path ordering (TSP-like greedy walk)
# --------------------------------------------------------------------------

def order_points(dist_matrix, start: int = 0) -> list | None:
    """Greedy nearest-neighbor walk through a distance matrix (approximate TSP path).

    Parameters
    ----------
    dist_matrix
        Square distance matrix between points (``inf`` for forbidden links).
    start
        Index of the starting point.

    Returns
    -------
    list | None
        Ordered list of point indices, or None if not all points could be
        connected (graph disconnected given the ``inf`` entries).
    """
    n_points = len(dist_matrix)
    ordered_points = [start]
    visited = [False] * n_points
    visited[start] = True

    for _ in range(1, n_points):
        min_distance = float("inf")
        next_point = None

        for j in range(n_points):
            if not visited[j] and dist_matrix[start, j] < min_distance:
                min_distance = dist_matrix[start, j]
                next_point = j

        if next_point is None:
            break

        ordered_points.append(next_point)
        visited[next_point] = True
        start = next_point

    if len(ordered_points) < n_points:
        return None
    return ordered_points


def sortedCentroidToLine(
    polygon: shapely.Polygon,
    centroids,
    length_max: float = 5,
) -> shapely.LineString | None:
    """Reorder centroid points into a non-self-intersecting LineString.

    Parameters
    ----------
    polygon
        Polygon the centroids live in.
    centroids
        Array of (x, y) centroid coordinates (e.g. KMeans cluster centers).
    length_max
        Maximum allowed length of a centroid-to-centroid segment lying
        outside the polygon before that link is forbidden. Should scale with
        the polygon/point-cloud resolution -- see ``centerline``'s automatic
        default.

    Returns
    -------
    shapely.LineString | None
        The shortest valid ordered path found, or None if no valid full
        ordering exists (e.g. too many long/forbidden links).
    """
    n = len(centroids)
    min_dist = float("inf")
    min_ordered_line = None

    dist_matrix = scipy.spatial.distance_matrix(centroids, centroids)

    for i in range(n):
        for j in range(i + 1, n):
            line = shapely.LineString([centroids[i], centroids[j]])
            if polygon.boundary.intersects(line):
                if line.difference(polygon).length > length_max:
                    # warnings.warn(f'Lline unauthorized, line intersects polygon and the length is > {length_max}') 
                    dist_matrix[i, j] = float("inf")
                    dist_matrix[j, i] = float("inf")
                else:
                    warnings.warn(
                        f"Line authorized, line intersects polygon but the length is < {length_max}"
                    )

    for i in range(n):
        ordered_points = order_points(dist_matrix, start=i)
        if ordered_points:
            ordered_array = centroids[ordered_points]
            ordered_line = shapely.LineString(ordered_array)
            if ordered_line.is_simple:
                path_length = ordered_line.length
                if path_length < min_dist:
                    min_dist = path_length
                    min_ordered_line = ordered_line

    return min_ordered_line


# --------------------------------------------------------------------------
# Line extension to the polygon boundary
# --------------------------------------------------------------------------

def extendLine(
    point1, #: shapely.Point, 
    point2, #: shapely.Point, 
    distance: float = 5000,
) -> shapely.LineString:
    """Extend a line segment formed by two points by a given distance on both ends.

    Parameters
    ----------
    point1, point2
        Coordinates of the two points defining the segment direction.
    distance
        Distance by which to extend the line on each side.

    Returns
    -------
    shapely.LineString
        4-point line: [extended_before, point1, point2, extended_after].
    """
    p1 = np.array(point1)
    p2 = np.array(point2)

    # Compute the direction vector of the line and normalize the direction vector
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)

    # Compute the new points by adding/subtracting the direction vector
    new_point1 = p1 - direction * distance
    new_point2 = p2 + direction * distance

    return shapely.LineString([new_point1, point1, point2, new_point2])


def get_angle(p1, p2, p3, degree: bool = True) -> float:
    """Compute the angle at ``p2`` formed by segments ``p2-p1`` and ``p2-p3``.

    Parameters
    ----------
    p1, p2, p3
        Coordinates.
    degree
        If True, return degrees, otherwise radians.

    Returns
    -------
    float
        The angle at ``p2``.
    """
    vec1 = p1 - p2
    vec2 = p3 - p2

    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle) if degree else angle


def addPoints(
    polygon: shapely.Polygon,
    line: shapely.LineString,
    dict_position: dict | None = None,
    distance: float = 5000,
) -> shapely.LineString:
    """Extend the line at both ends until it touches the polygon boundary.

    Parameters
    ----------
    polygon
        Polygon the line lives in.
    line
        Ordered centerline (without boundary-touching endpoints yet).
    dict_position
        Mapping of endpoint name -> pair of indices into ``line``'s
        coordinates used to define the extension direction.
        Defaults to ``{'Start': [0, 1], 'End': [-1, -2]}``.
    distance
        Length used to build the extended ray searched for boundary intersections.
        Should be large enough to guarantee the ray exits the polygon
        (see ``centerline``'s automatic default based on the polygon's own scale).

    Returns
    -------
    shapely.LineString
        The input line with two extra points added at each end, snapped to
        the polygon boundary.

    Notes
    -----
    When the extended ray crosses the polygon boundary more than once (e.g.
    a concave notch), the intersection point kept is the one furthest along
    the extension direction (i.e. the true exit point), not simply the
    closest one to the last centroid -- picking the closest one can
    incorrectly stop the line at a local notch rather than the shape's tip.
    """
    if dict_position is None:
        dict_position = {"Start": [0, 1], "End": [-1, -2]}

    points = {}
    order_centers = shapely.get_coordinates(line)

    for loc, pos in dict_position.items():
        print(f"Add point at the {loc} position : {pos}")
        p_from = order_centers[pos[0], :]
        p_to = order_centers[pos[1], :]

        extendedLine = extendLine(p_from, p_to, distance)
        touch_bound = shapely.get_coordinates(
            polygon.boundary.intersection(
                shapely.LineString([extendedLine.coords[0], extendedLine.coords[1]])
            )
        )

        if len(touch_bound) > 1:
            print(f"The {loc} touches {len(touch_bound)} points: keep the furthest along direction")
            direction = p_from - p_to
            direction = direction / np.linalg.norm(direction)

            max_proj = -float("inf")
            for point in touch_bound:
                proj = np.dot(np.array(point) - p_from, direction)
                if proj > max_proj:
                    max_proj = proj
                    points[loc] = point
        else:
            points[loc] = touch_bound

    lineFinal = shapely.LineString(np.vstack([points["Start"], order_centers, points["End"]]))
    return lineFinal


# --------------------------------------------------------------------------
# Main centerline search
# --------------------------------------------------------------------------

def centerline(
    polygon: shapely.Polygon,
    n_clusters: int = 3,
    distance: float | None = None,
    length_max: float | None = None,
    random_state: int = 130,
    img_val: np.ndarray | None = None,
    cell_points: np.ndarray | None = None,
    min_points_per_cluster: int = 15,
    threshold: float = 120,
    max_clusters: int = 30,
    max_consecutive_failures: int = 5,
    use_sampling: bool = True,
    n_sample_points: int = 3000,
    distance_frac: float = 1.0,
    length_max_frac: float = 0.02,
) -> shapely.LineString:
    """Compute the centerline of an elongated polygon (e.g. cortex ribbon).
 
    Searches for the smallest number of KMeans clusters ``k`` such that
    increasing ``k`` further starts to introduce a locally sharp bend
    (a sign of over-segmentation), then snaps both ends of the resulting
    ordered path to the polygon boundary.
 
    Parameters
    ----------
    polygon
        Polygon to compute the centerline of. Should typically be smoothed
        first (see ``_geometry_utils.auto_process_shape``) to avoid spurious
        branches/notches breaking the boundary-snapping step.
    n_clusters
        Starting number of clusters for the search.
    distance
        Length used by :func:`addPoints`/:func:`extendLine` to search for
        boundary intersections. If None, automatically set to
        ``equivalent_radius(polygon) * distance_frac`` (large enough to
        guarantee the ray exits the polygon regardless of its absolute size).
    length_max
        Maximum allowed out-of-polygon segment length in
        :func:`sortedCentroidToLine`. If None, automatically set to
        ``equivalent_radius(polygon) * length_max_frac``.
    random_state
        Seed for KMeans.
    img_val
        Optional precomputed point cloud (interior points of the polygon).
        If given, used as-is and ``cell_points``/``use_sampling`` are ignored.
        Prefer ``cell_points`` when you have real cell centroids available.
    cell_points
        Real cell centroids belonging to the shape (e.g. cortex cell
        centroids), shape ``(n_cells, 2)``. Preferred over synthetic
        sampling since it reflects the true local cell density rather than
        assuming a uniform distribution inside the polygon. Used as the
        KMeans input directly if there are at least
        ``max(n_sample_points, min_points_per_cluster * max_clusters)``
        points; otherwise topped up with uniformly sampled points (see
        ``use_sampling``) to reach that same target, so the point-cloud
        density stays consistent whether or not ``cell_points`` is given.
    min_points_per_cluster
        Safety floor: minimum number of points required per cluster (at
        ``max_clusters``). Only raises the target point-cloud size above
        ``n_sample_points`` if the latter would be too low for the requested
        ``max_clusters``.
    threshold
        Angle (in degrees) below which a bend is considered "sharp" when
        comparing consecutive k's.
    max_clusters
        Hard cap on ``n_clusters`` to guarantee termination.
    max_consecutive_failures
        Number of consecutive failed k's (no valid ordered line found)
        tolerated before stopping the search and falling back to the last
        valid k.
    use_sampling
        If True (default) and ``img_val`` is None, use fast rejection
        sampling (:func:`_geometry_utils.sample_points_in_polygon`) instead
        of full rasterization (:func:`shapeToImg`) to build the point cloud.
    n_sample_points
        Number of points to sample when ``use_sampling=True``.
    distance_frac, length_max_frac
        Fractions of the polygon's equivalent radius used to derive
        ``distance``/``length_max`` automatically when they are None.
 
    Returns
    -------
    shapely.LineString
        The final centerline, extended to touch the polygon boundary at both ends.
    """
    eq_radius = equivalent_radius(polygon)
    if distance is None:
        distance = eq_radius * distance_frac
    if length_max is None:
        length_max = eq_radius * length_max_frac
    print(f'dist = {distance}, length_max = {length_max}')

    if img_val is None:
        # Single, consistent target point-cloud size across both branches:
        # n_sample_points is the intended total, with min_points_per_cluster *
        # max_clusters only acting as a safety floor (in case n_sample_points
        # was set too low for the requested max_clusters).
        n_target = max(n_sample_points, min_points_per_cluster * max_clusters)
 
        if cell_points is not None:
            cell_points = np.asarray(cell_points)
            if len(cell_points) >= n_target:
                img_val = cell_points
            else:
                print(
                    f"Only {len(cell_points)} cell centroids available "
                    f"(< {n_target} target points); topping up with uniformly sampled points."
                )
                n_extra = n_target - len(cell_points)
                if use_sampling:
                    extra_points = sample_points_in_polygon(
                        polygon, n_points=n_extra, random_state=random_state
                    )
                    img_val = np.vstack([cell_points, extra_points])
                else:
                    img_val = cell_points
        elif use_sampling:
            img_val = sample_points_in_polygon(
                polygon, n_points=n_target, random_state=random_state
            )
        else:
            img_val = shapeToImg(polygon=polygon)   # time-consuming ++
 
    print("===========================================")
    print("Start research of the best k...")
 
    search = True
    lineK_Order = None
    prev_valid_k = None
    n_failures = 0
 
    while search:
        print(f"n_clusters = {n_clusters}...")
 
        if n_clusters > max_clusters:
            print(f"max_clusters ({max_clusters}) reached, stopping search.")
            break
 
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(img_val)
        lineK_1_Order = sortedCentroidToLine(polygon, model.cluster_centers_, length_max=length_max)
 
        if lineK_1_Order is None:
            print("No line...")
            n_failures += 1
            n_clusters += 1
            if n_failures >= max_consecutive_failures:
                print(
                    f"{n_failures} consecutive failures, falling back to last valid k = {prev_valid_k}"
                )
                search = False
            continue
 
        n_failures = 0
 
        # Only compare angle patterns between strictly consecutive k's --
        # comparing against a stale lineK_Order from several failed k's ago
        # produces a meaningless "best k" (this was the original bug).
        if lineK_Order is not None and prev_valid_k == n_clusters - 1:
            coordinates = shapely.get_coordinates(lineK_1_Order)
            angles = []
 
            if not lineK_1_Order.is_simple:
                warnings.warn("Warning Message: the line is not simple")
 
            for i in range(1, len(coordinates) - 1):
                angle = get_angle(
                    p1=coordinates[i - 1], p2=coordinates[i], p3=coordinates[i + 1], degree=True
                )
                angles.append(angle)
 
            val = np.argwhere(np.array(angles) < threshold)
            for i in range(len(val) - 1):
                if (val[i] + 1 == val[i + 1]) | (val[i] + 2 == val[i + 1]):
                    print(f"Best n_clusters found = {n_clusters - 1}")
                    n_clusters -= 1
                    search = False
                    break
            if not search:
                break
        elif prev_valid_k is not None:
            print(
                f"Non-consecutive k detected (last valid k={prev_valid_k}, current={n_clusters}), "
                "skipping angle comparison for this step."
            )
 
        prev_valid_k = n_clusters
        n_clusters += 1
        lineK_Order = lineK_1_Order
 
    if lineK_Order is None:
        raise RuntimeError(
            "No valid ordered centerline found for any tested n_clusters. "
            "Consider increasing length_max_frac, smoothing the polygon further, "
            "or checking that the polygon is a single simple Polygon."
        )
 
    lineFinal = addPoints(polygon=polygon, line=lineK_Order, distance=distance)
    return lineFinal


## OLD

#garde le points Plus PROCHE ==> new garde PTS PLUS LOIN !!
# def addPoints(
#     polygon: shapely.Polygon, 
#     line: shapely.LineString, 
#     dict_position: dict = {'Start': [0,1], 
#                            'End': [-1,-2]}, 
#     distance: int= 5000
# ) -> shapely.LineString:
#     """
#     Add points to touch the boundary 
    
#     Parameters
#     ----------
#         polygon (shapely.Polygon): _description_
#         line (shapely.LineString): _description_
#         dict_position (_type_, optional): _description_. Defaults to {'Start': [0,1], 'End': [-1,-2]}.
#         distance (int, optional): _description_. Defaults to 5000.

#     Returns
#     -------
#         shapely.LineString: _description_
#     """
#     points = {}
#     order_centers = shapely.get_coordinates(line)

#     for loc, pos in dict_position.items():
#         print(f'Add point at the {loc} position : {pos}')
#         extendedLine = extendLine(order_centers[pos[0], :], 
#                                   order_centers[pos[1], :], distance)
#         touch_bound = shapely.get_coordinates(polygon.boundary.intersection(
#             shapely.LineString([extendedLine.coords[0], extendedLine.coords[1]])))
#         # touch_bound = shapely.get_coordinates(polygon.boundary.intersection(extendedLine))

#         if len(touch_bound) > 1 :
#             min_dist = float('inf')
#             print(f"The {loc} touches 2 points : keep the closest")
#             for point in touch_bound:
#                 dist_pts = shapely.distance(shapely.Point(point), 
#                                             shapely.Point(order_centers[pos[0]]))
#                 if dist_pts < min_dist:
#                     # print(f"Smaller distance for the {point = } : {dist_pts}")
#                     min_dist = dist_pts
#                     points[loc] = point
#         else:
#             points[loc] = touch_bound

#     lineFinal = shapely.LineString(
#         np.vstack([points['Start'], order_centers, points['End']]))
#     return lineFinal



         
def centerline_OLD(
    polygon: shapely.Polygon,
    n_clusters: int = 3,
    distance: int = 5000,
    length_max: int = 5,
    random_state: int = 130,
    img_val = None,
    # seuil: float = 0.75,
    threshold: int = 120,
    # max_clusters = 100,
) -> shapely.LineString:
    """Compute the centerline

    Parameters
    ----------
        polygon (shapely.Polygon): _description_
        n_clusters (int, optional): _description_. Defaults to 3.
        distance (int, optional): _description_. Defaults to 5000.
        length_max (int, optional): _description_. Defaults to 5.
        random_state (int, optional): _description_. Defaults to 130.
        img_val (_type_, optional): _description_. Defaults to None.
        threshold (int, optional): _description_. Defaults to 120.

    Returns
    -------
        shapely.LineString: _description_
    """
 
    if img_val is None :
        img_val = shapeToImg(polygon=polygon)
    print("===========================================")
    print(f"Start research of the best k...")
    # threshold=None
    search = True
    lineK_Order = None
    
    while search:
        print(f"n_clusters = {n_clusters}...")

        model = KMeans(n_clusters= n_clusters, random_state = random_state).fit(img_val)
        lineK_1_Order = sortedCentroidToLine(polygon, model.cluster_centers_, length_max=length_max)
    
        if lineK_1_Order:
            if lineK_Order:
                coordinates = shapely.get_coordinates(lineK_1_Order)
                angles=[]
                
                if not lineK_1_Order.is_simple:
                    warnings.warn("Warning Message: the line is not simple") 

                for i in range(1, len(coordinates) - 1):
                    angle = get_angle(p1=coordinates[i-1], 
                                    p2=coordinates[i], 
                                    p3=coordinates[i+1], 
                                    degree = True)
                    angles.append(angle)
                    
                # val = np.argwhere(angles < np.quantile(angles,0.20))
                # for i in range(len(val)-1):
                #     if val[i] + 1 == val[i+1]:
                #         print(f'Best n_clusters found = {n_clusters-1}')
                #         n_clusters -= 1 
                #         search = False
                #         break   
                
                val = np.argwhere(np.array(angles) < threshold)
                for i in range(len(val)-1):
                    if (val[i] + 1 == val[i+1]) | (val[i] + 2 == val[i+1]) :
                        print(f'Best n_clusters found = {n_clusters-1}')
                        n_clusters -= 1 
                        search = False
                        break   
                if not search:
                    break
                
            n_clusters += 1
            lineK_Order = lineK_1_Order
        
        else:
            print("No line...")
            n_clusters += 1
           
    lineFinal = addPoints(polygon = polygon, 
                            line = lineK_Order, distance = distance)
    return lineFinal
