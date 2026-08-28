import scipy
import shapely
import numpy as np
import skimage as ski
from sklearn.cluster import KMeans 
import warnings
from shapely.ops import nearest_points
from shapely import LineString

from ._geometry_utils import equivalent_radius, sample_points_in_polygon

# 1) shapeToImg
# 2) sortedCentroidToLine
# order_points
# 3) get_angle 
# extendLine 
# 4) addPoints
# 5) _resolve_img_val
# 6) _count_sharp
# 7) _robust_sorted_centroid_to_line
# 8) _search_optimal_k
# 9) centerline 
 

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
    img[col, row] = 1           # row = y, col = x
    yy, xx = np.where(img)
    posImg = np.column_stack((xx, yy))
    if only_position:
        return posImg
    return img, posImg


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
    # Path ordering (TSP-like greedy walk)
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
        default. On shapes with a narrow pinch/notch (e.g. two lobes joined
        by a thin bridge), a valid ordering may not exist at all below a
        certain ``length_max`` -- see ``_robust_sorted_centroid_to_line``.

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


def _get_angle(p1, p2, p3, degree: bool = True) -> float:
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
    vec1 = np.asarray(p1) - np.asarray(p2)
    vec2 = np.asarray(p3) - np.asarray(p2)
    # vec1 = p1 - p2
    # vec2 = p3 - p2

    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle) if degree else angle


def addPoints(
    polygon: shapely.Polygon,
    line: shapely.LineString,
    dict_position: dict | None = None,
    distance: float = 5000,
    notch_lookahead_frac: float = 0.05,
    verbose: bool = False,
) -> shapely.LineString:
    if dict_position is None:
        dict_position = {"Start": [0, 1], "End": [-1, -2]}
 
    points = {}
    order_centers = shapely.get_coordinates(line)
    line_len = line.length
 
    for loc, pos in dict_position.items():
        p_from = order_centers[pos[0], :]
        p_to = order_centers[pos[1], :]
 
        extendedLine = extendLine(p_from, p_to, distance)
        touch_bound = shapely.get_coordinates(
            polygon.boundary.intersection(
                shapely.LineString([extendedLine.coords[0], extendedLine.coords[1]])
            )
        )
 
        # if verbose:
        #     print(f"[{loc}] p_from={p_from}, p_to={p_to}, touch_bound={touch_bound}")

        if len(touch_bound) > 1:
            p_from_pos = line.project(shapely.Point(p_from))    # should be around 0 or line.length
        
            chosen = None
            best_dist_euclid = -1.0
            fallback_point = None
            fallback_dist = float("inf")
 
            for pt in touch_bound:
                pt = np.asarray(pt)
                nearest_on_line = nearest_points(shapely.Point(pt), line)[1]
                pos_on_line = line.project(nearest_on_line)
                dist_along_line = abs(pos_on_line - p_from_pos)
                frac = dist_along_line / line_len if line_len > 0 else float("inf")

                dist_euclid = shapely.distance(shapely.Point(pt), shapely.Point(p_from))
 
                if dist_euclid < fallback_dist:
                    fallback_dist = dist_euclid
                    fallback_point = pt

                if frac <= notch_lookahead_frac and dist_euclid > best_dist_euclid:
                    best_dist_euclid = dist_euclid
                    chosen = pt
                    if verbose:
                        print(f"[{loc}]     -> new best so far (dist_euclid={best_dist_euclid:.2f})")

            if chosen is None:
                chosen = fallback_point
                if verbose:
                    print(
                        f"[{loc}] no candidate passed the filter, falling back to "
                        f"nearest candidate: point={np.round(chosen, 1)} "
                        f"(dist_euclid={fallback_dist:.2f})"
                    )
            else:
                if verbose:
                    print(
                        f"[{loc}] chosen point (max dist_euclid among kept): "
                        f"{np.round(chosen, 1)} (dist_euclid={best_dist_euclid:.2f})"
                    )
 
            points[loc] = chosen
        else:
            if len(touch_bound) == 0 and verbose:
                print(
                    f"[{loc}] the extended ray found NO boundary intersection "
                    f"-- leaving this end un-extended (line stops at the last centroid)."
                )
            points[loc] = touch_bound
 
    lineFinal = shapely.LineString(np.vstack([points["Start"], order_centers, points["End"]]))
    return lineFinal

 
# def addPoints(
#     polygon: shapely.Polygon,
#     line: shapely.LineString,
#     dict_position: dict | None = None,
#     distance: float = 5000,
# ) -> shapely.LineString:
#     """Extend the line at both ends until it touches the polygon boundary.

#     Parameters
#     ----------
#     polygon
#         Polygon the line lives in.
#     line
#         Ordered centerline (without boundary-touching endpoints yet).
#     dict_position
#         Mapping of endpoint name -> pair of indices into ``line``'s
#         coordinates used to define the extension direction.
#         Defaults to ``{'Start': [0, 1], 'End': [-1, -2]}``.
#     distance
#         Length used to build the extended ray searched for boundary intersections.

#     Returns
#     -------
#     shapely.LineString
#         The input line with two extra points added at each end, snapped to
#         the polygon boundary.

#     Notes
#     -----
#     When the extended ray crosses the polygon boundary more than once (e.g.
#     a concave notch), the intersection point kept is the one furthest along
#     the extension direction (i.e. the true exit point), not simply the
#     closest one to the last centroid.
#     """
#     if dict_position is None:
#         dict_position = {"Start": [0, 1], "End": [-1, -2]}

#     points = {}
#     order_centers = shapely.get_coordinates(line)

#     for loc, pos in dict_position.items():
#         # print(f"Add point at the {loc} position : {pos}")
#         p_from = order_centers[pos[0], :]
#         p_to = order_centers[pos[1], :]

#         extendedLine = extendLine(p_from, p_to, distance)
#         touch_bound = shapely.get_coordinates(
#             polygon.boundary.intersection(
#                 shapely.LineString([extendedLine.coords[0], extendedLine.coords[1]])
#             )
#         )

#         if len(touch_bound) > 1:
#             print(f"The {loc} touches {len(touch_bound)} points: keep the furthest along direction")
#             direction = p_from - p_to
#             direction = direction / np.linalg.norm(direction)

#             max_proj = -float("inf")
#             for point in touch_bound:
#                 proj = np.dot(np.array(point) - p_from, direction)
#                 if proj > max_proj:
#                     max_proj = proj
#                     points[loc] = point
#         else:
#             points[loc] = touch_bound

#     lineFinal = shapely.LineString(np.vstack([points["Start"], order_centers, points["End"]]))
#     return lineFinal


def _resolve_img_val(
    polygon: shapely.Polygon,
    cell_points: np.ndarray | None,
    use_sampling: bool,
    n_sample_points: int,
    min_points_per_cluster: int,
    max_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Build the point array to use for clustering when img_val is not given.

    Priority: cell_points (padded with sampled points if needed), else
    pure sampling, else polygon rasterization.
    """
    n_target = max(n_sample_points, min_points_per_cluster * max_clusters)

    if cell_points is not None:
        cell_points = np.asarray(cell_points)
        if len(cell_points) >= n_target:
            return cell_points

        n_extra = n_target - len(cell_points)
        if use_sampling:
            print(
                f"Only {len(cell_points)} cell centroids available "
                f"(< {n_target} target points); topping up with uniformly sampled points."
            )
            extra_points = sample_points_in_polygon(
                polygon, n_points=n_extra, random_state=random_state
            )
            return np.vstack([cell_points, extra_points])
        return cell_points

    if use_sampling:
        return sample_points_in_polygon(
            polygon, n_points=n_target, random_state=random_state
        )

    return shapeToImg(polygon=polygon) # time-consuming ++


def _count_sharp(coords: np.ndarray, threshold_deg: float) -> tuple[int, bool, list]:
    """Number of interior vertices with angle < threshold_deg, and whether
    any two of them are adjacent along the path.
 
    Parameters
    ----------
    coords
        Ordered vertex coordinates of the centerline.
    threshold_deg
        Angle (in degrees) below which a vertex is considered "sharp".
 
    Returns
    -------
    (nb, has_adjacent_pair, sharp_indices)
        nb: number of sharp vertices.
        has_adjacent_pair: True if at least two sharp vertices sit at
        consecutive indices along the path (i.e. i and i+1 are both sharp)
        -- this is the local-zigzag signal (both walls of one notch, or
        noise piling up in one spot). False if all sharp vertices are
        isolated from each other (separated by at least one non-sharp
        vertex) -- these are more likely distinct, legitimate bends at
        different locations (e.g. the two curves of an "S" shape), not a
        zigzag. Always False if nb <= 1.
        sharp_indices: the raw list of sharp vertex indices, exposed so the
        caller can track whether the *same* configuration persists across
        consecutive k (see ``_search_optimal_k``'s warning-tracking logic).
 
    Notes
    -----
    A spatial-distance version of this check (flagging two sharp vertices
    as "the same feature" when they're within some fixed distance of each
    other) was tried and reverted: the right distance threshold scales with
    the *local point spacing at the current k*, which varies enormously
    across the search (sparse at low k, dense at high k). A fixed fraction
    of the polygon's equivalent radius was either too small (never
    triggers, search never stops) or too large (triggers on unrelated
    vertices), and no single value worked across k. Index-adjacency doesn't
    have this problem: it's implicitly relative to the current point
    density, since "adjacent" always means "no other vertex in between",
    regardless of k.
    """
    sharp_indices = []
    for i in range(1, len(coords) - 1):
        angle = _get_angle(coords[i - 1], coords[i], coords[i + 1], degree=True)
        if angle < threshold_deg:
            sharp_indices.append(i)
 
    nb = len(sharp_indices)
    if nb <= 1:
        return nb, False, sharp_indices
 
    sharp_set = set(sharp_indices)
    has_adjacent_pair = any(
        (i - 1) in sharp_set or (i + 1) in sharp_set for i in sharp_indices
    )
    return nb, has_adjacent_pair, sharp_indices

 

def _robust_sorted_centroid_to_line(
    polygon: shapely.Polygon,
    centroids,
    length_max: float,
    growth_factor: float = 2.0,
    max_retries: int = 4,
    verbose: bool = False,
) -> tuple[shapely.LineString | None, float]:
    """``sortedCentroidToLine`` with automatic ``length_max`` relaxation.

    On shapes with a narrow pinch (e.g. a notch between two lobes), a strict
    ``length_max`` tuned for smooth ribbons can make ``order_points`` fail to
    find *any* valid ordering, even though one exists -- crossing the pinch
    unavoidably requires one out-of-polygon segment longer than the
    ribbon-tuned default. This wrapper geometrically grows ``length_max``
    up to ``max_retries`` times before giving up.

    Returns
    -------
    (line, length_max_used)
        ``line`` is None if no ordering was found even at the largest
        ``length_max`` tried.
    """
    current = length_max
    for attempt in range(max_retries + 1):
        line = sortedCentroidToLine(polygon, centroids, length_max=current)
        if line is not None:
            if attempt > 0 and verbose:
                print(f"    (length_max relaxed to {current:.1f} to find a valid ordering)")
            return line, current
        current *= growth_factor
    return None, current


def _adaptive_threshold(
    coords: np.ndarray,
    default_threshold_deg: float,
    min_threshold_deg: float,
    margin_deg: float,
) -> float:
    """Derive ``threshold_deg`` from the sharpest angle in a reference line
    (typically the first successful one), so shapes with multiple genuine
    but non-noise curves (e.g. an "S") don't get their natural bends
    misflagged as "sharp" by an overly permissive fixed threshold, while
    shapes with a single gentle bend don't get an overly strict one either.
 
    The reasoning: at a low k, the ordered path is coarse and mostly free
    of over-segmentation noise, so its sharpest angle is a reasonable proxy
    for "the tightest genuine curve this shape actually has". Setting the
    working threshold just below that (by ``margin_deg``) means this
    reference line's own bends won't later be re-flagged as "sharp" at
    higher k just because they're genuinely curved -- only angles sharper
    than what the shape has already shown itself to genuinely need will
    trigger the zigzag machinery.
 
    Parameters
    ----------
    coords
        Ordered vertex coordinates of the reference line.
    default_threshold_deg
        Upper bound: never return a threshold above this (a shape that is
        already perfectly straight at the reference k shouldn't get an
        unreasonably permissive threshold).
    min_threshold_deg
        Lower bound: never return a threshold below this (a guard against
        calibrating on a reference line that already contains noise).
    margin_deg
        Safety margin subtracted from the sharpest observed angle.
 
    Returns
    -------
    float
        The adaptive threshold, clipped to [min_threshold_deg, default_threshold_deg].
    """
    angles = [
        get_angle(coords[i - 1], coords[i], coords[i + 1], degree=True)
        for i in range(1, len(coords) - 1)
    ]
    if not angles:
        return default_threshold_deg
 
    sharpest = min(angles)
    adaptive = sharpest - margin_deg
    return float(np.clip(adaptive, min_threshold_deg, default_threshold_deg))
 
 

def _search_optimal_k(
    polygon: shapely.Polygon,
    img_val: np.ndarray,
    n_clusters_start: int,
    length_max: float,
    threshold_deg: float,
    max_clusters: int,
    max_consecutive_failures: int,
    random_state: int,
    length_max_growth: float,
    length_max_max_retries: int,
    adaptive_threshold: bool, #NEW
    min_threshold_deg: float, #NEW
    threshold_margin_deg: float, #NEW
    verbose: bool,
) -> tuple[shapely.LineString, int]:
    """Search the smallest k such that nb_sharp(k) stays at 0 or 1, OR has
    multiple sharp vertices with none of them adjacent along the path
    (distinct legitimate bends, e.g. the two curves of an "S" shape).
    Rolls back as soon as an adjacent pair of sharp vertices is confirmed
    (a genuine local zigzag), following the warning/persistence rule below.
 
    Decision rule
    -------------
    - nb_sharp == 0: reset any pending warning, keep going.
    - nb_sharp == 1: this is a *warning*, not an immediate verdict -- it may
      be a genuine bend or the first sign of over-segmentation. Keep going.
    - nb_sharp >= 2 with no adjacent pair: multiple sharp vertices, but each
      isolated from the others -- more likely distinct genuine bends than a
      local zigzag. Treated the same as nb_sharp == 0 (reset warning, keep
      going).
    - nb_sharp >= 2 with an adjacent pair: a genuine local zigzag signal.
      Roll back:
        * if no warning was pending: roll back one step (k - 1).
        * if the warning appeared only at k - 1 (not yet confirmed stable):
          roll back an extra step, to before the warning (k - 2).
        * if the warning had already persisted >= 2 steps (confirmed
          genuine bend): roll back only one step, rejecting just this
          explosion step.
 
    Validated against 10 real samples (simple ribbon-shaped biopsies with a
    single genuine bend, all resolved via adjacent sharp pairs) -- see
    conversation history for the validation script.
 
    Returns
    -------
    (best_line, best_k)
    """
    n_clusters = n_clusters_start
    n_failures = 0
    found_any_line = False      # new
    threshold_calibrated = not adaptive_threshold  # skip calibration if disabled

    warning_k = None           # k at which the current sharp configuration first appeared
    prev_sharp_indices = []    # sharp_indices from the last processed (non-zigzag) k
    last_valid_k = None
    last_valid_line = None
    k_to_line = {}             # keep recent valid lines around for rollback


    while True:
        if n_clusters > max_clusters:
            if verbose:
                if not found_any_line:
                    print(
                        f"max_clusters ({max_clusters}) reached without ever finding "
                        f"a valid ordered line -- this shape likely needs more points "
                        f"to be traceable. Try calling centerline() again with a "
                        f"higher max_clusters."
                    )
                else:
                    print(f"max_clusters ({max_clusters}) reached, stopping search.")
            break

        if verbose:
            print(f"n_clusters = {n_clusters}...")

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(img_val)
        line, used_length_max = _robust_sorted_centroid_to_line(
            polygon,
            model.cluster_centers_,
            length_max=length_max,
            growth_factor=length_max_growth,
            max_retries=length_max_max_retries,
            verbose=verbose,
        )

        if line is None:
            if verbose:
                print("No line (even after relaxing length_max)...")
            n_clusters += 1

            if not found_any_line:  # new
                # No line has ever been found yet
                continue

            n_failures += 1
            if n_failures >= max_consecutive_failures:
                if verbose:
                    print(f"{n_failures} consecutive failures, stopping search.")
                break
            continue
        n_failures = 0
        found_any_line = True       # new

        coords = shapely.get_coordinates(line)

        if not threshold_calibrated:
            threshold_deg = _adaptive_threshold(
                coords, threshold_deg, min_threshold_deg, threshold_margin_deg
            )
            threshold_calibrated = True
            if verbose:
                print(f"    threshold_deg calibrated to {threshold_deg:.1f} from first line (k={n_clusters})")
 
        nb_sharp, has_adjacent_pair, sharp_indices = _count_sharp(coords, threshold_deg)
        k_to_line[n_clusters] = line

    
        if verbose:
            print(
                f"    nb_sharp={nb_sharp}, has_adjacent_pair={has_adjacent_pair}, "
                f"sharp_indices={sharp_indices}, warning_k={warning_k}"
            )
 

        if nb_sharp == 0:
            warning_k = None
            prev_sharp_indices = []
            last_valid_k, last_valid_line = n_clusters, line
            n_clusters += 1
            continue
        
        if nb_sharp == 1 or not has_adjacent_pair:
            # not (yet) a confirmed zigzag: track whether the sharp-vertex
            # configuration itself is stable (unchanged) or just changed
            # vs the previous k -- only a truly unchanged configuration
            # counts as "confirmed" for the gentler rollback below. This
            # matters because KMeans re-fits from scratch at every k, so a
            # sharp vertex can silently jump to a different index/location
            # between consecutive k even while nb_sharp stays low -- that's
            # a NEW risk, not a persisting one, even though the raw count
            # alone wouldn't show it.
            if warning_k is None or sharp_indices != prev_sharp_indices:
                warning_k = n_clusters
            prev_sharp_indices = sharp_indices
            last_valid_k, last_valid_line = n_clusters, line
            n_clusters += 1
            continue


        # if nb_sharp == 1:
        #     if warning_k is None:
        #         warning_k = n_clusters
        #     last_valid_k, last_valid_line = n_clusters, line
        #     n_clusters += 1
        #     continue

        # if not has_adjacent_pair: # NEW
        #     # 2+ sharp vertices, but none of them adjacent along the path:
        #     # these are more likely distinct, legitimate bends at separate
        #     # locations (e.g. the two curves of an "S" shape) rather than a
        #     # local zigzag -- treat as safe, same as the "no sharp vertex"
        #     # case, and keep searching for a possibly still-smaller k.
        #     warning_k = None
        #     last_valid_k, last_valid_line = n_clusters, line
        #     n_clusters += 1
        #     continue

        # nb_sharp >= 2 AND at least one adjacent pair -> genuine local
        # zigzag signal (e.g. both walls of the same notch, or noise).
        # `warning_k` here reflects the age of the PRE-zigzag configuration
        # (from the last non-zigzag k) -- it is deliberately NOT updated
        # using this zigzag k's own (just-changed) sharp_indices.
        if warning_k is None:
            # direct zigzag, no prior single-point warning
            best_k = n_clusters - 1
        elif warning_k == n_clusters - 1:
            # the pre-zigzag configuration appeared just one step ago ->
            # not yet confirmed, distrust it too and roll back an extra step
            best_k = warning_k - 1
        else:
            # the pre-zigzag configuration had persisted unchanged for >= 2
            # steps -> confirmed genuine, only reject this explosion step
            best_k = n_clusters - 1
 
        if best_k in k_to_line:
            return k_to_line[best_k], best_k
        if best_k < n_clusters_start:
            # print(f"{n_failures} consecutive failures, stopping search.")
            # nothing usable before the very first tested k: the rollback
            # wants to land below n_clusters_start, which means the very
            # first k(s) tested were already showing a warning/zigzag --
            # likely n_clusters_start was set too high for this polygon.
            if verbose:
                print(
                    f"    best_k ({best_k}) < n_clusters_start ({n_clusters_start}): "
                    f"the search never had a clean baseline to roll back to. "
                    f"Try calling centerline() again with a lower n_clusters "
                    f"(e.g. n_clusters={max(2, n_clusters_start - 3)})."
                )
            break
        
        # best_k line wasn't kept (e.g. it failed/was skipped) -> refit
        model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10).fit(img_val)
        refit_line, _ = _robust_sorted_centroid_to_line(
            polygon, model.cluster_centers_, length_max=length_max,
            growth_factor=length_max_growth, max_retries=length_max_max_retries,
        )
        return (refit_line if refit_line is not None else last_valid_line), best_k

    if last_valid_line is None:
        raise RuntimeError(
            "No valid ordered centerline found for any tested n_clusters. "
            "Consider increasing length_max_frac / length_max_growth, smoothing "
            "the polygon further, or checking that the polygon is a single "
            "simple Polygon."
        )
    return last_valid_line, last_valid_k




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
    adaptive_threshold: bool = False,
    min_threshold_deg: float = 90.0,
    threshold_margin_deg: float = 10.0,
    max_clusters: int = 50,
    max_consecutive_failures: int = 5,
    use_sampling: bool = True,
    n_sample_points: int = 3000,
    distance_frac: float = 1.0,
    length_max_frac: float = 0.02,
    length_max_growth: float = 2.0,
    length_max_max_retries: int = 4,
    notch_lookahead_frac: float = 0.05,
    verbose: bool = True,
) -> shapely.LineString:
    """Compute the centerline of an elongated polygon (e.g. cortex ribbon).

    Searches for the smallest number of KMeans clusters ``k`` such that the
    number of sharp bends (angle < ``threshold``) in the ordered path stays
    at 0 or 1, rolling back as soon as it's confirmed to jump to >= 2. See
    the module docstring for the exact decision rule and its validation
    against real samples.

    Parameters
    ----------
    polygon
        Polygon to compute the centerline of. Should typically be smoothed
        first (see ``_geometry_utils.auto_process_shape``).
    n_clusters
        Starting number of clusters for the search.
    distance
        Length used by :func:`addPoints`/:func:`extendLine` to search for
        boundary intersections. If None, set to
        ``equivalent_radius(polygon) * distance_frac``.
    length_max
        Maximum allowed out-of-polygon segment length in
        :func:`sortedCentroidToLine`. If None, set to
        ``equivalent_radius(polygon) * length_max_frac``. Relaxed
        automatically if no valid ordering is found -- see
        ``length_max_growth``/``length_max_max_retries``.
    random_state
        Seed for KMeans.
    img_val
        Optional precomputed point cloud. If given, used as-is.
    cell_points
        Real cell centroids belonging to the shape. Preferred over synthetic
        sampling. Topped up with uniform sampling if too few points.
    min_points_per_cluster
        Safety floor: minimum number of points required per cluster (at
        ``max_clusters``).
    threshold
        Angle (in degrees) below which a bend is considered "sharp".
    max_clusters
        Hard cap on ``n_clusters``.
    max_consecutive_failures
        Number of consecutive k's with no valid ordered line (even after
        relaxing ``length_max``) tolerated before stopping the search.
    use_sampling
        If True and ``img_val`` is None, use fast rejection sampling instead
        of full rasterization.
    n_sample_points
        Number of points to sample when ``use_sampling=True``.
    distance_frac, length_max_frac
        Fractions of the polygon's equivalent radius used to derive
        ``distance``/``length_max`` automatically when they are None.
    length_max_growth, length_max_max_retries
        See ``_robust_sorted_centroid_to_line``: how much and how many times
        ``length_max`` is grown when no valid ordering is found at the
        current value (useful for narrow pinches/notches).
    verbose
        Print search progress.

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

    if verbose:
        print(f"dist = {distance}, length_max = {length_max}")

    if img_val is None:
        img_val = _resolve_img_val(
            polygon=polygon,
            cell_points=cell_points,
            use_sampling=use_sampling,
            n_sample_points=n_sample_points,
            min_points_per_cluster=min_points_per_cluster,
            max_clusters=max_clusters,
            random_state=random_state,
        )

    if verbose:
        print("===========================================")
        print("Start research of the best k (warning/persistence rule)...")

    best_line, best_k = _search_optimal_k(
        polygon=polygon,
        img_val=img_val,
        n_clusters_start=n_clusters,
        length_max=length_max,
        threshold_deg=threshold,
        adaptive_threshold=adaptive_threshold,
        min_threshold_deg=min_threshold_deg,
        threshold_margin_deg=threshold_margin_deg,
        max_clusters=max_clusters,
        max_consecutive_failures=max_consecutive_failures,
        random_state=random_state,
        length_max_growth=length_max_growth,
        length_max_max_retries=length_max_max_retries,
        verbose=verbose,
    )

    if verbose:
        print(f"Best n_clusters found = {best_k}")

    lineFinal = addPoints(
        polygon=polygon,
        line=best_line,
        distance=distance,
        notch_lookahead_frac=notch_lookahead_frac, # new param
        verbose=verbose,
    )

    return lineFinal



def align_centerline_by_axis(
    line: LineString,
    axis: int = 1,
    start_at: str = "top",
) -> LineString:
    """Orient a centerline so its start point is on a given side along one axis.

    Unlike pairwise alignment against a reference centerline (which requires
    samples to share a common coordinate frame), this uses an intrinsic
    per-line criterion: compare the coordinate value of the start vs end
    point along `axis` and flip if needed.

    Parameters
    ----------
    line
        Centerline to orient.
    axis
        Coordinate axis to use: 0 for x, 1 for y.
    start_at
        Which side the start point should be on: "top"/"bottom" for axis=1
        (y), or "left"/"right" for axis=0 (x). "top"/"left" means start
        should have the larger coordinate value; "bottom"/"right" means
        start should have the smaller value.

    Returns
    -------
    The centerline, reversed if needed.
    """
    coords = np.array(line.coords)
    start_val = coords[0, axis]
    end_val = coords[-1, axis]

    start_should_be_larger = start_at in ("top", "left")
    is_correct = (start_val > end_val) == start_should_be_larger

    if not is_correct:
        return LineString(coords[::-1])
    return line


def align_centerlines_by_axis(
    centerlines: dict[str, LineString],
    axis: int = 1,
    start_at: str = "top",
) -> dict[str, LineString]:
    """Orient all centerlines in a dict independently, using an intrinsic axis criterion.

    Use this instead of `align_centerlines` (reference-based) when samples
    don't share a common coordinate frame (different tissue sections,
    different positions/rotations).

    Parameters
    ----------
    centerlines
        Mapping of sample id -> centerline LineString.
    axis
        0 for x, 1 for y.
    start_at
        "top", "bottom", "left", or "right" — desired position of the start point.

    Returns
    -------
    New dict with all centerlines consistently oriented.
    """
    return {
        key: align_centerline_by_axis(line, axis=axis, start_at=start_at)
        for key, line in centerlines.items()
    }


def centerline_V1(
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
        img_val = _resolve_img_val(
            polygon=polygon,
            cell_points=cell_points,
            use_sampling=use_sampling,
            n_sample_points=n_sample_points,
            min_points_per_cluster=min_points_per_cluster,
            max_clusters=max_clusters,
            random_state=random_state,
        )

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
 
        if lineK_Order is not None and prev_valid_k == n_clusters - 1:
            coordinates = shapely.get_coordinates(lineK_1_Order)
            angles = []
 
            if not lineK_1_Order.is_simple:
                warnings.warn("Warning Message: the line is not simple")
            # removed in new version !!!

            for i in range(1, len(coordinates) - 1):
                angle = _get_angle(
                    p1=coordinates[i - 1], p2=coordinates[i], p3=coordinates[i + 1], degree=True
                )
                angles.append(angle)
            print(angles)

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
                    angle = _get_angle(p1=coordinates[i-1], 
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


