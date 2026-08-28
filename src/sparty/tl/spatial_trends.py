import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from shapely import LineString, Point, get_coordinates, Polygon
from spatialdata import SpatialData
import shapely
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from ..pp.transcripts import subset_transcripts
from ..pp.transformations import compute_bounds_gpd


def _prepare_geodataframe(
    data,
    key,
    coordinates,
    genes=None,
    cells=None,
    cell_id_col="cell_id",
    techno="Xenium",
    only_in_cell=False,
    transform=False,
    scale=False,
):
    """
    Prépare le GeoDataFrame de travail (shapes ou points), avec filtrage
    optionnel en amont du calcul de distance.

    - Transcripts issus d'un SpatialData (élément dask, ex: key='transcripts') :
      délègue à `spt.pp.transcripts.subset_transcripts`, qui gère déjà le
      dask -> geopandas, le transform + scale des coordonnées (important
      pour Xenium/Merscope), le filtre qualité, le filtre par gène
      (`genes`) et le filtre "assigné à une cellule" (`only_in_cell`).
      `cells` est appliqué en plus, après coup, sur `cell_id_col` si la
      colonne existe (utile pour ne garder que les transcripts d'une
      sous-population de cellules).
    - Shapes (ex: 'cell_boundaries') ou tout autre GeoDataFrame/DataFrame
      déjà en mémoire : chemin générique inchangé, avec filtrage `cells`
      sur `cell_id_col` si présent, sinon sur l'index.

    Retourne (gdf, is_points).
    """
    is_sdata = isinstance(data, sd.SpatialData)
    is_transcript_element = is_sdata and isinstance(data[key], dd.DataFrame)

    if is_transcript_element:
        gdf = subset_transcripts(
            data,
            genes=genes,
            transcript_key=key,
            techno=techno,
            only_in_cell=only_in_cell,
            transform=transform,
            scale=scale,
            return_gpd=True,
        )
        if cells is not None:
            if cell_id_col in gdf.columns:
                gdf = gdf[gdf[cell_id_col].isin(cells)]
            else:
                print(
                    f"`cells` fourni mais colonne '{cell_id_col}' absente des "
                    "transcripts -> filtre ignoré."
                )
        return gdf, True

    df = data[key] if is_sdata else data

    if isinstance(df, dd.DataFrame):
        df = df.compute()
    else:
        df = df.copy()

    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[coordinates[0]], df[coordinates[1]])
        )

    is_points = bool((df.geometry.geom_type == "Point").all())

    if is_points and genes is not None:
        gene_col = next((c for c in ("feature_name", "gene", "genes") if c in df.columns), None)
        if gene_col is not None:
            df = df[df[gene_col].isin(genes)]
        else:
            print("`genes` fourni mais aucune colonne de gène reconnue -> filtre ignoré.")

    if cells is not None:
        if cell_id_col in df.columns:
            df = df[df[cell_id_col].isin(cells)]
        else:
            df = df[df.index.isin(cells)]

    return df, is_points


def _write_back(data, key, gdf, cols):
    """Réinjecte les colonnes calculées dans l'objet d'origine, quand c'est possible."""
    if isinstance(data, sd.SpatialData):
        target = data[key]
        if isinstance(target, dd.DataFrame):
            print(
                f"'{key}' est une dask.DataFrame (typiquement les transcripts) : "
                "les résultats ne sont pas réinjectés automatiquement dans le sdata "
                "(dask n'aime pas les assignations par index comme ça). "
                "Utilise return_df=True et gère l'écriture toi-même "
                "(ex: nouvel élément 'points', ou merge sur cell_id/transcript_id)."
            )
            return
        target[cols] = gdf[cols]
    elif isinstance(data, dd.DataFrame):
        print(
            "L'input est une dask.DataFrame : utilise return_df=True, "
            "le résultat est retourné en pandas.DataFrame (déjà calculé)."
        )
    else:
        data[cols] = gdf[cols]


def _clamp_n_jobs(n_jobs):
    """Empêche de demander plus de process que de cœurs disponibles."""
    if not n_jobs or n_jobs <= 1:
        return 1
    available = cpu_count()
    if n_jobs > available:
        print(f"n_jobs={n_jobs} > cpu_count()={available} -> réduit à {available}.")
        return available
    return n_jobs


def _chunk_star(chunk, worker, fixed_kwargs):
    return worker(*chunk, **fixed_kwargs)


def _chunked_map(worker, arrays, chunk_size, n_jobs, show_progress, desc, **fixed_kwargs):
    n = len(arrays[0])

    if n <= chunk_size:
        return worker(*arrays, **fixed_kwargs)

    chunks = list(zip(*[
        [a[i:i + chunk_size] for i in range(0, n, chunk_size)]
        for a in arrays
    ]))

    if n_jobs and n_jobs > 1:
        worker_partial = partial(_chunk_star, worker=worker, fixed_kwargs=fixed_kwargs)
        with Pool(n_jobs) as pool:
            results = list(
                tqdm(pool.imap(worker_partial, chunks), total=len(chunks),
                     disable=not show_progress, desc=f"{desc} (multiprocessing)")
            )
    else:
        results = [
            worker(*chunk, **fixed_kwargs)
            for chunk in tqdm(chunks, disable=not show_progress, desc=f"{desc} (chunked)")
        ]

    return np.concatenate(results)


# ----------------------------------------------------------------------
# ALONG — distance projetée sur la centerline
# ----------------------------------------------------------------------

def _along_chunk_worker(pts_chunk, centerline):
    return shapely.line_locate_point(centerline, pts_chunk)


def _compute_along(dist_geom, centerline, nb_interval, round_val,
                   bin_size=None,
                    chunk_size=200_000, n_jobs=1, show_progress=True):
    pts_arr = np.asarray(dist_geom)

    dist_along = _chunked_map(
        _along_chunk_worker, [pts_arr], chunk_size, n_jobs, show_progress, "along",
        centerline=centerline,
    )

    if bin_size is not None:
        # bins de taille fixe (microns) -> nombre d'intervalles dérivé
        cuts = np.arange(0, centerline.length + bin_size, bin_size)
    else:
        # comportement historique : nb_interval fixe
        step = centerline.length / nb_interval
        cuts = np.arange(0, centerline.length + step, step)

    labels = [str(i) for i in range(len(cuts) - 1)]

    out = pd.DataFrame(index=dist_geom.index)
    out["distance_along"] = dist_along
    out["cat_along"] = pd.cut(
        out["distance_along"], bins=cuts, labels=labels, right=True, include_lowest=True
    )
    out["dst_along_norm"] = (out["distance_along"] / out["distance_along"].max()).round(round_val)
    return out


# ----------------------------------------------------------------------
# ACROSS — distance orthogonale signée
# ----------------------------------------------------------------------

def _across_chunk_worker(dist_geom_chunk, anchor_geom_chunk, centerline, center):
    dist = shapely.distance(dist_geom_chunk, centerline)

    coords_anchor = shapely.get_coordinates(anchor_geom_chunk)
    coords_center = np.tile(shapely.get_coordinates(center), (len(anchor_geom_chunk), 1))
    line_coords = np.stack([coords_anchor, coords_center], axis=1)  # (n, 2, 2)
    connectors = shapely.linestrings(line_coords)

    crosses = shapely.intersects(connectors, centerline)
    return np.where(crosses, -dist, dist)


def _compute_across(dist_geom, anchor_geom, centerline, center,
                     chunk_size=200_000, n_jobs=1, show_progress=True):
    dist_arr = np.asarray(dist_geom)
    anchor_arr = np.asarray(anchor_geom)

    return _chunked_map(
        _across_chunk_worker, [dist_arr, anchor_arr], chunk_size, n_jobs, show_progress, "across",
        centerline=centerline, center=center,
    )

def compute_distances_to_axis(
    data,
    centerline: shapely.LineString,
    polygon: shapely.Polygon | None = None,
    mode: str = "both",                 # "along" | "across" | "both"
    key: str = "cell_boundaries",       # shape_key OU transcript/points key
    distance_type: str = "centroid",    # "centroid" | "cell" (ignoré si points)
    coordinates: list = ["x", "y"],
    genes: list | None = None,          # filtre gènes (transcripts uniquement)
    cells: list | None = None,          # filtre cellules (shapes ET transcripts)
    cell_id_col: str = "cell_id",       # colonne utilisée pour filtrer `cells`
    techno: str = "Xenium",             # "Xenium" | "Merscope" (transcripts uniquement)
    only_in_cell: bool = False,         # transcripts assignés à une cellule uniquement
    transform: bool = False,
    scale: bool = False,
    nb_interval: int = 10,
    bin_size: float | None = None,
    norm_scope: str = "global",       # NEW: "global" | "local"
    quantile_clip: tuple = (0.01, 0.99),  # NEW: percentiles pour robustesse aux outliers
    round_val: int = 3,
    n_jobs: int = 1,
    chunk_size: int = 200_000,
    show_progress: bool = True,
    return_df: bool = False,
):
    """
    Calcule la distance le long ('along') et/ou orthogonale signée ('across')
    par rapport à une centerline, sur une table de shapes (polygones,
    ex: cell_boundaries) ou de points (ex: transcripts).

    Parameters
    ----------
    data : sd.SpatialData | pd.DataFrame | gpd.GeoDataFrame | dd.DataFrame
        Si SpatialData, `key` désigne l'élément à utiliser (shapes ou points).
    centerline : shapely.LineString
        La centerline.
    polygon : shapely.Polygon, optionnel
        Requis si mode inclut 'across'. Son centroid sert d'ancre pour
        déterminer le signe (côté) de la distance orthogonale.
    mode : {'along', 'across', 'both'}
    key : str
        Nom de l'élément dans le sdata (shape_key ou transcript_key).
    distance_type : {'centroid', 'cell'}
        Ignoré pour les tables de points (toujours 'point' dans ce cas).
        'centroid' : distance mesurée depuis le centroid du polygone.
        'cell' : distance mesurée depuis le polygone entier.
        Dans les deux cas, l'ancre du test de signe reste le centroid.
    coordinates : list
        Noms des colonnes x/y si la donnée n'est pas déjà un GeoDataFrame.
    genes : list, optionnel
        Restreint aux transcripts de ces gènes. Ignoré pour les shapes.
        Si `key` pointe vers un élément transcripts d'un SpatialData, c'est
        `spt.pp.transcripts.subset_transcripts` qui applique ce filtre
        (avec le filtre qualité et le transform/scale des coordonnées).
    cells : list, optionnel
        Restreint aux cellules listées. Pour les shapes (cell_boundaries),
        filtre sur `cell_id_col` si la colonne existe, sinon sur l'index.
        Pour les transcripts, filtre sur `cell_id_col` si la colonne existe
        (utile combiné à `only_in_cell=True`).
    cell_id_col : str
        Nom de la colonne d'identifiant de cellule utilisée par `cells`.
    techno : {'Xenium', 'Merscope'}
        Technologie, transmise à `subset_transcripts` (transcripts uniquement).
    only_in_cell : bool
        Si True, ne garde que les transcripts assignés à une cellule
        (transcripts uniquement — transmis à `subset_transcripts`).
    norm_scope : {'global', 'local'}
        Portée de la normalisation de `dst_across_norm` (ignoré en mode 'along'
        seul, requiert 'both' pour 'local').
        'global' (défaut) : une seule échelle 0-1 pour tout le tissu, basée
            sur les percentiles `quantile_clip` de `distance_across`.
            Nécessaire si tu compares des cellules à des positions `along`
            différentes entre elles.
        'local' : une échelle 0-1 par tranche `cat_along` (donc dépendante
            de la largeur locale du tissu). Utile uniquement si tu compares
            des cellules à la même position `along`, entre groupes/conditions
            (auquel cas la comparaison inter-tranches n'est plus valide).
    quantile_clip : tuple(float, float)
        Percentiles bas/haut utilisés pour la normalisation robuste (clip
        des outliers avant mise à l'échelle 0-1).
    nb_interval : int
        Nombre d'intervalles réguliers le long de la centerline.
    n_jobs : int
        1 = séquentiel (mais chunké si besoin). >1 = multiprocessing.Pool,
        automatiquement plafonné à cpu_count().
        Utile surtout pour la mémoire sur de très grosses tables de transcripts.
    chunk_size : int
        Taille des chunks pour les calculs 'along'/'across'.
    return_df : bool
        Si True, retourne le GeoDataFrame avec les colonnes calculées.

    Returns
    -------
    gpd.GeoDataFrame ou None (selon return_df)
    """
    if mode not in ("along", "across", "both"):
        raise ValueError("mode must be 'along', 'across' or 'both'")
    if norm_scope not in ("global", "local", None):
        raise ValueError("norm_scope must be 'global' or 'local'")
    if norm_scope == "local" and mode != "both":
        raise ValueError("norm_scope='local' requires mode='both' (needs cat_along).")

    n_jobs = _clamp_n_jobs(n_jobs)

    gdf, is_points = _prepare_geodataframe(
        data, key, coordinates,
        genes=genes, cells=cells, cell_id_col=cell_id_col,
        techno=techno, only_in_cell=only_in_cell,
        transform=False, scale=False,
    )
    # gdf, is_points = _extract_geodataframe(data, key, coordinates)

    if is_points:
        dist_geom = gdf.geometry
        anchor_geom = gdf.geometry
    else:
        if distance_type == "centroid":
            dist_geom = gdf.geometry.centroid
        elif distance_type == "cell":
            dist_geom = gdf.geometry
        else:
            raise ValueError("distance_type must be 'centroid' or 'cell'")
        anchor_geom = gdf.geometry.centroid

    new_cols = []

    if mode in ("along", "both"):
        along_df = _compute_along(
            dist_geom, centerline, nb_interval, round_val,
            bin_size=bin_size,
            chunk_size=chunk_size, n_jobs=n_jobs, show_progress=show_progress,
        )
        gdf[along_df.columns] = along_df
        new_cols += list(along_df.columns)

    if mode in ("across", "both"):
        if polygon is None:
            raise ValueError(
                "`polygon` est requis pour calculer la distance 'across' "
                "(son centroid sert d'ancre pour le signe)."
            )
        center = polygon.centroid
        signed = _compute_across(
            dist_geom, anchor_geom, centerline, center,
            chunk_size=chunk_size, n_jobs=n_jobs, show_progress=show_progress,
        )
        gdf["cat_across"] = (signed > 0).astype(int)
        shifted = signed - signed.min()
        gdf["distance_across"] = shifted

        lo, hi = quantile_clip

        def _norm(x):
            arr = np.asarray(x)
            q_lo, q_hi = np.quantile(arr, [lo, hi])
            spread = q_hi - q_lo
            if spread == 0 or np.isnan(spread):
                result = np.full(arr.shape, 0.5)
            else:
                result = np.clip((arr - q_lo) / spread, 0, 1)
            if isinstance(x, pd.Series):
                return pd.Series(result, index=x.index)
            return result

        if norm_scope == "local":
            gdf["dst_across_norm"] = (
                gdf.groupby("cat_along")["distance_across"]
                .transform(_norm)
                .round(round_val)
            )
        elif norm_scope == "global":
            gdf["dst_across_norm"] = _norm(shifted).round(round_val)
        else:
            gdf["dst_across_norm"] = (shifted / shifted.max()).round(round_val)

        new_cols += ["cat_across", "distance_across", "dst_across_norm"]

    if transform:
        transformation = sd.transformations.get_transformation(data[key])

        gdf['geometry'] = compute_bounds_gpd(
            shape=gdf, 
            transfo=transformation, scale=scale
        )
    
    _write_back(data, key, gdf, new_cols)

    return gdf if return_df else None



# def _extract_geodataframe(data, key, coordinates):
#     """
#     Récupère un GeoDataFrame à partir d'un SpatialData, d'une dask.DataFrame,
#     d'un pandas.DataFrame ou d'un GeoDataFrame déjà prêt.

#     Retourne (gdf, is_points) où is_points indique si les géométries sont
#     des Points (transcripts) plutôt que des Polygones (shapes).
#     """
#     if isinstance(data, sd.SpatialData):
#         df = data[key]
#     else:
#         df = data

#     if isinstance(df, dd.DataFrame):
#         df = df.compute()
#     else:
#         df = df.copy()

#     if not isinstance(df, gpd.GeoDataFrame):
#         df = gpd.GeoDataFrame(
#             df, geometry=gpd.points_from_xy(df[coordinates[0]], df[coordinates[1]])
#         )

#     is_points = bool((df.geometry.geom_type == "Point").all())
#     return df, is_points





# # ----------------------------------------------------------------------
# # ALONG — distance projetée sur la centerline
# # ----------------------------------------------------------------------

# def _compute_along(dist_geom, centerline, nb_interval, round_val):
#     pts_arr = np.asarray(dist_geom)
#     dist_along = shapely.line_locate_point(centerline, pts_arr)

#     labels = [str(i) for i in range(nb_interval)]
#     bin_size = centerline.length / nb_interval
#     cuts = np.arange(0, centerline.length + bin_size, bin_size)

#     out = pd.DataFrame(index=dist_geom.index)
#     out["distance_along"] = dist_along
#     out["cat_along"] = pd.cut(
#         out["distance_along"], bins=cuts, labels=labels, right=True, include_lowest=True
#     )
#     out["dst_along_norm"] = (out["distance_along"] / out["distance_along"].max()).round(round_val)
#     return out


# # ----------------------------------------------------------------------
# # ACROSS — distance orthogonale signée, chunkée / parallélisable
# # ----------------------------------------------------------------------

# def _compute_across_chunk(dist_geom_chunk, anchor_geom_chunk, centerline, center):
#     """Un chunk = deux tableaux numpy de géométries de même longueur."""
#     dist = shapely.distance(dist_geom_chunk, centerline)

#     coords_anchor = shapely.get_coordinates(anchor_geom_chunk)
#     coords_center = np.tile(shapely.get_coordinates(center), (len(anchor_geom_chunk), 1))
#     line_coords = np.stack([coords_anchor, coords_center], axis=1)  # (n, 2, 2)
#     connectors = shapely.linestrings(line_coords)

#     crosses = shapely.intersects(connectors, centerline)
#     return np.where(crosses, -dist, dist)


# def _compute_across_chunk_star(chunk, centerline, center):
#     dist_geom_chunk, anchor_geom_chunk = chunk
#     return _compute_across_chunk(dist_geom_chunk, anchor_geom_chunk, centerline, center)


# def _compute_across(dist_geom, anchor_geom, centerline, center,
#                      chunk_size=200_000, n_jobs=1, show_progress=True):
#     n = len(dist_geom)
#     dist_arr = np.asarray(dist_geom)
#     anchor_arr = np.asarray(anchor_geom)

#     if n <= chunk_size:
#         return _compute_across_chunk(dist_arr, anchor_arr, centerline, center)

#     chunks = [
#         (dist_arr[i:i + chunk_size], anchor_arr[i:i + chunk_size])
#         for i in range(0, n, chunk_size)
#     ]

#     if n_jobs and n_jobs > 1:
#         worker = partial(_compute_across_chunk_star, centerline=centerline, center=center)
#         with Pool(n_jobs) as pool:
#             results = list(
#                 tqdm(pool.imap(worker, chunks), total=len(chunks),
#                      disable=not show_progress, desc="across (multiprocessing)")
#             )
#     else:
#         results = [
#             _compute_across_chunk(d, a, centerline, center)
#             for d, a in tqdm(chunks, disable=not show_progress, desc="across (chunked)")
#         ]

#     return np.concatenate(results)


# def compute_distances_to_axis(
#     data,
#     centerline: shapely.LineString,
#     polygon: shapely.Polygon | None = None,
#     mode: str = "both",                 # "along" | "across" | "both"
#     key: str = "cell_boundaries",       # shape_key OU transcript/points key
#     distance_type: str = "centroid",    # "centroid" | "cell" (ignoré si points)
#     coordinates: list = ["x", "y"],
#     nb_interval: int = 10,
#     round_val: int = 3,
#     n_jobs: int = 1,
#     chunk_size: int = 200_000,
#     show_progress: bool = True,
#     return_df: bool = False,
# ):
#     if mode not in ("along", "across", "both"):
#         raise ValueError("mode must be 'along', 'across' or 'both'")

#     gdf, is_points = _extract_geodataframe(data, key, coordinates)

#     if is_points:
#         dist_geom = gdf.geometry     # les transcripts sont déjà des points
#         anchor_geom = gdf.geometry
#     else:
#         if distance_type == "centroid":
#             dist_geom = gdf.geometry.centroid
#         elif distance_type == "cell":
#             dist_geom = gdf.geometry
#         else:
#             raise ValueError("distance_type must be 'centroid' or 'cell'")
#         anchor_geom = gdf.geometry.centroid  # toujours le centroid pour le signe

#     new_cols = []

#     if mode in ("along", "both"):
#         along_df = _compute_along(dist_geom, centerline, nb_interval, round_val)
#         gdf[along_df.columns] = along_df
#         new_cols += list(along_df.columns)

#     if mode in ("across", "both"):
#         if polygon is None:
#             raise ValueError(
#                 "`polygon` est requis pour calculer la distance 'across' "
#                 "(son centroid sert d'ancre pour le signe)."
#             )
#         center = polygon.centroid
#         signed = _compute_across(
#             dist_geom, anchor_geom, centerline, center,
#             chunk_size=chunk_size, n_jobs=n_jobs, show_progress=show_progress,
#         )
#         gdf["cat_across"] = (signed > 0).astype(int)
#         shifted = signed - signed.min()
#         gdf["distance_across"] = shifted
#         gdf["dst_across_norm"] = (shifted / shifted.max()).round(round_val)
#         new_cols += ["cat_across", "distance_across", "dst_across_norm"]

#     _write_back(data, key, gdf, new_cols)

#     return gdf if return_df else None





def fromAxisMedialToDf(
    data: sd.SpatialData | pd.DataFrame | gpd.GeoDataFrame, # NEW choice of multiple input, BEFORE only sdata
    axisMedial: shapely.LineString,
    nb_interval: int = 10,
    shape_key: str = 'cell_boundaries',
    group_by: str = "cell_type_pred",
    coordinates: list = ['x', 'y'], # NEW
    # group_lst: '[]|None' = None,
    # scale_factor: float = 1/0.2125, # because xenium in microns and sdata in global
    return_df: bool = False,
):
    """Compute 'nb_interval' regular intervals along the centerline

    Parameters
    ---------- 
    sdata (sd.SpatialData): _description_
    axisMedial (shapely.LineString): _description_
    nb_interval (int, optional): _description_. Defaults to 10.
    shape_key (str, optional): _description_. Defaults to 'cell_boundaries'.
    group_by (str, optional): _description_. Defaults to "cell_type_pred".
    scale_factor (float, optional): _description_. Defaults to 1/0.2125.

    Returns
    -------
        _type_: _description_
    """
    if isinstance(data,sd.SpatialData):
        df_along = data[shape_key].copy()
    elif isinstance(data,pd.DataFrame):
        df_along = data.copy()

    if not isinstance(df_along, gpd.GeoDataFrame):
        df_along = gpd.GeoDataFrame(df_along, 
                                      geometry=gpd.points_from_xy(df_along[coordinates[0]], 
                                                                  df_along[coordinates[1]]))

    labels = [str(i) for i in range(nb_interval)]
    bin_size = axisMedial.length / nb_interval
    x = np.arange(0, axisMedial.length, bin_size)
    x = np.append(x, axisMedial.length)
    # interval_dict = {pos: shapely.line_interpolate_point(axis_medial, pos) for pos in x}
    
    # if group_lst == None:
    #     group_lst = list(sdata.table.obs[group_by].unique())
    df_along["distance_along"] = shapely.line_locate_point(axisMedial, df_along.centroid)
    # Return the distance to the line origin of given point. 
    # If given point does not intersect with the line, the point will first be projected onto the line after which the distance is taken

    # find pts sur la ligne le plus proche de la cellule
    # df_along["closest_point_on_line"] = df_along.apply(
    #     lambda row: shapely.ops.nearest_points(row.geometry.centroid, axisMedial)[1] , axis = 1)
    # # distance du pts start au point closest (plus proche de la cellule)
    # # ATTENTION il faut faire la distance sur la courbe
    # df_along["distance_along"] = df_along['closest_point_on_line'].apply(
    #     lambda row: axisMedial.project(row))
  
    # colonne distance en colonne categorie par rapport a x si distance entre 0 et 1 label 0 etc.
    df_along['cat_along'] = pd.cut(df_along['distance_along'], 
                             bins=x, labels=labels, right=True, include_lowest=True)
    df_along['dst_along_norm']  = (df_along['distance_along'] / df_along['distance_along'].max()).round(3)
    
    if isinstance(data,sd.SpatialData):
        data[shape_key][['cat_along', 'dst_along_norm']] = df_along[['cat_along', 'dst_along_norm']]
    elif isinstance(data,pd.DataFrame):
        data[['cat_along', 'dst_along_norm']] = df_along[['cat_along', 'dst_along_norm']]

    if return_df:
        return df_along
    else: 
        return
    
    
def df_for_genes(
    sdata: sd.SpatialData,
    axisMedial: shapely.LineString,
    genes: str | list,
    nb_interval: int = 10,
    transcript_key: str = 'transcripts',
    feature_key: str = 'feature_name',
    qv: int = 20,
    # shape_key: 'str' = 'cell_boundaries',
    group_by: str = "cell_type",
    # group_lst: '[]|None' = None,
    # scale_factor: 'float' = 1/0.2125, # because xenium in microns and sdata in global
    # return_df: 'bool' = False,
):
    """Calculate the number of transcripts for a list of genes in 'nb_interval' regulars intervals

    Parameters
    ----------
    sdata (sd.SpatialData): _description_
    axisMedial (shapely.LineString): _description_
    genes (str | list): _description_
    nb_interval (int, optional): _description_. Defaults to 10.
    transcript_key (str, optional): _description_. Defaults to 'transcripts'.
    feature_key (str, optional): _description_. Defaults to 'feature_name'.
    qv (int, optional): _description_. Defaults to 20.
    group_by (str, optional): _description_. Defaults to "cell_type".

    Returns
    -------
    df_trans_sub: _description_
    """
    labels = [str(i) for i in range(nb_interval)]

    bin_size = axisMedial.length / nb_interval
    x = np.arange(0, axisMedial.length, bin_size)
    x = np.append(x, axisMedial.length)
    # interval_df = pd.DataFrame(x, columns= ["position"])

    # # depuis le start -> find pts a une distance de 'position'
    # interval_df['point'] = interval_df.apply(lambda row: shapely.line_interpolate_point(axis_medial, row.position) , axis = 1)
    # interval_df
    
    df_transcripts = sdata[transcript_key].compute()
    df_trans_sub = df_transcripts[df_transcripts[feature_key].isin(genes)]
    df_trans_sub = df_trans_sub[df_trans_sub['qv'] >= qv]

    # find pts sur la ligne le plus proche de la cellule
    df_trans_sub["closest"] = df_trans_sub.apply(lambda row: 
        shapely.ops.nearest_points(
            shapely.Point([row.x, row.y]),
            axisMedial)[1] , axis = 1)

    # distance du pts start au point closest (plus proche de la cellule)
    # ATTENTION il faut faire la distance sur la courbe
    df_trans_sub["distance"] = df_trans_sub['closest'].apply(
        lambda row: axisMedial.project(row))

    # colonne distance en colonne categorie par rapport a x si distance entre 0 et 1 label 0 etc.
    df_trans_sub['cat'] = pd.cut(df_trans_sub['distance'], 
                                 bins=x, labels=labels, right=True)
    df_trans_sub[feature_key] = df_trans_sub[feature_key].cat.remove_unused_categories()
    df_trans_sub = df_trans_sub.merge(sdata['table'].obs[['cell_id', group_by]], 
                                      on = 'cell_id', how = 'left')
    # how = left -> else remove transcript not assign to a cell ? 
    # can be put by default on inner to remove transcript not assigned to a cell
    return df_trans_sub



def _centroid_intersects(point, centroid, line, distance):
    if shapely.LineString([point, centroid]).intersects(line):
        return -distance
    else:
        return distance

def orthogonalDistance(
    data: sd.SpatialData | pd.DataFrame | gpd.GeoDataFrame,
    polygon: shapely.Polygon, 
    centerline: shapely.LineString,
    shape_key: str = 'cell_boundaries',
    # group_by: str | None = None,
    # distance: int = 30,
    distance : str = 'centroid',
    round: int = 3,
    return_df: bool = False,
    coordinates: list = ['x', 'y'],
) -> gpd.GeoDataFrame:  
    """Normalize the distance by following the othogonal axis

    Parameters
    ----------
        data (pd.DataFrame | gpd.GeoDataFrame): _description_
        polygon (shapely.Polygon): _description_
        centerline (shapely.LineString): _description_
        group_by (str | None, optional): _description_. Defaults to None.
        distance (int, optional): _description_. Defaults to 30.
        round (int, optional): _description_. Defaults to 3.

    Returns
    -------
        gpd.GeoDataFrame: _description_
    """
    if isinstance(data, sd.SpatialData):
        df_compute = data[shape_key].copy()
    elif isinstance(data, pd.DataFrame):
        df_compute = data.copy()

    if not isinstance(df_compute, gpd.GeoDataFrame):
        df_compute = gpd.GeoDataFrame(df_compute, 
                                      geometry=gpd.points_from_xy(df_compute[coordinates[0]], 
                                                                  df_compute[coordinates[1]]))
    
    if distance == 'centroid':
        df_compute['distance_to_line'] = df_compute.centroid.distance(centerline)
        dist_along = shapely.line_locate_point(centerline, df_compute.centroid)
        df_compute["project_on_line"] = shapely.line_interpolate_point(centerline, dist_along)

        # df_compute['distance_to_line'] = df_compute.centroid.distance(centerline)
        # df_compute['project_on_line'] = centerline.interpolate(centerline.project(df_compute.centroid))
    elif distance == 'cell':
        df_compute['distance_to_line'] = df_compute.centroid.distance(centerline)
        dist_along = shapely.line_locate_point(centerline, df_compute.centroid)
        df_compute["project_on_line"] = shapely.line_interpolate_point(centerline, dist_along)

        # df_compute['distance_to_line'] = df_compute.distance(centerline)
        # df_compute['project_on_line'] = centerline.interpolate(centerline.project(df_compute))
    else:
        print("Distance unknown. Please select centroid or cell.")
        return

    pol_ctr = polygon.centroid
    # check if the line between cell and shape's centroid intersect the centerline or not 
    # distance < 0 if intersect and > 0 if not 
    df_compute['distance'] = df_compute.apply(
        lambda row: _centroid_intersects(row['geometry'].centroid, 
                                        pol_ctr, centerline, row['distance_to_line']),
                                        axis=1)


    df_compute['cat_orth'] = 0
    df_compute.loc[df_compute['distance'] > 0, 'cat_orth'] = 1
    df_compute['distance'] -= df_compute['distance'].min()
    df_compute['dst_orth_norm'] = (df_compute['distance'] / df_compute['distance'].max()).round(round)
    
    if isinstance(data, sd.SpatialData):
        data[shape_key][['cat_orth', 'dst_orth_norm']] = df_compute[['cat_orth', 'dst_orth_norm']]
    elif isinstance(data, pd.DataFrame):
        data[['cat_orth', 'dst_orth_norm']] = df_compute[['cat_orth', 'dst_orth_norm']]

    if return_df:
        return df_compute
    else:
        return

