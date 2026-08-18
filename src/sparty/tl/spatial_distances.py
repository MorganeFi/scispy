import math
import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata import SpatialData

from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Translation, set_transformation, get_transformation, Sequence

import shapely
from shapely.geometry import Point
from shapely.ops import nearest_points

from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from ..tl.unfolding import extendLine

def _process_batch_orth(batch_df, distance_type, center, centerline):
    # Distance centroid ou géométrie
    if distance_type == 'centroid':
        batch_df['distance_to_line'] = batch_df.centroid.distance(centerline)
        batch_df['project_on_line'] = batch_df.centroid.apply(
            lambda c: centerline.interpolate(centerline.project(c))
        )
    elif distance_type == 'cell':
        batch_df['distance_to_line'] = batch_df.distance(centerline)
        batch_df['project_on_line'] = batch_df.geometry.apply(
            lambda g: centerline.interpolate(centerline.project(g))
        )
    else:
        raise ValueError("distance_type must be 'centroid' or 'cell'")

    # Orientation du signe (+/-) selon intersection
    def _signed(row):
        if shapely.LineString([row.geometry.centroid, center]).intersects(centerline):
            return -row.distance_to_line
        return row.distance_to_line

    batch_df["distance_orth"] = batch_df.apply(_signed, axis=1)
    return batch_df[["distance_orth"]]


def compute_geometric_distances(
    data,
    axisMedial: shapely.LineString,
    polygon: shapely.Polygon | None = None,
    mode: str = "both",                # "along" | "orth" | "both"
    nb_interval: int = 10,
    shape_key: str = 'cell_boundaries',
    coordinates: list = ['x', 'y'],
    distance_type: str = 'centroid',    # centroid or cell for orth
    round_val: int = 3,
    batch_size: int = 100_000,
    return_df: bool = False
):
    """
    Fonction unifiée pour :
    - distance ALONG une ligne (projection + bins)
    - distance ORTHOGONALE à une ligne/polygone
    """

    # ----------- Préparation des données -----------
    if "SpatialData" in str(type(data)):
        df = data[shape_key].copy()
    else:
        df = data.copy()

    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[coordinates[0]], df[coordinates[1]])
        )

    # ------------------------------------------------
    # 1) ALONG — distance le long de l’axe médial
    # ------------------------------------------------
    if mode in ("along", "both"):

        labels = [str(i) for i in range(nb_interval)]
        bin_size = axisMedial.length / nb_interval
        cuts = np.arange(0, axisMedial.length + bin_size, bin_size)

        df["distance_along"] = shapely.line_locate_point(axisMedial, df.centroid)

        # distance projetée sur la ligne
        df["distance_along"] = df["closest_point_on_line"].apply(
            lambda p: axisMedial.project(p)
        )

        df['cat_along'] = pd.cut(
            df['distance_along'], bins=cuts, labels=labels,
            right=True, include_lowest=True
        )

        df["dst_along_norm"] = (
            df["distance_along"] / df["distance_along"].max()
        ).round(round_val)

    # ------------------------------------------------
    # 2) ORTH — distance orthogonale (multiprocessing)
    # ------------------------------------------------
    if mode in ("orth", "both"):

        if polygon is None:
            raise ValueError("polygon must be provided when computing orthogonal distances")

        center = polygon.centroid

        batches = [
            df.iloc[i:i + batch_size].copy()
            for i in range(0, len(df), batch_size)
        ]

        worker = partial(
            _process_batch_orth,
            distance_type=distance_type,
            center=center,
            centerline=axisMedial
        )

        results = []
        with Pool(cpu_count()) as pool:
            for res in tqdm(pool.imap(worker, batches), total=len(batches),
                            desc="Orthogonal distance"):
                results.append(res)

        orth = pd.concat(results)
        orth["distance_orth"] -= orth["distance_orth"].min()
        orth["dst_orth_norm"] = (
            orth["distance_orth"] / orth["distance_orth"].max()
        ).round(round_val)

        df[["distance_orth", "dst_orth_norm"]] = orth[["distance_orth", "dst_orth_norm"]]
        df["cat_orth"] = (df["distance_orth"] > 0).astype(int)

    # ------------------------------------------------
    # Output
    # ------------------------------------------------
    if isinstance(data, sd.SpatialData):
        data[shape_key][df.columns] = df
    elif isinstance(data, pd.DataFrame):
        data[df.columns] = df
    else:
        Warning("Only accept spatialdata or datafram")

    return df if return_df else None



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
    
    if isinstance(data, sd.SpatialData):
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


def centroid_intersects(point, centroid, line, distance):
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
        lambda row: centroid_intersects(row['geometry'].centroid, 
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




# def find_polygon(geometry, up, down):
#     if up.intersects(geometry.centroid):
#         return 1
#     elif down.intersects(geometry.centroid):
#         return 2
#     elif up.intersects(geometry):
#         return 1
#     elif down.intersects(geometry):
#         return 2
#     else:
#         return 0


# def orthogonalDistance(
#     data: pd.DataFrame | gpd.GeoDataFrame,
#     polygon: shapely.Polygon, 
#     centerline: shapely.LineString,
#     # shape_key: str = 'cell_boundaries',
#     group_by: str | None = None,
#     distance: int = 30,
#     round: int = 3,
# ) -> gpd.GeoDataFrame:  
   

#     gdf_polygons = gpd.GeoDataFrame({'cat_layers': [1, 2]}, geometry=[up_shape, down_shape])
#     df_compute = gpd.sjoin(df_compute, gdf_polygons, predicate="intersects", how="left")
#     # type(df_trans_sub) # geopandas.geodataframe.GeoDataFrame

#     df_compute.loc[df_compute['cat_layers'] == 1, 'distance_pts_line'] *= -1
#     df_compute['distance_pts_line'] -= df_compute['distance_pts_line'].min()
#     # print(df_compute['distance_pts_line'].min())
#     df_compute['distance_normalize']  = (df_compute['distance_pts_line'] / df_compute['distance_pts_line'].max()).round(round)
    
#     return df_compute



    
# def orthogonalDistance(
#     sdata: sd.SpatialData,
#     polygon: shapely.Polygon, 
#     centerline: shapely.LineString,
#     shape_key: str = 'cell_boundaries',
#     distance: int = 30,
#     round: int = 3,
# ):
#     if len(shapely.ops.split(polygon, centerline).geoms) == 1 :
#         order_centers= shapely.get_coordinates(centerline)
#         extendedLine_start = scis.tl.unfolding.extendLine(order_centers[0, :], 
#                                         order_centers[1, :], distance=distance)
#         extendedLine_end = scis.tl.unfolding.extendLine(order_centers[-1, :], 
#                                         order_centers[-2, :], distance=distance)
#         lineFinal = shapely.LineString(np.vstack([shapely.get_coordinates(extendedLine_start)[0], 
#                                                 order_centers,
#                                                 shapely.get_coordinates(extendedLine_end)[0]]))
#         split_shapes = shapely.ops.split(polygon, lineFinal)
        
#         if len(split_shapes.geoms) == 2:
#             up_shape = split_shapes.geoms[0]
#             down_shape = split_shapes.geoms[1]
#         else:
#             print(len(split_shapes.geoms))
#             print("Increase distance")
#             return
    
#     sdata[shape_key]["distance_pts_line"] = sdata[shape_key]["geometry"].apply(
#         lambda row: shapely.distance(row.centroid, centerline))
#     sdata[shape_key]['cat_layers']  = sdata[shape_key]["geometry"].apply(
#         lambda row: find_polygon(row, up_shape,down_shape))    
#     sdata[shape_key].loc[sdata[shape_key]['cat_layers'] == 1, 'distance_pts_line'] *= -1
#     sdata[shape_key]['distance_pts_line'] -= sdata[shape_key]['distance_pts_line'].min()
#     sdata[shape_key]['distance_normalize']  = (sdata[shape_key]['distance_pts_line'] / sdata[shape_key]['distance_pts_line'].max()).round(round)
#     # print(sdata[shape_key])
