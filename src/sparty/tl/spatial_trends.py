import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from shapely import LineString, Point, get_coordinates, affinity, Polygon
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, Identity, Translation, set_transformation, get_transformation, Sequence
import shapely



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

