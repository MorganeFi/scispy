import numpy as np
import dask.dataframe as dd
import geopandas as gpd
from shapely import affinity
from shapely.geometry.base import BaseGeometry
from spatialdata.transformations import Identity, Scale, Sequence


# ---------------------------------------------------------------------
# Resolve which transformation to apply (independent of the object type)
# ---------------------------------------------------------------------

def _resolve_transfo(transfo, scale=False):
    """
    Determine the effective transformation to apply.

    If `scale=False`, any `Scale` components (standalone or inside a
    `Sequence`) are ignored.

    Returns
    -------
    transformation or None
        None means no transformation should be applied
        (Identity, a standalone Scale with scale=False, or a Sequence
        that reduces to nothing once Scale components are stripped).
    """
    if isinstance(transfo, Identity):
        return None

    if isinstance(transfo, Scale):
        return transfo if scale else None

    if isinstance(transfo, Sequence):
        if scale:
            return transfo
        new_tr = [t for t in transfo.transformations if not isinstance(t, Scale)]
        if len(new_tr) == 0:
            return None
        if len(new_tr) == 1:
            return new_tr[0]
        return Sequence(new_tr)

    return transfo


# ---------------------------------------------------------------------
# Actual application, per object type
# ---------------------------------------------------------------------

def _to_affine_shapely(M):
    """Convert a spatialdata affine matrix to a shapely tuple (a, b, d, e, xoff, yoff)"""
    a, b, xoff = M[0]
    d, e, yoff = M[1]
    return (a, b, d, e, xoff, yoff)


def _transform_coords(ddf, M):
    """Matrix transformation for a Dask partition"""
    coords = np.vstack([ddf['x'], ddf['y'], np.ones(len(ddf))])
    transformed = M @ coords
    ddf['x'] = transformed[0]
    ddf['y'] = transformed[1]
    return ddf


def apply_affine_dask(transcripts, transfo):
    M = transfo.to_affine_matrix(
        input_axes=("x", "y"), 
        output_axes=("x", "y"))
     # NO .T transpose else we get issue !! .T # .T transpose or not ???
    return transcripts.map_partitions(_transform_coords, M)


def apply_affine_gpd(shape, transfo):
    """Apply an affine transformation to a GeoDataFrame / GeoSeries"""
    M = transfo.to_affine_matrix(
        input_axes=("x", "y"), 
        output_axes=("x", "y")) # .T transpose or not ???
    A = _to_affine_shapely(M)
    return shape.affine_transform(A)


def apply_affine_shapely(shape, transfo):
    """Apply an affine transformation to a shapely geometry (Polygon, LineString, Point...)"""
    M = transfo.to_affine_matrix(
        input_axes=("x", "y"), 
        output_axes=("x", "y"))
    A = _to_affine_shapely(M)
    return affinity.affine_transform(shape, A)


# ---------------------------------------------------------------------
# Single entry point
# ---------------------------------------------------------------------

def transform(obj, transfo, scale=False):
    """
    Apply a spatialdata transformation to `obj`, regardless of its type.

    Automatic dispatch:
        - dask.dataframe.DataFrame          -> apply_affine_dask
        - geopandas.GeoSeries/GeoDataFrame  -> apply_affine_gpd
        - shapely geometry (Polygon, LineString, Point, MultiPolygon...) -> apply_affine_shapely

    By default (`scale=False`), the `Scale` component(s) of the
    transformation are ignored (useful when coordinates are already at
    the desired scale, e.g. `center_x`/`center_y` already in microns).

    Parameters
    ----------
    obj : dask.dataframe.DataFrame | geopandas.GeoSeries | geopandas.GeoDataFrame | shapely.Geometry
        Object to transform.
    transfo : spatialdata.transformations.BaseTransformation
        Transformation to apply (Identity, Scale, Affine, Sequence...).
    scale : bool, default False
        If False, `Scale` components are ignored in the transformation.

    Returns
    -------
    Same type as `obj`, transformed (or unchanged if transfo resolves to Identity).
    """
    resolved = _resolve_transfo(transfo, scale=scale)
    if resolved is None:
        return obj

    if isinstance(obj, dd.DataFrame):
        return apply_affine_dask(obj, resolved)
    elif isinstance(obj, (gpd.GeoSeries, gpd.GeoDataFrame)):
        return apply_affine_gpd(obj, resolved)
    elif isinstance(obj, BaseGeometry):
        return apply_affine_shapely(obj, resolved)
    else:
        raise TypeError(
            f"Unsupported type for transform(): {type(obj)}. "
            "Expected: dask.DataFrame, geopandas.GeoSeries/GeoDataFrame, or a shapely geometry."
        )

    
    
def scale_line(line, scale= 0.2125):
    ln_scale = shapely.get_coordinates(line) *  scale
    return shapely.LineString(ln_scale)

def scale_pol(polygon, scale= 0.2125):
    pol_scale = shapely.get_coordinates(polygon) *  scale
    return shapely.Polygon(pol_scale)





def compute_bounds_dask(transcripts, transfo, scale=False):
    """Apply a transformation to a Dask DataFrame
    
    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    if not (isinstance(transfo, Identity) or isinstance(transfo, Scale) and not scale):
        if isinstance(transfo, Sequence):
            if scale:
                return apply_affine_dask(transcripts, transfo)
            else:
                # Ignorer les Scale dans la sequence
                new_transfo = [t for t in transfo.transformations if not isinstance(t, Scale)]
                if len(new_transfo) == 1:
                    transfo_to_apply = new_transfo[0]
                elif len(new_transfo) > 1:
                    transfo_to_apply = Sequence(new_transfo)
                else:
                    transfo_to_apply = None  

                if transfo_to_apply is not None:
                    return apply_affine_dask(transcripts, transfo_to_apply)
        else:
            return apply_affine_dask(transcripts, transfo)
    return transcripts


def compute_bounds_gpd(shape, transfo, scale=False):
    """
    Computes the bounds (xmin, ymin, xmax, ymax) based on the applied transformation.
    
    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    if (isinstance(transfo, Identity)) or (isinstance(transfo, Scale) and not scale):
        return shape

    elif isinstance(transfo, Sequence):
        if scale:
            return apply_affine_gpd(shape, transfo)
        else:
            # ignored scale in sequence
            new_tr = [t for t in transfo.transformations if not isinstance(t, Scale)]
            if len(new_tr) == 0:
                return shape
            elif len(new_tr) == 1:
                transfo_to_apply = new_tr[0]
            else:
                transfo_to_apply = Sequence(new_tr)
            return apply_affine_gpd(shape, transfo_to_apply)
    else:
        return apply_affine_gpd(shape, transfo)


def compute_bounds_shapely(shape, transfo, scale=False):
    """
    Applique une transformation à une géométrie shapely (Polygon, LineString, Point...)

    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    if (isinstance(transfo, Identity)) or (isinstance(transfo, Scale) and not scale):
        return shape

    elif isinstance(transfo, Sequence):
        if scale:
            return apply_affine_shapely(shape, transfo)
        else:
            # ignore Scale dans la séquence
            new_tr = [t for t in transfo.transformations if not isinstance(t, Scale)]
            if len(new_tr) == 0:
                return shape
            elif len(new_tr) == 1:
                transfo_to_apply = new_tr[0]
            else:
                transfo_to_apply = Sequence(new_tr)
            return apply_affine_shapely(shape, transfo_to_apply)
    else:
        return apply_affine_shapely(shape, transfo)