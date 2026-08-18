from spatialdata.transformations import get_transformation
import geopandas as gpd
import pandas as pd

from .._registry import TECHNO_REGISTRY
from .._constants import GENE_EXCLUDE_PATTERN #, XeniumKeys, MerscopeKeys, 
from .._validation import (
    _assert_spatialdata,
    _assert_element_in_sdata,
    _assert_technology_supported,
    _assert_positive,
    _assert_str_or_list_of_str,
)

from ..pp.transformations import compute_bounds_dask

def subset_transcripts(
    sdata,
    genes: str | list = None,
    cells: list = None,
    qv: int = 20,
    transcript_key: str= "transcripts",
    techno = "Xenium", # 'Xenium' or 'Merscope'
    only_in_cell: bool = True,
    only_outside: bool = False,
    gene_exclude_pattern: str = GENE_EXCLUDE_PATTERN,
    # feature_key: str = 'feature_name',
    # cell_key: str = "cell_id",
    transform: bool = True,
    scale: str = False,
    copy: bool = True,
    return_gpd: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Filter transcripts dask dataframe.

    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    _assert_spatialdata(sdata)
    techno = _assert_technology_supported(techno)
    _assert_element_in_sdata(sdata, transcript_key)
    _assert_positive(qv, "qv", strict=False)

    genes = _assert_str_or_list_of_str(genes, "genes")
    cells = _assert_str_or_list_of_str(cells, "cells")

    config = TECHNO_REGISTRY[techno]
    Keys = config["keys"]
    filter_fn = config["filter_fn"]

    ## WARNINGS FOR MERSCOPE TRANSCRIPT_KEY not always the same
    # df_transcripts = sdata[Keys.TRANSCRIPT_KEY].copy() if copy else sdata[Keys.TRANSCRIPT_KEY]
    df_transcripts = sdata[transcript_key].copy() if copy else sdata[transcript_key]
    
    if transform:
        df_transcripts = compute_bounds_dask(
            transcripts=df_transcripts, 
            transfo=get_transformation(df_transcripts), 
            scale=scale)

    df_transcripts = filter_fn(
            df=df_transcripts,
            genes=genes,
            cells=cells,
            qv=qv,
            only_in_cell=only_in_cell,
            only_outside=only_outside,
            gene_exclude_pattern=gene_exclude_pattern
        )
    
    df_transcripts = df_transcripts.compute() if hasattr(df_transcripts, "compute") else df_transcripts
    df_transcripts[Keys.FEATURE_KEY] = df_transcripts[Keys.FEATURE_KEY].cat.remove_unused_categories()
    # df_transcripts[Keys.CELL_ID] = df_transcripts[Keys.CELL_ID].cat.remove_unused_categories()

    if (return_gpd) and (not isinstance(df_transcripts, gpd.GeoDataFrame)):
        print("Create geopandas...")
        df_transcripts = gpd.GeoDataFrame(
            df_transcripts,
            geometry=gpd.points_from_xy(
                df_transcripts["x"],
                df_transcripts["y"]
            )
        )
        
    return df_transcripts
 


