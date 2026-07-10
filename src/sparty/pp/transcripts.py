from spatialdata.transformations import get_transformation
import geopandas as gpd
import pandas as pd

from ..registry import TECHNO_REGISTRY
from ..constants import GENE_EXCLUDE_PATTERN #, XeniumKeys, MerscopeKeys, 
from ..pp.transformations import compute_bounds_dask

def subset_transcripts(
    sdata,
    genes: str | list = None,
    qv: int = 20,
    transcript_key: str= "transcripts",
    techno = "Xenium", # 'Xenium' or 'Merscope'
    only_in_cell: bool = True,
    only_outside: bool = False,
    gene_exclude_pattern = GENE_EXCLUDE_PATTERN,
    feature_key: str = 'feature_name',
    transform: bool = True,
    scale: str = False,
    copy: bool = True,
    return_gpd: bool = False,
):
    if techno not in TECHNO_REGISTRY:
        raise ValueError(f"Techno '{techno}' not supported. Available: {list(TECHNO_REGISTRY.keys())}")
    
    if type(genes) == str:
        genes = [genes]

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
            qv=qv,
            only_in_cell=only_in_cell,
            only_outside=only_outside,
            gene_exclude_pattern=gene_exclude_pattern
        )
    
    df_transcripts = df_transcripts.compute() if hasattr(df_transcripts, "compute") else df_transcripts
    df_transcripts[Keys.FEATURE_KEY] = df_transcripts[Keys.FEATURE_KEY].cat.remove_unused_categories()
    
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
 


# def _subset_transcripts(
#     sdata,
#     genes: list,
#     qv: int = 20,
#     transcript_key: str= "transcript",
#     feature_key: str = 'feature_name',
#     gene_exclude_pattern = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*",
#     copy: bool = True,
# ):
#     if copy: 
#         df_transcripts = sdata[transcript_key].copy()
#     else:
#         df_transcripts = sdata[transcript_key]

#     df_transcripts = df_transcripts[(df_transcripts['qv'] >= qv) & 
#                                     (df_transcripts.is_gene) & 
#                                     (df_transcripts.cell_id != "UNASSIGNED") &
#                                     (df_transcripts[feature_key].isin(genes))
#                                     ].dropna(subset=[feature_key])
#     df_transcripts = df_transcripts[~(df_transcripts[feature_key].str.contains(gene_exclude_pattern, regex=True))].compute()
#     df_transcripts[feature_key] = df_transcripts[feature_key].cat.remove_unused_categories()
#     return df_transcripts


def compute_stat_in_cells(
    sdata,
    sample: str | None = None,
    genes: str | list | None = None,
    # data: pd.DataFrame,
    group_by: list = ['feature_name', 'cell_id'],
    qv: int = 20,
    techno = "Xenium", # 'Xenium' or 'Merscope'
    # gene_exclude_pattern = GENE_EXCLUDE_PATTERN,
    feature_key: str = 'feature_name',
    transcript_key: str= "transcripts",
    nucleus_key: str = 'overlaps_nucleus',
) -> pd.DataFrame:
    
    data = subset_transcripts(
        sdata=sdata,
        genes=genes,
        qv=qv,
        transcript_key=transcript_key,
        techno = techno, # 'Xenium' or 'Merscope'
        only_in_cell = True,
        only_outside = False,
        feature_key=feature_key,
        # transform = True,
        # scale = False,
        # copy = True,
        return_gpd = False,
    )

    percent_in_nucleus = data.groupby(group_by, observed = True)[nucleus_key].mean() * 100

    result = pd.DataFrame({
        'percent_in_nucleus': percent_in_nucleus,
        'percent_in_cytoplasm': 100 - percent_in_nucleus
    })
    result['counts'] = data.groupby(group_by, observed = True)[group_by[0]].count()
    if sample:
        result['sample'] = sample
    return result.reset_index()


# def compute_stat_in_cells(
#     data: pd.DataFrame,
#     sample: str,
#     group_by: list = ['feature_name', 'cell_id']
# ) -> pd.DataFrame:
#     percent_in_nucleus = data.groupby(group_by, observed = True)['overlaps_nucleus'].mean() * 100

#     result = pd.DataFrame({
#         'percent_in_nucleus': percent_in_nucleus,
#         'percent_in_cytoplasm': 100 - percent_in_nucleus
#     })
#     result['counts'] = data.groupby(group_by, observed = True)[group_by[0]].count()
#     result['sample'] = sample
#     return result.reset_index()