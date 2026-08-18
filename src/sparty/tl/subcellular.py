
import geopandas as gpd
import pandas as pd
 
from .._registry import TECHNO_REGISTRY
from .._validation import (
    _assert_spatialdata,
    _assert_table_in_sdata,
    _assert_technology_supported,
    _assert_gene_in_panel,
    _assert_key_in_obs,
    _assert_positive,
)
from ..pp.transcripts import subset_transcripts
 
 
def compute_transcript_nucleus_distance(
    sdata,
    genes: list,
    group_key: str,
    groups: str,  # single group value; list | None = None could be a future extension
    max_dist: float = 20.0,
    techno: str = "xenium",
    cell_key: str | None = None,
    feature_key: str | None = None,
    shape_key: str = "nucleus_boundaries",
    table_key: str = "table",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Compute the distance between transcripts and the nearest nucleus
    centroid, for a given set of genes and a given cell group.
 
    Parameters
    ----------
        to be completed
 
    Returns
    -------
        to be completed
    """
    _assert_spatialdata(sdata)
    techno = _assert_technology_supported(techno)
    adata = _assert_table_in_sdata(sdata, table_key)
 
    if len(genes) < 2:
        raise ValueError(f"`genes` must contain at least 2 genes, got {len(genes)}.")
    
    _assert_gene_in_panel(adata, genes)
    _assert_key_in_obs(adata, group_key)

    if groups not in adata.obs[group_key].unique():
        raise ValueError(
            f"`groups` value `{groups}` not found in `adata.obs['{group_key}']`. "
            f"Available values: {sorted(adata.obs[group_key].unique().tolist())}."
        )
    _assert_positive(max_dist, "max_dist")
 
    # Resolve technology-specific column names / unassigned sentinel value
    # instead of hardcoding "UNASSIGNED" (which is Xenium-only: Merscope
    # uses -1, CosMx uses 0).
    Keys = TECHNO_REGISTRY[techno]["keys"]
    cell_key = cell_key or Keys.CELL_ID
    feature_key = feature_key or Keys.FEATURE_KEY
    unassigned_id = Keys.UNASSIGNED_CELL_ID
 
    df_transcripts = subset_transcripts(
        sdata=sdata,
        genes=genes,
        techno=techno,
        only_in_cell=False,
        transform=False,
        return_gpd=True,
    )
 
    # STATS
    n_total = df_transcripts.groupby(feature_key).size()
    n_in_cell = (
        (df_transcripts[cell_key] != unassigned_id)
        .groupby(df_transcripts[feature_key])
        .sum()
    )
 
    stats = pd.DataFrame({"n_total": n_total, "n_in_cell": n_in_cell})
    stats["percentage"] = stats["n_in_cell"] / stats["n_total"] * 100
    stats = stats.reset_index()
 
    obs = adata.obs
    cell_grp_id = obs.loc[obs[group_key] == groups, cell_key].values.tolist()
 
    sub_df_ct = df_transcripts[
        (df_transcripts[cell_key].isin(cell_grp_id))
        | (df_transcripts[cell_key] == unassigned_id)
    ]
 
    nucl_df = sdata[shape_key][sdata[shape_key].index.isin(cell_grp_id)].copy()
    nucl_df["nucleus"] = nucl_df["geometry"]
    nucl_df["geometry"] = nucl_df.centroid
 
    result = gpd.sjoin_nearest(
        sub_df_ct,
        nucl_df,
        how="left",
        distance_col="distance_closest",
    )
    result = result[result["distance_closest"] <= max_dist]
 
    return result, stats

