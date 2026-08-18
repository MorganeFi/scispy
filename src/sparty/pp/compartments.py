import dask.dataframe as dd
import pandas as pd
import dask
from spatialdata import SpatialData

from .transcripts import subset_transcripts

from .._registry import TECHNO_REGISTRY
from .._constants import GENE_EXCLUDE_PATTERN
from .._validation import (
    _assert_technology_supported,
    _assert_dict_of_spatialdata,
    _assert_element_in_sdata,
)

def cells_dist_min(
    sdata,
    point,
    shapes_key = 'cell_boundaries',
    centroid: bool = False,
):
    if centroid:
        return sdata[shapes_key].geometry.centroid.distance(point).idxmin()
    else:
        return sdata[shapes_key].geometry.distance(point).idxmin()




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
    """
    Compute number the proportion of each gene in each compartment (nucleus, cytoplasm, outside cell).

    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    data = subset_transcripts(
        sdata=sdata,
        genes=genes,
        qv=qv,
        transcript_key=transcript_key,
        techno = techno, # 'Xenium' or 'Merscope'
        only_in_cell = True,
        only_outside = False,
        # feature_key=feature_key,
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

# def compute_stat_in_cells(
#     data: pd.DataFrame,
#     sample: str,
#     only_in_cells: bool = True,
#     group_by: list = ['feature_name', 'cell_id'],
#     unassigned: str = 'UNASSIGNED',
#     cell_id: str = 'cell_id',
# ) -> pd.DataFrame:
#     total = len(data)

#     percent_in_nucleus = data.groupby(group_by, observed = True)['overlaps_nucleus'].mean() * 100

#     if not only_in_cells:
#         percent_outside = (data[cell_id] == unassigned).sum() / total * 100
        
#         result = pd.DataFrame({
#             'percent_outside': percent_outside,
#             'percent_in_nucleus': percent_in_nucleus,
#             'percent_in_cytoplasm': 100 - percent_in_nucleus - percent_outside
#         })
#     else:
#         result = pd.DataFrame({
#             'percent_in_nucleus': percent_in_nucleus,
#             'percent_in_cytoplasm': 100 - percent_in_nucleus
#         })
        
#     result['counts'] = data.groupby(group_by, observed = True)[group_by[0]].count()
#     result['sample'] = sample
#     return result.reset_index()




def compute_unassigned_transcripts_stats(
    sdatas: dict[str, SpatialData],
    qv: float = 20,
    technology: str = "xenium",
    gene_exclude_pattern: str = GENE_EXCLUDE_PATTERN,
    transcripts_key: str = "transcripts",
    # feature_key: str = None,
    # cell_id_key: str = None,
    # unassigned_label: str = None,
) -> pd.DataFrame:
    """Compute the proportion of high-quality transcripts left unassigned to a cell.

    Filtering is delegated to the technology-specific `filter_fn` registered
    in `TECHNO_REGISTRY`, so QV thresholding (Xenium only) and gene exclusion
    patterns stay consistent with the rest of the filtering API.

    Parameters
    ----------
    sdata_dict
        Dictionary mapping sample names to their corresponding `SpatialData`.
    technology
        Key of `TECHNO_REGISTRY` (e.g. "xenium", "merscope", "cosmx").
    qv
        Minimum quality value. Ignored for technologies without a QV column.
    gene_exclude_pattern
        Regex excluding control/blank probes from `feature_key`.
    transcripts_key
        Key of the transcripts element in each `SpatialData`.

    Returns
    -------
    Tidy `DataFrame` with columns
    `['sample', 'pct_unassigned', 'total_unassigned', 'total_transcripts']`.
    """
    technology = _assert_technology_supported(technology=technology)
    _assert_dict_of_spatialdata(sdatas, "sdatas")

    keys = TECHNO_REGISTRY[technology]["keys"]
    filter_fn = TECHNO_REGISTRY[technology]["filter_fn"]

    records = []

    for sample, sdata in sdatas.items():
        _assert_element_in_sdata(sdata, transcripts_key)
        df_transcripts = sdata[transcripts_key]

        # pool of high-quality gene transcripts, regardless of cell assignment
        mol_hq = filter_fn(
            df_transcripts,
            qv=qv,
            only_in_cell=False,
            only_outside=False,
            gene_exclude_pattern=gene_exclude_pattern,
        )
        
        is_unassigned = keys.is_unassigned(mol_hq[keys.CELL_ID])
        # is_unassigned = mol_hq[keys.CELL_ID].eq(keys.UNASSIGNED_CELL_ID)

        if isinstance(mol_hq, dd.DataFrame):
            nb_unassigned, nb_transcripts = dask.compute(
                is_unassigned.sum(), mol_hq.shape[0] 
            )
            if technology == "merscope":
                nb_nan_cell_id = mol_hq[keys.CELL_ID].isna().sum().compute()
        else:
            nb_unassigned = int(is_unassigned.sum())
            nb_transcripts = len(mol_hq)

            if technology == "merscope":
                nb_nan_cell_id = mol_hq[keys.CELL_ID].isna().sum()         

        if nb_transcripts == 0:
            print(f"Sample '{sample}': no transcripts passed the filters.")
            continue

        if (technology == "merscope") and (nb_nan_cell_id > 0):
            print(f"Nan is cell_id ({nb_nan_cell_id})!! ")

        records.append({
            "sample": sample,
            "pct_unassigned": nb_unassigned / nb_transcripts,
            "total_unassigned": nb_unassigned,
            "total_transcripts": nb_transcripts,
        })

    return pd.DataFrame(records)


  
def compute_gene_compartment_percentages(
    ddf: dd.DataFrame,
    sample: str,
    gene_col: str = "feature_name",
    cell_col: str = "cell_id",
    nucleus_col: str = "overlaps_nucleus",
    unassigned_label: str = "UNASSIGNED",
    # condition = None,
) -> pd.DataFrame:  
    """
    Compute number the proportion of each gene in each compartment (nucleus, cytoplasm, outside cell).

    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    is_in_cell = ddf[cell_col] != unassigned_label
    is_outside = ddf[cell_col] == unassigned_label
    is_nucleus = (ddf[nucleus_col] == 1) & is_in_cell
    is_cytoplasm = (ddf[nucleus_col] == 0) & is_in_cell
   
    ddf = ddf.assign(
        nucleus_i = is_nucleus.astype(int),
        cytoplasm_i = is_cytoplasm.astype(int),
        outside_i = is_outside.astype(int)
    )
    
    agg = ddf.groupby(gene_col, observed=True).aggregate(
        total_counts = (gene_col, "size"),
        outside_counts = ("outside_i", "sum"),
        nucleus_counts = ("nucleus_i", "sum"),
        cytoplasm_counts = ("cytoplasm_i", "sum")
    )

    agg = agg.assign(
        pct_outside = agg["outside_counts"] / agg["total_counts"] * 100,
        pct_nucleus = agg["nucleus_counts"] / agg["total_counts"] * 100,
        pct_cytoplasm = agg["cytoplasm_counts"] / agg["total_counts"] * 100,
        sample = sample,
        # condition = condition
    )

    return agg.reset_index()


def compute_gene_compartment_percentages(
    sdatas: dict[str, SpatialData],
    qv: float = 20,
    technology: str = "xenium",
    gene_exclude_pattern: str = GENE_EXCLUDE_PATTERN,
    transcripts_key: str = "transcripts",
) -> pd.DataFrame:
    """Compute, per gene and per sample, the proportion of transcripts
    falling in the nucleus, cytoplasm, or outside any cell (unassigned).

    Filtering is delegated to the technology-specific `filter_fn` registered
    in `TECHNO_REGISTRY`, so QV thresholding (Xenium only) and gene exclusion
    patterns stay consistent with the rest of the filtering API.

    Parameters
    ----------
    sdatas
        Dictionary mapping sample names to their corresponding `SpatialData`.
    technology
        Key of `TECHNO_REGISTRY` (e.g. "xenium", "merscope", "cosmx").
    qv
        Minimum quality value. Ignored for technologies without a QV column.
    gene_exclude_pattern
        Regex excluding control/blank probes from the gene/feature column.
    transcripts_key
        Key of the transcripts element in each `SpatialData`.

    Returns
    -------
    Tidy `DataFrame` with columns
    `['sample', 'feature_name', 'total_counts', 'outside_counts',
    'nucleus_counts', 'cytoplasm_counts', 'pct_outside', 'pct_nucleus',
    'pct_cytoplasm']`.
    """
    technology = _assert_technology_supported(technology=technology)
    _assert_dict_of_spatialdata(sdatas, "sdatas")

    keys = TECHNO_REGISTRY[technology]["keys"]
    filter_fn = TECHNO_REGISTRY[technology]["filter_fn"]

    results = []

    for sample, sdata in sdatas.items():
        _assert_element_in_sdata(sdata, transcripts_key)
        df_transcripts = sdata[transcripts_key]

        # pool of high-quality gene transcripts, regardless of cell assignment
        mol_hq = filter_fn(
            df_transcripts,
            qv=qv,
            only_in_cell=False,
            only_outside=False,
            gene_exclude_pattern=gene_exclude_pattern,
        )

        is_outside = keys.is_unassigned(mol_hq[keys.CELL_ID])
        is_in_cell = ~is_outside
        is_nucleus = (mol_hq[keys.NUCLEUS_ID] == 1) & is_in_cell
        is_cytoplasm = (mol_hq[keys.NUCLEUS_ID] == 0) & is_in_cell

        mol_hq = mol_hq.assign(
            outside_i=is_outside.astype(int),
            nucleus_i=is_nucleus.astype(int),
            cytoplasm_i=is_cytoplasm.astype(int),
        )

        agg = mol_hq.groupby(keys.GENE, observed=True).aggregate(
            total_counts=(keys.GENE, "size"),
            outside_counts=("outside_i", "sum"),
            nucleus_counts=("nucleus_i", "sum"),
            cytoplasm_counts=("cytoplasm_i", "sum"),
        )

        if isinstance(agg, dd.DataFrame):
            agg = agg.compute()

        if agg.empty:
            print(f"Sample '{sample}': no transcripts passed the filters.")
            continue

        agg = agg.assign(
            pct_outside=agg["outside_counts"] / agg["total_counts"] * 100,
            pct_nucleus=agg["nucleus_counts"] / agg["total_counts"] * 100,
            pct_cytoplasm=agg["cytoplasm_counts"] / agg["total_counts"] * 100,
            sample=sample,
        )

        results.append(agg.reset_index())

    return pd.concat(results, ignore_index=True)


def compute_top_compartment_genes(
    sdatas: dict[str, SpatialData],
    qv: float = 20,
    technology: str = "xenium",
    gene_exclude_pattern: str = GENE_EXCLUDE_PATTERN,
    transcripts_key: str = "transcripts",
    min_counts: int = 100,
    n_top: int = 10,
) -> pd.DataFrame:
    """Identify, per sample, the genes most and least assigned to a cell.

    Builds on `compute_gene_compartment_percentages`: genes below
    `min_counts` total transcripts are discarded, then the `n_top` genes
    with the highest `pct_outside` ("unassigned") and the `n_top` genes with
    the lowest `pct_outside` ("assigned") are kept for each sample.

    Parameters
    ----------
    sdatas
        Dictionary mapping sample names to their corresponding `SpatialData`.
    technology
        Key of `TECHNO_REGISTRY` (e.g. "xenium", "merscope", "cosmx").
    qv
        Minimum quality value. Ignored for technologies without a QV column.
    gene_exclude_pattern
        Regex excluding control/blank probes from the gene/feature column.
    transcripts_key
        Key of the transcripts element in each `SpatialData`.
    min_counts
        Minimum total transcript count for a gene to be considered.
    n_top
        Number of genes to keep per sample and per direction.

    Returns
    -------
    Tidy `DataFrame`, same columns as `compute_gene_compartment_percentages`
    plus `rank_type` (`"assigned"` or `"unassigned"`).
    """
    res = compute_gene_compartment_percentages(
        sdatas=sdatas,
        qv=qv,
        technology=technology,
        gene_exclude_pattern=gene_exclude_pattern,
        transcripts_key=transcripts_key,
    )

    res = res[res["total_counts"] > min_counts]

    tops = []
    for sample, res_sample in res.groupby("sample", observed=True):
        top_unassigned = res_sample.sort_values(
            "pct_outside", ascending=False
        ).head(n_top).assign(rank_type="unassigned")

        top_assigned = res_sample.sort_values(
            "pct_outside", ascending=True
        ).head(n_top).assign(rank_type="assigned")

        tops.append(pd.concat([top_unassigned, top_assigned], ignore_index=True))

    return pd.concat(tops, ignore_index=True)