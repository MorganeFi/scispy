import itertools
import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import edgepython as ep
import patsy

from tqdm import tqdm

from .._validation import _assert_key_in_obs, _resolve_groups


def _resolve_pairwise_comparisons(
    adata: ad.AnnData,
    condition: str,
    conds: list[str] | list[tuple[str, str]] | list[list[str]] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Resolve `conds` into the list of conditions and pairwise (test, ref) comparisons.

    `conds` can be a flat list (e.g. ``["ctrl", "mild", "severe"]``), in
    which case all pairwise combinations are built in input order (first
    element = test/up, second = ref/down), or a nested list of explicit
    pairs (e.g. ``[("severe", "mild"), ("mild", "ctrl")]``) to run only
    those specific, directed comparisons. If `conds` is None, every
    category of `condition` is used.

    Parameters
    ----------
    adata
        AnnData object.
    condition
        obs column holding the condition to compare (e.g. disease status).
    conds
        Flat list of conditions, or nested list of explicit (test, ref) pairs.
    
    Returns
    -------
    tuple[list[str], list[tuple[str, str]]]
        Resolved `(conds, pairwise)`, with `pairwise` entries as tuples.
    
    Raises
    ------
    KeyError
        If `condition` or a listed condition is missing from `adata.obs`.
    TypeError
        If `condition` is not categorical, or `conds` mixes flat and nested elements.
    ValueError
        If `conds` is empty, or a nested pair doesn't have exactly 2 elements.
    """
    _assert_key_in_obs(adata, condition, categorical=True)
 
    if conds is None:
        conds = list(adata.obs[condition].cat.categories)
        pairwise = list(itertools.combinations(conds, 2))
    else:
        conds = list(conds)
        if not conds:
            raise ValueError("`conds` cannot be an empty list.")
 
        is_pair = [isinstance(c, (list, tuple)) for c in conds]
        if all(is_pair):
            pairwise = []
            for pair in conds:
                if len(pair) != 2:
                    raise ValueError(
                        f"Each pairwise comparison must have exactly 2 "
                        f"elements (test, ref), got {pair!r}."
                    )
                pairwise.append(tuple(pair))
            conds = list(dict.fromkeys(itertools.chain(*pairwise)))  # unique, order-preserving
        elif not any(is_pair):
            pairwise = list(itertools.combinations(conds, 2))
        else:
            raise TypeError(
                "`conds` must be either a flat list of condition labels "
                "(e.g. ['ctrl', 'mild']) or a nested list of explicit "
                "(test, ref) pairs (e.g. [('mild', 'ctrl')]), not a mix "
                "of both."
            )
 
    available = set(adata.obs[condition].cat.categories)
    missing = [c for c in conds if c not in available]
    if missing:
        raise KeyError(
            f"Condition(s) not found in `adata.obs['{condition}']`: {missing}. "
            f"Available: {sorted(available)}."
        )
 
    return conds, pairwise
 
 
def _build_pdata(
    adconds: ad.AnnData,
    replicate: str,
    groups_key: str,
    condition: str,
    layer: str,
    min_cells: int,
    min_counts: int,
    join_by: str,
) -> ad.AnnData:
    """Build the per-(replicate, group, condition) pseudobulk AnnData.
 
    Sums counts per sample/group/condition, filters low-coverage samples,
    and assigns composite obs_names/labels used downstream.
    """
    print("Build pseudobulk...")
    pdata = dc.pp.pseudobulk(
        adata=adconds,
        sample_col=replicate,
        groups_col=[groups_key, condition],
        layer=layer,
        mode="sum",
    )
    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)
 
    pdata.obs_names = (
        pdata.obs[[replicate, groups_key, condition]].astype(str).agg(join_by.join, axis=1).values
    )
    pdata.obs[f"{groups_key}_{condition}"] = (
        pdata.obs[[groups_key, condition]].astype(str).agg(join_by.join, axis=1).values
    )
    return pdata


def _prepare_subset(
    pdata: ad.AnnData,
    ct: str,
    groups_key: str,
    condition: str,
    test: str,
    ref: str,
    replicate: str,
    paired: bool,
    continuous_vars: list[str] | None,
) -> ad.AnnData:
    """Subset pdata to one cell type and one pairwise comparison.

    Drops unused condition categories, casts continuous covariates to
    float, and restricts to paired donors if requested.
    """
    sub = pdata[
        (pdata.obs[groups_key] == ct) & (pdata.obs[condition].isin([ref, test]))
    ].copy()
    sub.obs[condition] = sub.obs[condition].cat.remove_unused_categories()
    # sub.obs[condition] = (sub.obs[condition].astype("category")
                #     .cat.remove_unused_categories()) 

    if continuous_vars is not None:
        for col in continuous_vars:
            sub.obs[col] = sub.obs[col].astype(float)

    if paired:
        nb_paired = sub.obs.groupby(replicate)[condition].nunique()
        paired_donors = nb_paired[nb_paired == 2].index
        sub = sub[sub.obs[replicate].isin(paired_donors)].copy()

    return sub


def _is_comparable(
    sub: ad.AnnData,
    condition: str,
    replicate: str,
    min_count_gene: int,
    min_total_count_gene: int,
) -> bool:
    """Check whether a subset has enough conditions/replicates/genes to run DESeq2.

    Applies dc.pp.filter_by_expr in place, then checks that both conditions
    have >=2 replicates, at least one gene survives filtering, and >=3
    replicates remain overall.
    """
    conds_present = sub.obs[condition].unique()
    rep_counts = sub.obs.groupby(condition)[replicate].nunique()

    if len(conds_present) != 2 or rep_counts.min() < 2:
        return False

    dc.pp.filter_by_expr(
        sub,
        group=condition,
        min_count=min_count_gene,
        min_total_count=min_total_count_gene,
    )
    # need >=1 gene surviving filtering and >=3 distinct replicates
    # (redundant if unpaired, but critical when paired: same donor counted in both conditions)
    return sub.n_vars > 0 and sub.obs[replicate].nunique() >= 3



def _run_deseq2(
    sub: ad.AnnData,
    design: str,
    condition: str,
    test: str,
    ref: str,
    shrink_LFC: bool,
    quiet: bool,
) -> pd.DataFrame:
    """Run DESeq2 on a prepared, filtered subset and return the raw results_df.

    Pure w.r.t. the caller's state: only reads `sub`, doesn't mutate
    anything outside it, doesn't touch adata.uns.
    """
    dds = DeseqDataSet(adata=sub, design=design, refit_cooks=True, quiet=quiet)
    # print(dds.obsm["design_matrix"])
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=[condition, test, ref], quiet=quiet)
    stat_res.summary()

    if shrink_LFC:
        stat_res.lfc_shrink(stat_res.contrast_vector.index[1])

    return stat_res.results_df


def _run_edger(
    sub: ad.AnnData,
    design: str,
    condition: str,
    test: str,
    ref: str,
    quiet: bool,
) -> pd.DataFrame:
    """Run edgeR (via edgePython, https://github.com/pachterlab/edgePython) on a
    prepared, filtered subset and return the raw results_df.

    Mirrors `_run_deseq2`'s contract (same inputs, gene-indexed results_df
    output) so the two engines are interchangeable in `pseudobulk()`. Pure
    w.r.t. the caller's state: only reads `sub`.

    Pipeline: build a DGEList -> TMM normalization -> dispersion estimation
    -> quasi-likelihood GLM fit -> QL F-test on the `test` vs `ref`
    coefficient -> extract all genes' results, re-indexed by gene and with
    columns renamed to match DESeq2's naming (`log2FoldChange`, `pvalue`,
    `padj`) so downstream code doesn't need to branch on which engine ran.

    Requires the optional `edgepython` and `patsy` dependencies
    (`pip install "edgepython[formula]"`).
    """
    try:
        import edgepython as ep
        import patsy
    except ImportError as e:
        raise ImportError(
            "`_run_edger` requires the optional `edgepython` and `patsy` "
            "dependencies. Install with `pip install \"edgepython[formula]\"`."
        ) from e

    # edgePython expects a genes x samples count matrix; `sub` is samples x genes.
    counts = np.asarray(sub.X.T)
    group = sub.obs[condition].astype(str).to_numpy()

    y = ep.make_dgelist(counts=counts, group=group)
    y = ep.calc_norm_factors(y)

    # Force `ref` as the reference level so the resulting coefficient reads
    # as test-vs-ref, matching DESeq2's `contrast=[condition, test, ref]`.
    design_formula = design.replace(
        condition, f'C({condition}, Treatment(reference="{ref}"))'
    )
    dmatrix = patsy.dmatrix(design_formula, sub.obs, return_type="dataframe")

    coef_name = f'C({condition}, Treatment(reference="{ref}"))[T.{test}]'
    if coef_name not in dmatrix.columns:
        raise ValueError(
            f"Could not find coefficient `{coef_name}` in the design matrix "
            f"columns: {dmatrix.columns.tolist()}. Check that `design` "
            f"includes `{condition}` exactly as written."
        )
    coef_idx = list(dmatrix.columns).index(coef_name)

    y = ep.estimate_disp(y, design=dmatrix.to_numpy())
    fit = ep.glm_ql_fit(y, dmatrix.to_numpy())
    res = ep.glm_ql_ftest(fit, coef=coef_idx)

    # sort_by="none" keeps the original gene order, needed to remap the
    # integer-positional index back to gene names below.
    top = ep.top_tags(res, n=sub.n_vars, sort_by="none")
    results_df = top["table"]
    results_df.index = sub.var_names[results_df.index]
    results_df = results_df.rename(
        columns={"logFC": "log2FoldChange", "PValue": "pvalue", "FDR": "padj"}
    )

    return results_df


def _add_pseudobulk_stats(
    results_df: pd.DataFrame,
    sub: ad.AnnData,
    adconds: ad.AnnData,
    ct: str,
    groups_key: str,
    condition: str,
    test: str,
    ref: str,
    replicate: str,
    layer: str,
    join_by: str,
    digits: int,
) -> pd.DataFrame:
    """Enrich a DESeq2 results_df with cell-type/condition labels and pct/sum stats."""
    nb_cells_1 = int(sub.obs.loc[sub.obs[condition] == test, "psbulk_cells"].sum())
    nb_cells_2 = int(sub.obs.loc[sub.obs[condition] == ref, "psbulk_cells"].sum())

    results_df["cell_type"] = ct
    results_df["condition"] = test + join_by + ref
    results_df["cond_1"] = test
    results_df["cond_2"] = ref
    results_df["nbCellsTotal_1"] = nb_cells_1
    results_df["nbCellsTotal_2"] = nb_cells_2
    results_df["sum_1"] = sub[sub.obs[condition] == test].X.sum(axis=0)
    results_df["sum_2"] = sub[sub.obs[condition] == ref].X.sum(axis=0)

    replicates_kept = sub.obs[replicate].unique()
    mask_1 = (
        (adconds.obs[groups_key] == ct)
        & (adconds.obs[condition] == test)
        & (adconds.obs[replicate].isin(replicates_kept))
    )
    mask_2 = (
        (adconds.obs[groups_key] == ct)
        & (adconds.obs[condition] == ref)
        & (adconds.obs[replicate].isin(replicates_kept))
    )

    results_df["pct_1"] = np.round(
        (adconds[mask_1, results_df.index].layers[layer] > 0).sum(axis=0) / nb_cells_1,
        decimals=digits,
    ).T
    results_df["pct_2"] = np.round(
        (adconds[mask_2, results_df.index].layers[layer] > 0).sum(axis=0) / nb_cells_2,
        decimals=digits,
    ).T

    return results_df.reset_index(names="gene")


def pseudobulk(
    adata: ad.AnnData,
    replicate: str,
    condition: str,  # needs to be unique per replicate
    groups_key: str,
    conds: list[str] | list[tuple[str, str]] | list[list[str]] | None = None,
    groups: list[str] | None = None,
    key_added: str = "pseudobulk",
    force: bool = False,
    paired: bool = False,
    layer: str = "counts",
    min_cells: int = 5,
    min_counts: int = 100,
    min_count_gene: int = 10,
    min_total_count_gene: int = 15,
    design: str | None = None,
    continuous_vars: list[str] | None = None,
    digits: int = 3,
    shrink_LFC: bool = False,
    join_by: str = "..",
    quiet: bool = True,
) -> pd.DataFrame:
    """Decoupler / pydeseq2 pseudobulk handler.
 
    Orchestrates: build the pseudobulk AnnData, then for each pairwise
    comparison and each cell type, subset -> check comparability -> run
    DESeq2 -> enrich with stats -> collect.
 
    Parameters
    ----------
    adata
        AnnData object.
    replicate
        replicate key
    condition
        condition key
    conds
        Conditions to compare. Either a flat list (all pairwise
        combinations are built, in input order: first = test/up, second =
        ref/down), or a nested list of explicit (test, ref) pairs to run
        only those specific, directed comparisons. If None, uses every
        category of `condition` and all pairwise combinations.
    groups_key
        sdata.table.obs key, i.e. cell types
    groups
        specify the cell types to work with
    key_added
        Key under `adata.uns['sparty']` where `{'params', 'matrix', 'result'}`
        for this run are stored, self-contained and independent from other
        runs stored under different `key_added` values.
    force
        If `adata.uns['sparty'][key_added]` already exists, skip
        recomputation and return the cached result unless `force=True`.
    layer
        sdata.table count values layer
    min_cells
        minimum cell number to keep replicate
    min_counts
        minimum total count to keep replicate
    min_count_gene
        min_count for dc.pp.filter_by_expr
    min_total_count_gene
        min_total_count for dc.pp.filter_by_expr
    design
        Model design, default will be '~condition' (or '~replicate + condition' if paired)
    continuous_vars
        list of obs columns to cast to float before DESeq2 (e.g. for continuous covariates
        in the design formula)
    digits
        rounding for pct_1 / pct_2
    shrink_LFC
        whether to apply lfc_shrink on the contrast
    join_by
        separator used to build composite obs_names / condition labels
    quiet
        passed to DeseqDataSet / DeseqStats
 
    Returns
    -------
    pd.DataFrame
        Concatenated pseudobulk DE results across all cell types and pairwise comparisons.
        Also stored in `adata.uns['sparty'][key_added]['result']`, alongside the
        `params` and `matrix` used to compute it.
    """
    if "sparty" in adata.uns and key_added in adata.uns["sparty"] and not force:
        print(
            f"`adata.uns['sparty']['{key_added}']` already exists, skipping "
            f"computation. Use `force=True` to recompute and overwrite."
        )
        return adata.uns["sparty"][key_added]["result"]
 
    groups = _resolve_groups(adata, groups_key, groups)
    conds, pairwise = _resolve_pairwise_comparisons(adata, condition, conds)
 
    adconds = adata[
        (adata.obs[condition].isin(conds)) & (adata.obs[groups_key].isin(groups))
    ].copy()
 
    pdata = _build_pdata(
        adconds, replicate, groups_key, condition, layer, min_cells, min_counts, join_by
    )
    matrix = pd.DataFrame(pdata.X.T, index=pdata.var_names, columns=pdata.obs_names)
 
    if not design:
        design = f"~{replicate} + {condition}" if paired else f"~{condition}"
 
    results_list: list[pd.DataFrame] = []
 
    for test, ref in pairwise:
        print(
            f"Start pseudobulk by comparing {test} versus {ref} in the condition "
            f"{condition}, with the design {design}."
        )
        for ct in tqdm(groups, total=len(groups), desc=groups_key):
            sub = _prepare_subset(
                pdata, ct, groups_key, condition, test, ref, replicate, paired, continuous_vars
            )
 
            if not _is_comparable(sub, condition, replicate, min_count_gene, min_total_count_gene):
                continue
 
            results_df = _run_deseq2(sub, design, condition, test, ref, shrink_LFC, quiet)
            results_df = _add_pseudobulk_stats(
                results_df, sub, adconds, ct, groups_key, condition, test, ref,
                replicate, layer, join_by, digits,
            )
            results_list.append(results_df)
 
    df_total = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
 
    adata.uns.setdefault("sparty", {})
    adata.uns["sparty"][key_added] = {
        "params": {
            "replicate": replicate,
            "groups_col": [groups_key, condition],
            "design": design,
            "layer": layer,
            "paired": paired,
            "conds": conds,
            "pairwise": pairwise,
            "min_cells": min_cells,
            "min_counts": min_counts,
            "min_count_gene": min_count_gene,
            "min_total_count_gene": min_total_count_gene,
            "continuous_vars": continuous_vars,
            "shrink_LFC": shrink_LFC,
        },
        "matrix": matrix,
        "result": df_total,
    }
    return df_total


