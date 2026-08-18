import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import anndata as ad
import matplotlib.pyplot as plt
import adjustText as at
from matplotlib.transforms import Bbox
import matplotlib.gridspec as gridspec
from statannotations.Annotator import Annotator
from itertools import combinations

from .statistics import (
    ttest,
    anova,
    kruskal,
    glm_binomial,
    beta_regression
)


def proportion_test(
        df,
        method="auto",
        groupby="CellType",
        condition="stage"
):

    results = []

    for celltype, data in df.groupby(groupby):

        res = _run_test(
            data,
            method=method,
            condition=condition
        )

        res["CellType"] = celltype

        results.append(res)

    return pd.DataFrame(results)


def scis_prop(
    adata: ad.AnnData,
    sample_col: str = "sample",
    condition_col: str = 'condition',
    group_col: str = "cell_type",
    group_only: str | list = None,
    group_top: int = None,
    strip_hue_col: str = None,
    palette_box: dict = None,
    palette_strip: dict = None,
    condition_order: list = None, # ["CTRL", "PAH"],  # might be possible to provide more conditions
    stat_test: str = "t-test_ind", #t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal, Brunner-Munzel
    ncols: int = 4,
    sub_figsize: tuple = (5,5),
    hspace: float = 0.5, 
    wspace: float = 0.5,
    save: str | None = None,
    rotate:int = 0,
    bbox_to_anchor=(0.98, 0.5),
):
    """Compute per zone celltype proportion between 2 conditions using replicate for statistical testing

    Parameters
    ----------
    adata
        AnnData object.
    group_by
        group
    group_only
        just plot this group
    split_by
        x value split_by
    split_only
        focus on this split_by
    split_by_top
        top split_by to consider
    replicate
        replicate key in adata.obs
    condition
        condition key in adata.obs
    condition_order
        tuple of the x conditions to test
    
    figsize
        figure size
    Returns
    -------

    """
    def make_pairs(data, condition_col, condition_order=None):
        levels = data[condition_col].dropna().unique().tolist()
        if condition_order is not None:
            levels = [x for x in condition_order if x in levels]
        if len(levels) < 2:
            return []
        return list(combinations(levels, 2))

    df = adata.obs[[sample_col, condition_col, group_col]]
    nb_cells = df.groupby([sample_col, condition_col, group_col]).size().unstack()
    nb_cells = nb_cells.div(nb_cells.sum(axis=1), axis=0).reset_index()
    nb_cells = nb_cells.melt(id_vars=[sample_col, condition_col])
    nb_cells = nb_cells.dropna()

    if isinstance(group_only, str):
        group_only = [group_only]

    if group_only:
        nb_cells = nb_cells[nb_cells[group_col].isin(group_only)]
    elif group_top:
        group_only = list(df[group_col].value_counts().head(group_top).index)
        nb_cells = nb_cells[nb_cells[group_col].isin(group_only)]
    else:
        group_only = list(df[group_col].unique())

    if not condition_order:
        condition_order = list(adata.obs[condition_col].unique())

    nrows = (len(group_only) + ncols - 1) // ncols

    fig = plt.figure(figsize=(sub_figsize[0]*ncols, sub_figsize[1]*nrows))
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    legend_handles = None
    legend_labels = None

    for i, group in enumerate(group_only, 1):
        ax = fig.add_subplot(nrows, ncols, i)
        data_sub = nb_cells[nb_cells[group_col] == group]

        sns.boxplot(
            data=data_sub,
            x=condition_col,
            y="value",
            order=condition_order,
            width=0.7,
            # gap=0.1,
            showfliers=False,
            linewidth=0.2,
            ax=ax,
            palette=palette_box
        )

        if strip_hue_col:
            strip_df = adata.obs[[sample_col, strip_hue_col]].drop_duplicates()
            st_dict = dict(zip(strip_df[sample_col], strip_df[strip_hue_col].astype(str)))
            data_sub = data_sub.copy()
            data_sub[strip_hue_col] = data_sub[sample_col].map(st_dict)
            sns.stripplot(
                data=data_sub,
                x=condition_col,
                y="value",
                hue=strip_hue_col,
                palette=palette_strip,
                dodge=False,
                jitter=0.12,
                size=6,
                linewidth=0.2,
                edgecolor="black",
                ax=ax,
                legend=(legend_handles is None)
            )
        else:
            sns.stripplot(
                data=data_sub,
                x=condition_col,
                y="value",
                palette=palette_box,
                dodge=False,
                jitter=0.12,
                size=6,
                linewidth=0.2,
                edgecolor="black",
                ax=ax,
                legend=(legend_handles is None)
            )
        if strip_hue_col and i==1:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
        pairs = make_pairs(data_sub, condition_col, condition_order)
        
        if pairs:
            annotator = Annotator(
                ax, pairs, data=data_sub,
                x=condition_col, y="value",
                order=condition_order
            )
            annotator.configure(
                test=stat_test,
                text_format="star",
                hide_non_significant=True,
                line_width=0.2,
                text_offset=0.15,
                pvalue_thresholds=[
                    [1e-4, "****"],
                    [1e-3, "***"],
                    [1e-2, "**"],
                    [0.05, "*"]]
            )
            annotator.apply_and_annotate()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(group)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate) #, size=6)

        if i%ncols == 1:
            ax.set_ylabel("Proportion")
        else:
            ax.set_ylabel("")
    
    if strip_hue_col and legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title=strip_hue_col,
            loc="center right",
            frameon=True,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=12,    
            title_fontsize=14  
        )
    if isinstance(save, str):
        plt.savefig(save, bbox_inches="tight")
    plt.show()


