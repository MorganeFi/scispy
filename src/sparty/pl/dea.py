import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
# import spatialdata as sd
import anndata as ad
import matplotlib.pyplot as plt
# from spatialdata import SpatialData
# import spatialdata_plot
import adjustText as at
from matplotlib.transforms import Bbox
import decoupler as dc 
import matplotlib.gridspec as gridspec
import PyComplexHeatmap as pch
from scipy.cluster.hierarchy import linkage
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import glasbey

from .._config import FilterThresholds
from .._constants import (
    BASE_MEAN_COLUMN,
    LOGFC_COLUMN,
    PADJ_COLUMN,
    # PCT_COLUMN,
    PACKAGE_KEY,
)



def gene_filter_mask(
    df,
    padj=FilterThresholds.padj,
    logFC=FilterThresholds.logFC,
    pct=FilterThresholds.pct,
    baseMean=FilterThresholds.baseMean,
):
    return (
        (df[PADJ_COLUMN] <= padj)
        & (df[BASE_MEAN_COLUMN] >= baseMean)
        & ((df["pct_1"] >= pct) | (df["pct_2"] >= pct))
        & (df[LOGFC_COLUMN].abs() >= logFC)
    )


def filter_genes_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    try:
        return df.query(query).copy()
    except Exception as error:
        raise ValueError(
            f"Invalid filtering query: {query!r}"
        ) from error
    
def filter_de_genes(
    df: pd.DataFrame,
    query: str | None = None,
    padj: float = FilterThresholds.padj,
    logFC: float = FilterThresholds.logFC,
    pct: float = FilterThresholds.pct,
    baseMean: float = FilterThresholds.baseMean,
) -> pd.DataFrame:
    if query is not None:
        return filter_genes_query(df, query)

    mask = gene_filter_mask(
        df,
        padj=padj,
        logFC=logFC,
        pct=pct,
        baseMean=baseMean,
    )
    return df.loc[mask].copy()

# sc.set_figure_params(vector_friendly=True, dpi=300, dpi_save=300) 
# plt.rcParams.update(
#     {'ps.fonttype':42,
#     'ps.fonttype': 42, 
#     'pdf.fonttype': 42, 
#     'font.size': 10, 
#     # 'font.family': 'Arial', 
#     'mathtext.fontset': 'cm', 
#     # 'mathtext.rm': 'Arial',
#     'lines.linewidth': .2, 
#     'xtick.top': False, 
#     'ytick.right': False}
# )


def get_de_results(data, key="results", matrix=None) -> pd.DataFrame | tuple:
    if isinstance(data, ad.AnnData):
        sparty = data.uns.get(PACKAGE_KEY)
    
        if sparty is None:
            raise KeyError("No 'sparty' entry found in adata.uns. Run tl.pseudobulk first.")

        if key not in sparty:
            raise KeyError(f"No differential expression results found for key '{key}'.")
        
        if matrix is not None:
            if matrix not in sparty:
                raise KeyError(
                    f"Pseudobulk matrix '{matrix}' not found in adata.uns['{PACKAGE_KEY}']."
                )
            return (
                sparty[key].copy(),
                sparty[matrix].copy(),
            )

        return sparty[key].copy()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    raise TypeError(
        "`data` must be either an AnnData object or a pandas DataFrame."
    )


def barplotDE(
    data: ad.AnnData,
    # y_key: str = 'log2FoldChange',
    group_by: str = 'cell_type',
    split_by: str = 'condition',
    palette: tuple | str = 'deep',
    # title: str = None,
    groups: list | None = None,
    padj: float = FilterThresholds.padj,
    logFC: float = FilterThresholds.logFC,
    pct: float = FilterThresholds.pct,
    baseMean: float = FilterThresholds.baseMean,
    alpha: float = 0.5,
    join_by: str = '..',
    xticks_rotation: int = 90, 
    figsize: tuple = (8,3),
    save: str | None = None,
    global_pdf: bool = True,
    dpi: int = 300,
) -> None:
    """
    Plot the barplot of the number of genes DE per celltype for each compared condition.
    
    Parameters
    ----------
    data
        anndata object
    padj
        Adjusted p-value threshold (default: 0.05).
    logFC
        Absolute log2 fold-change threshold (default: 0.5).
    pct
        Minimum fraction of expressing cells (default: 0.1).
    baseMean
        Minimum mean normalized count (default: 10).
    """
    res_de = get_de_results(data)

    if groups:
        print("Filtrer groups...")
        res_de = res_de[res_de[group_by].isin(groups)]

    mask = gene_filter_mask(
        res_de,
        padj=padj,
        logFC=logFC,
        pct=pct,
        baseMean=baseMean,
    )

    res_de["updown"] = "NS"
    res_de.loc[mask & (res_de[LOGFC_COLUMN] >= 0), "updown"] = "Up"
    res_de.loc[mask & (res_de[LOGFC_COLUMN] <= 0), "updown"] = "Down"
    genes_DE_signif = res_de[res_de["updown"] != "NS"].copy()

    # cell_types = genes_DE_signif[group_by].unique()
    # all_combinations = pd.MultiIndex.from_product([cell_types, ["Up", "Down"]], names=[group_by, "updown"])
    # df_m = genes_DE_signif.groupby([group_by, "updown"]).size().reindex(all_combinations, fill_value=0).reset_index(name="value") 
     
    # up_df = df_m[df_m["updown"] == "Up"].reset_index(drop=True)
    # down_df = df_m[df_m["updown"] == "Down"].reset_index(drop=True)
    
    if len(genes_DE_signif[split_by].unique()) > 1 : 
        print("More than one pairwise condition.")

    if isinstance(save, str) and global_pdf:
        pdf = PdfPages(save)

    for cond in genes_DE_signif[split_by].unique():
        print(cond)
        sub_cond = genes_DE_signif[genes_DE_signif[split_by] == cond]
            
        cell_types = sub_cond[group_by].unique()
        all_combinations = pd.MultiIndex.from_product([cell_types, ["Up", "Down"]], names=[group_by, "updown"])
        df_m = sub_cond.groupby([group_by, "updown"]).size().reindex(all_combinations, fill_value=0).reset_index(name="value") 
            
        up_df = df_m[df_m["updown"] == "Up"].reset_index(drop=True)
        down_df = df_m[df_m["updown"] == "Down"].reset_index(drop=True)
    
        fig = plt.figure(figsize=figsize)
        sns.barplot(data=up_df, x=group_by, y="value", 
                    hue=group_by, dodge="auto", palette=palette)
        sns.barplot(data=down_df, x=group_by, y=-down_df["value"], 
                    hue=group_by, dodge="auto", palette=palette, alpha=alpha)
            
        # Add labels
        for i, row in up_df.iterrows():
            if row["value"] != 0:
                plt.text(i, row["value"]  , str(row["value"]), ha='center', va='bottom', fontsize=10)
            
        for i, row in down_df.iterrows():
            if row["value"] != 0:
                plt.text(i, -row["value"] , str(row["value"]), ha='center', va='top', fontsize=10)
            
        # Customize plot
        plt.axhline(0, color="grey", linestyle="--")
        plt.xticks(rotation=xticks_rotation, fontsize=12) # ha='center',
        plt.xlabel("Cell types")
        plt.ylabel("Number of genes DE")
        cond1, cond2 = cond.split(join_by)
        # if not title:
        title = f'Number of genes DE up and down in {cond1} versus {cond2} for each cell type'
        plt.title(title, fontsize=12) #, fontweight='bold')

        if isinstance(save, str) and not global_pdf:
            plt.savefig(save, bbox_inches="tight", dpi=dpi)
        
        if isinstance(save, str) and global_pdf:
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        plt.show()

    if isinstance(save, str) and global_pdf:
        pdf.close()

        
    
def stripPlotDE(
    data: ad.AnnData | pd.DataFrame, 
    x_key: str = 'cell_type',
    y_key: str = LOGFC_COLUMN,
    split_by = "condition",
    palette: tuple | str = "deep",
    # key: str = 'results',
    groups: list | None = None,
    padj: float = FilterThresholds.padj,
    logFC: float = FilterThresholds.logFC,
    pct: float = FilterThresholds.pct,
    baseMean: float = FilterThresholds.baseMean,
    top: int = 5,
    # order: list | None = None,
    join_by: str = '..',
    xticks_rotation: int = 90, 
    title: str = None,
    figsize: tuple = (8,3),
    save: str | None = None,
    global_pdf: bool = True,
    dpi: int = 300
) -> None:
    """
    Plot the stripplot of the number of genes DE per celltype for each compared condition.
    
    Parameters
    ----------
    data
        anndata object or pandas object
    x_key
        x key
    y_key
        y key
    key
        key in adata.uns['sparty'] storing the results to plot
    padj
        p adjusted to be significant
    log2FoldChange
        log2FoldChange to be significant
    figsize
        figure size
    save
        wether or not to save the figure
    """
    df = get_de_results(data)

    df["significative"] = gene_filter_mask(
        df,
        padj=padj,
        logFC=logFC,
        pct=pct,
        baseMean=baseMean,
    )
    
    # sig_idx = filter_genes(
    #     df,
    #     padj=padj,
    #     logfc=logFC,
    #     base_mean=baseMean,
    #     pct=pct,
    # ).index

    # df["significative"] = df.index.isin(sig_idx)

    if isinstance(save, str) and global_pdf:
        pdf = PdfPages(save)

    for cond in df[split_by].unique():

        sub_df = df[df[split_by] == cond].copy()

        if groups is not None:
            sub_df = sub_df[sub_df[x_key].isin(groups)]
            order = list(groups)
        else:
            order = list(sub_df[x_key].unique())

        # enforce order
        sub_df[x_key] = pd.Categorical(sub_df[x_key], categories=order, ordered=True)
        tmp = sub_df[sub_df['significative']].reset_index(drop=True)

        if tmp.empty:
            warnings.warn(
                f"{len(tmp)} DE gene(s) detected for {cond}. Skipping plot."
            )
        else:
            # Position centrale de chaque catégorie
            x_positions = {cat: i for i, cat in enumerate(order)}

            # Jitter reproductible
            rng = np.random.default_rng(42)

            sub_df["_x"] = sub_df[x_key].map(x_positions).astype(float)
            sub_df["_x_jitter"] = (
                sub_df["_x"]
                + rng.uniform(-0.35, 0.35, size=len(sub_df))
            )

            ns_df = sub_df.loc[~sub_df["significative"]].copy()
            sig_df = sub_df.loc[sub_df["significative"]].copy()

            fig, ax = plt.subplots(figsize=figsize)

            # Not significant
            ax.scatter(
                ns_df["_x_jitter"],
                ns_df[y_key],
                color="gray",
                alpha=0.5,
                s=6,
                linewidths=0,
            )

            # Significant
            sns.scatterplot(
                data=sig_df,
                x="_x_jitter",
                y=y_key,
                hue=x_key,
                palette=palette,
                alpha=0.8,
                s=30,
                linewidth=1,
                ax=ax,
                legend=False,
            )

            # Restaurer les catégories sur l’axe x
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=xticks_rotation)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)

            sig_df["rank_up"] = sig_df.groupby(x_key)[y_key].rank(
                method="first",
                ascending=False,
            )

            sig_df["rank_down"] = sig_df.groupby(x_key)[y_key].rank(
                method="first",
                ascending=True,
            )

            to_label = sig_df.loc[
                (sig_df["rank_up"] <= top)
                | (sig_df["rank_down"] <= top)
            ].copy()

            texts = []

            for _, row in to_label.iterrows():
                texts.append(
                    ax.text(
                        row["_x_jitter"],
                        row[y_key],
                        row["gene"],
                        ha="center",
                        va="bottom",
                        fontsize="x-small",
                    )
                )

            at.adjust_text(
                texts,
                ax=ax,
                target_x=to_label["_x_jitter"].to_numpy(),
                target_y=to_label[y_key].to_numpy(),
                expand=(1.5, 1.5),
                arrowprops={
                    "arrowstyle": "-",
                    "color": "black",
                    "lw": 0.8,
                },
            )
            
            if title:
                ax.set_title(title)
            else:
                cond1, cond2 = cond.split(join_by)
                ax.set_title(f"{cond1} vs {cond2}")

            ax.tick_params(axis='x', rotation=xticks_rotation)
            
            if isinstance(save, str) and not global_pdf:
                plt.savefig(save, bbox_inches="tight", dpi=dpi)
            
            if isinstance(save, str) and global_pdf:
                pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
                plt.close(fig)
            plt.show()

    if isinstance(save, str) and global_pdf:
        pdf.close()

           


            
def maplot(
    data: ad.AnnData,
    genes: list | None = None,
    thr_stat: float =0.5,
    thr_sign: float= 0.05,
    top: int = 10,
    x: str = BASE_MEAN_COLUMN,
    y: str = LOGFC_COLUMN,
    color_pos: str = "#D62728",
    color_neg: str = "#1F77B4",
    color_null: str = "gray",
    figsize: tuple = (8,6),
    fig_title: str = "MA plot",
) -> None:
    """
    Plot the maplt of the DGE.
    
    Parameters
    ----------
    adata
        anndata object
    x_key
        x key
    y_key
        y key
    padj
        p adjusted to be significant
    log2FoldChange
        log2FoldChange to be significant
    figsize
        figure size
    save
        wether or not to save the figure
    """
    df = get_de_results(data)

    df["log2_mean"] = np.log2(df[x])

    up_msk = (df[y] >= thr_stat) & (df[PADJ_COLUMN] <= thr_sign) 
    dw_msk = (df[y] <= -thr_stat) & (df[PADJ_COLUMN] <= thr_sign)
    not_sign = ~(up_msk | dw_msk)
    
    if type(genes) == list:
        signs = df[df['gene'].isin(genes)]
    else:
        signs = df[up_msk | dw_msk].sort_values(PADJ_COLUMN, ascending=False)
        signs = signs.iloc[:top]

    plt.figure(figsize=figsize)
    sc_mid = plt.scatter(df.loc[not_sign, "log2_mean"], df.loc[not_sign, y], color=color_null, alpha=0.6, s=20, label=f"Non-significant ({not_sign.sum()})")
    sc_up = plt.scatter(df.loc[up_msk, "log2_mean"], df.loc[up_msk, y], color=color_pos, alpha=0.7, s=20, label=f"Up ({up_msk.sum()})")
    sc_down = plt.scatter(df.loc[dw_msk, "log2_mean"], df.loc[dw_msk, y], color=color_neg, alpha=0.7, s=20, label=f"Down ({dw_msk.sum()})")

    ymax =  df[y].abs().max() + 0.5
    xmax = df["log2_mean"].max() + 0.5

    plt.ylim(-ymax, ymax)
    plt.xlim(0, xmax)
    plt.axhline(y=thr_stat, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-thr_stat, color='black', linestyle='--', linewidth=1)

    texts = []
    for x, y, s in zip(signs["log2_mean"], signs[y], signs['gene'], strict=False):
        texts.append(
            plt.text(x, y, s, 
                    bbox=dict(boxstyle="round",
                            facecolor='white', edgecolor='black',             
                    ),
                    fontweight = 'bold',
                    size='small'))
    if len(texts) > 0:
        at.adjust_text(
            texts, expand=(4, 4),
            arrowprops={"arrowstyle": "->", "color": "black"})
    plt.legend(
        handles=[sc_up, sc_mid, sc_down],
        loc='best', 
        bbox_to_anchor=(0.81, 0., 0.5, 0.5))
    plt.grid(True)
    plt.title(fig_title)
    plt.xlabel("Log2 mean expression")
    plt.ylabel('Log2 fold change')
    plt.show()


def _plot_heatmap(
    df_plot: pd.DataFrame,
    df_colors: pd.DataFrame,
    colors: dict,
    show_rownames: bool = True,
    col_dendrogram: bool = True,
    row_dendrogram: bool = True,
    row_cluster: bool = True,
    col_cluster: bool = True,
    cmap: str = "bwr",
    linkage = None,
) -> pch.ClusterMapPlotter:
    """Creates and returns a PyComplexHeatmap ClusterMapPlotter."""
    valid_cols = [col for col in df_colors.columns if col in colors]
    
    if valid_cols:
        ann_kwargs = {
            col: pch.anno_simple(df_colors[col], colors=colors[col], add_text=False, legend=True)
            for col in valid_cols
        }
        col_ha = pch.HeatmapAnnotation(
            **ann_kwargs,
            legend=True,
            legend_gap=1, #5
            hgap=0.5,
            axis=1,
        )
    else:
        col_ha = None

    if linkage is not None:
        row_dendrogram_kws={'linkage': linkage}
    else:
        row_dendrogram_kws=None      

    return pch.ClusterMapPlotter(
        data=df_plot,
        top_annotation=col_ha,
        label='values',
        col_dendrogram=col_dendrogram,
        row_dendrogram=row_dendrogram,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        show_rownames=show_rownames,
        show_colnames=True,
        row_dendrogram_kws=row_dendrogram_kws,
        verbose=0,
        legend_hpad=2,
        legend_vpad=25, # 10??
        cmap=cmap,  
        plot =False,
        plot_legend=False,
        center=0, 
        xticklabels_kws={'labelrotation': 90},
        # col_split=adata.obs[condition],
        # col_split_gap=0.8,
    ) 


def _resolve_groups_col(groups_col) -> tuple[str | None, str, list]:
    """
    Returns (groups_key, condition, col_to_add) based on the size of groups_col.
    Raise a ValueError if groups_col has fewer than 2 elements.
    """
    match len(groups_col):
        case 1:
            condition  = groups_col[0]
            groups_key = None
            col_to_add = [None, condition]          # replicate will be inserted after
        case 2:
            groups_key, condition = groups_col
            col_to_add = [None, groups_key, condition]
        case _:
            raise ValueError("'groups_col' must contain 1 or 2 elements.")
    return groups_key, condition, col_to_add


def _resolve_params(adata, replicate, groups_col):
    """Merges the explicit parameters with those stored in adata.uns."""
    params     = adata.uns["sparty"]["params"]
    replicate  = replicate  or params["replicate"]
    groups_col = groups_col or params["groups_col"]

    if not isinstance(groups_col, (list, np.ndarray)):
        raise ValueError("'groups_col' must be a list or a ndarray.")

    groups_key, condition, col_to_add = _resolve_groups_col(groups_col) # list(groups_col)
    col_to_add[0] = replicate          # insert the resolved replicate
    return replicate, groups_key, condition, col_to_add


def _get_colors(
    adata: ad.AnnData,
    condition: str,
    replicate: str,
) -> dict:
    """
    Build a default colors dict for heatmap_DE when colors=None:
    - one glasbey palette for the condition categories
    - one glasbey block palette for replicates, grouped by condition
    """
    condition_categories = (
        adata.obs[condition].astype("category").cat.categories.tolist()
    )
    n_condition = len(condition_categories)
    pal_condition = glasbey.create_palette(n_condition, colorblind_safe=True)
    condition_colors = dict(zip(condition_categories, pal_condition))

    # number of distinct replicates per condition, preserving condition order,
    # so the block palette groups sample colors by condition
    df_rep_cond = adata.obs[[replicate, condition]].drop_duplicates()
    samples_per_condition = [
        df_rep_cond.loc[df_rep_cond[condition] == c, replicate].nunique()
        for c in condition_categories
    ]
    pal_sample = glasbey.create_block_palette(
        samples_per_condition, optimize_palette=True, colorblind_safe=True
    )
    replicate_order_list = df_rep_cond.sort_values(condition)[replicate].tolist()
    sample_colors = dict(zip(replicate_order_list, pal_sample))

    return {
        condition: condition_colors,
        replicate: sample_colors,
    }   


def _build_heatmap_data(
    adata_hm: ad.AnnData,
    replicate: str,
    condition: str,
    colors: dict,
    paired: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Constructs df_plot and df_colors from a subselected AnnData.
    Also returns the effective replicate key (original or paired).
    """
    if paired:
        new_rep = f"{replicate}_{condition}"
        adata_hm.obs[new_rep] = (
            adata_hm.obs[replicate].astype(str) + "_" +
            adata_hm.obs[condition].astype(str)
        )
    else:
        new_rep = replicate

    df_plot = pd.DataFrame(
        adata_hm.X.T,
        index=adata_hm.var_names,
        columns=adata_hm.obs[new_rep],
    )
    df_colors = pd.DataFrame(
        adata_hm.obs[list(colors.keys())].values,
        index=adata_hm.obs[new_rep].values,
        columns=list(colors.keys()),
    )
    return df_plot, df_colors, new_rep


def _plot_volcano(
    ax, 
    sub_all: pd.DataFrame, 
    genes_sig, 
    genes: list | None = None, 
    x: str = LOGFC_COLUMN, 
    y: str = PADJ_COLUMN,
    top_volcano: int = 10,
    thr_stat: float = FilterThresholds.logFC,
    thr_sign: float = FilterThresholds.padj,
) -> None:
    """Plots the volcano plot, with optional gene highlighting."""
    if genes is not None:
        dc.pl.volcano(sub_all, x=x, y=y, ax=ax,
                      top=1, thr_stat=thr_stat, thr_sign=thr_sign)

        highlight = sub_all.loc[genes_sig]

        texts = []
        for gene, row in highlight.iterrows():
            texts.append(
                ax.text(
                    row[x], -np.log10(row[y]), gene, 
                    ha="center", va="bottom", 
                    color="black", fontsize=10
                    # bbox=dict(boxstyle="round",
                    #         facecolor='white', edgecolor='black',             
                    # ), fontweight = 'bold',# size='small',
                    )
            )

        if len(texts) > 0:
            at.adjust_text(texts,ax=ax,
                           expand=(2,2),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                    )
                )
    else:
        dc.pl.volcano(
            sub_all, 
            x=x, y=y,
            ax=ax,
            top=top_volcano, 
            thr_stat=thr_stat, 
            thr_sign=thr_sign
        )
    ax.set_title("Volcano")


# heatmap_volcano(
def heatmap_DE(
    adata: ad.AnnData,
    cell_type: str = "cell_type",
    genes: list | None = None,
    replicate_order: list | None = None, # a ajouter pour controler l'ordre des échantillons dans la heatmap et pas de dendogrammme
    padj: float = FilterThresholds.padj,
    logFC: float = FilterThresholds.logFC,
    pct: float = FilterThresholds.pct,
    baseMean: float = FilterThresholds.baseMean,
    groups_col: list | None = None,
    replicate: str | None = None,
    groups: list | None = None,
    colors: dict | None = None,
    query: str | None = None,
    cmap: str = "bwr",
    top_volcano: int = 10,
    join_by: str = "..",
    paired: bool = False,
    max_value: int = 10,
    only_heatmap: bool = False,
    only_volcano: bool = False,
    nb_to_show: int = 50,
    sort_by: list | None = None,
    method: str = 'ward', # average
    metric: str = 'euclidean', #correlation
    figsize: tuple = (20, 10),
    save: str | None = None,
    global_pdf: bool = True,
    dpi: int = 300,
) -> None:
    """
    Plot a heatmap and/or volcano plot of differentially expressed genes
    for each cell type present in adata.uns["sparty"].
    """
    df, mtx = get_de_results(adata, matrix = 'matrix')
    # df = adata.uns['sparty']['results'].copy()
    # mtx = adata.uns["sparty"]["matrix"].copy()

    replicate, groups_key, condition, col_to_add = _resolve_params(
        adata, replicate, groups_col
    )

    if colors is None:
        colors = _get_colors(adata, condition, replicate)
    print(colors)
    df_sig = filter_de_genes(
        df,
        query=query,
        padj=padj,
        logFC=logFC,
        pct=pct,
        baseMean=baseMean,
    )

    if groups:
        df = df[df[cell_type].isin(groups)]
        df_sig = df_sig[df_sig[cell_type].isin(groups)]
    
    if isinstance(save, str) and global_pdf:
        pdf = PdfPages(save)

    for cell, sub_sig in df_sig.groupby(cell_type):
        if genes is not None:
            genes_sig = np.intersect1d(genes, df[df[cell_type] == cell]["gene"].unique())
        else:
            genes_sig = sub_sig["gene"].unique()

        if len(genes_sig) < 2:
            continue
    
        print(f"{cell} -> {len(genes_sig)} genes DE")

        column_ct = mtx.columns.str.contains(f'{join_by}{cell}{join_by}', regex=False)
        adata_tmp = ad.AnnData(mtx.loc[:, column_ct].T)
        adata_tmp.obs[col_to_add] =  adata_tmp.obs.index.str.split(join_by, regex=False).tolist() 
  
        for col in colors:
            if (col not in adata_tmp.obs.columns) and (col in adata.obs.columns):
                mapping = adata.obs[[replicate, col]].drop_duplicates().set_index(replicate)[col]
                adata_tmp.obs[col] = adata_tmp.obs[replicate].map(mapping)

        sc.pp.normalize_total(adata_tmp)
        sc.pp.log1p(adata_tmp)
        sc.pp.scale(adata_tmp, max_value=max_value)

        adata_hm = adata_tmp[:, genes_sig]
        df_plot, df_colors, _ = _build_heatmap_data(
            adata_hm, replicate, condition, colors, paired
        )

        fig = plt.figure(figsize=figsize)
        show_both = not only_heatmap and not only_volcano
        # gs = gridspec.GridSpec(1, 1 + show_both, wspace=0.5) #, width_ratios=[1, 1])
        if show_both:
            gs = gridspec.GridSpec(1, 2, wspace=0.6, width_ratios=[1.5, 1])
        else:
            gs = gridspec.GridSpec(1, 1)

        i = 0

        # ===== HEATMAP =====
        if not only_volcano:
            if replicate_order is not None:
                df_colors = df_colors[df_colors[replicate].isin(replicate_order)]
                df_colors[replicate] = pd.Categorical(
                    df_colors[replicate],
                    categories=replicate_order,
                    ordered=True
                )
                if sort_by:
                    df_colors = df_colors.sort_values(sort_by)
                else:
                    # df_colors = df_colors.sort_values(replicate)
                    df_colors = df_colors.sort_values([condition, replicate])

                df_plot = df_plot[df_colors.index]
                col_cluster = col_dendrogram = False 
                
                # ordered = [c for c in replicate_order if c in df_plot.columns]
                # df_plot = df_plot[ordered]
                # df_colors = df_colors.loc[ordered]
            else:
                col_cluster = col_dendrogram = True

            if (method != 'ward') & (metric != 'euclidean'):
                Z = linkage(df_plot.values, method=method, metric=metric)    
            else:
                Z = None
            ax = fig.add_subplot(gs[i])
            cluster = _plot_heatmap(
                df_plot, df_colors, colors,
                show_rownames=(len(genes_sig) <= nb_to_show),
                cmap=cmap,
                linkage=Z,
                col_cluster=col_cluster,
                col_dendrogram=col_dendrogram,
            )
            cluster.plot(ax=ax, subplot_spec=gs[i])
            cluster.plot_legends(ax=ax)
            ax.set_title("Heatmap")
            i += 1

        # ===== VOLCANO =====
        if not only_heatmap:
            ax = fig.add_subplot(gs[i])
            sub_all = df[df[cell_type] == cell].set_index("gene")
            _plot_volcano(
                ax=ax, 
                sub_all=sub_all, 
                genes_sig=genes_sig,
                genes=genes, 
                top_volcano=top_volcano, 
                thr_stat=logFC,
                thr_sign=padj
            )

        fig.suptitle(cell, fontsize=18)
        if isinstance(save, str) and not global_pdf:
            plt.savefig(save, bbox_inches="tight", dpi=dpi)
        
        if isinstance(save, str) and global_pdf:
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
        plt.show()

    if isinstance(save, str) and global_pdf:
        pdf.close()




# def extract_sample(name, samples):
#     for s in samples:
#         if name.startswith(s):
#             return s
#     return None  # si rien ne matche

# def extract_sample(name: str, samples: list[str]) -> str | None:
#     """Returns the first prefix of `samples` that matches `name`."""
#     return next((s for s in samples if name.startswith(s)), None)
    

# def filter_genes(df: pd.DataFrame, thres: dict) -> pd.DataFrame:
#     """Filters genes based on the padj, log2FoldChange, baseMean, and pct thresholds."""
#     masks = [
#         df["padj"] <= thres["padj"],
#         df["log2FoldChange"].abs() >= thres["log2FoldChange"],
#         df["baseMean"] >= thres["baseMean"],
#         (df["pct_1"] >= thres["pct"]) | (df["pct_2"] >= thres["pct"]),
#     ]
#     return df[np.logical_and.reduce(masks)]


# def filter_genes(
#     df: pd.DataFrame, 
#     padj: float = FilterThresholds.padj,
#     logfc: float = FilterThresholds.logfc,
#     pct: float = FilterThresholds.pct,
#     base_mean: float = FilterThresholds.base_mean,
# ) -> pd.DataFrame:
    
#     padj_mask = df[PADJ_COLUMN] <= padj
#     lfc_mask = np.abs(df[LOGFC_COLUMN]) >= logfc
#     base_mask = df[BASE_MEAN_COLUMN] >= base_mean
#     pct_mask = (df["pct_1"] >= pct) | (df["pct_2"] >= pct)

#     mask = padj_mask & lfc_mask & base_mask & pct_mask
#     return df[mask].copy()
    # return df.loc[mask, "gene"].unique() ==> RETURN ONLY gene NAME !!!

# def filter_genes(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
#     padj_mask = df[PADJ_COLUMN] <= thres["padj"]
#     lfc_mask = np.abs(df[LOGFC_COLUMN]) >= thres["log2FoldChange"]
#     base_mask = df[BASE_MEAN_COLUMN] >= thres["baseMean"]
#     pct_mask = (df["pct_1"] >= thres["pct"]) | (df["pct_2"] >= thres["pct"])

#     mask = padj_mask & lfc_mask & base_mask & pct_mask
#     return df[mask]
    # return df.loc[mask, "gene"].unique() ==> RETURN ONLY gene NAME !!!




# def stripPlotDE(
#     data: ad.AnnData | pd.DataFrame, 
#     x_key: str = 'cell_type',
#     y_key: str = LOGFC_COLUMN,
#     split_by = "condition",
#     palette: tuple | str = "deep",
#     # key: str = 'results',
#     groups: list | None = None,
#     padj: float = FilterThresholds.padj,
#     logFC: float = FilterThresholds.logFC,
#     pct: float = FilterThresholds.pct,
#     baseMean: float = FilterThresholds.baseMean,
#     top: int = 5,
#     # order: list | None = None,
#     join_by: str = '..',
#     xticks_rotation: int = 90, 
#     title: str = None,
#     figsize: tuple = (8,3),
#     save: str | None = None,
#     dpi: int = 300
# ) -> None:
#     """
#     Plot the stripplot of the number of genes DE per celltype for each compared condition.
    
#     Parameters
#     ----------
#     data
#         anndata object or pandas object
#     x_key
#         x key
#     y_key
#         y key
#     key
#         key in adata.uns['sparty'] storing the results to plot
#     padj
#         p adjusted to be significant
#     log2FoldChange
#         log2FoldChange to be significant
#     figsize
#         figure size
#     save
#         wether or not to save the figure
#     """
#     df = get_de_results(data)

#     df["significative"] = gene_filter_mask(
#         df,
#         padj=padj,
#         logFC=logFC,
#         pct=pct,
#         baseMean=baseMean,
#     )
    
#     # sig_idx = filter_genes(
#     #     df,
#     #     padj=padj,
#     #     logfc=logFC,
#     #     base_mean=baseMean,
#     #     pct=pct,
#     # ).index

#     # df["significative"] = df.index.isin(sig_idx)


#     for cond in df[split_by].unique():

#         sub_df = df[df[split_by] == cond].copy()

#         if groups is not None:
#             sub_df = sub_df[sub_df[x_key].isin(groups)]
#             order = list(groups)
#         else:
#             order = list(sub_df[x_key].unique())

#         # enforce order
#         sub_df[x_key] = pd.Categorical(sub_df[x_key], categories=order, ordered=True)
#         tmp = sub_df[sub_df['significative']].reset_index(drop=True)

#         if tmp.empty:
#             warnings.warn(
#                 f"{len(tmp)} DE gene(s) detected for {cond}. Skipping plot."
#             )
#         else:
#             _, ax = plt.subplots(figsize=figsize)

#             sns.stripplot(
#                 data=sub_df[~sub_df["significative"]],
#                 x=x_key, y=y_key,
#                 order=order,
#                 color="gray",
#                 # orient='v',
#                 alpha=0.5,
#                 size=2,
#                 jitter=0.4,
#                 linewidth=0, 
#                 ax=ax
#             )

#             sns.stripplot(
#                 data=sub_df[sub_df["significative"]],
#                 x=x_key, y=y_key,
#                 order=order,
#                 # orient='v',
#                 hue=x_key,
#                 palette=palette,
#                 alpha=0.8,
#                 size=5,
#                 jitter=0.4,
#                 linewidth=1, 
#                 ax=ax
#             )
#             tmp["rank"] = tmp.groupby(x_key)[LOGFC_COLUMN].rank(method="first", ascending=False)
#             tmp["top_bottom"] = tmp.groupby(x_key).apply(
#                 lambda g: pd.Series(
#                     np.where(
#                         g["rank"] <= min(top, len(g)), "Top",
#                         np.where(g["rank"] > len(g) - min(top, len(g)), "Bottom", None)
#                     ), index=g.index
#                 )
#             ).reset_index(level=0, drop=True)

#             texts = []
#             # Build a mapping from category label → integer x-position
#             x_positions = {cat: i for i, cat in enumerate(order)}

#             to_label = tmp[tmp["top_bottom"].notna()].copy()

#             for _, row in to_label.iterrows():
#                 x_pos = x_positions.get(row[x_key])
#                 if x_pos is None:
#                     continue
#                 # Use a tiny deterministic offset so labels don't all stack at x_pos
#                 # adjust_text will handle the fine-grained repositioning
#                 texts.append(ax.text(
#                     x_pos, row[y_key], row["gene"],
#                     ha="center", va="bottom", color="black", size="x-small"
#                 ))
#             # for collection, (_, group_data) in zip(
#             #     ax.collections[-len(tmp[x_key].unique()):], tmp.groupby(x_key)):
#             #     for i, (x, y) in enumerate(collection.get_offsets()):
#             #         if group_data.iloc[i]['top_bottom'] is not None:
#             #             texts.append(ax.text(
#             #                 x, y, group_data.iloc[i]['gene'],
#             #                 ha='center', va='bottom', color='black', size='x-small'
#             #             ))

#             at.adjust_text(
#                 texts,
#                 ax=ax,
#                 expand=(2,2),
#                 arrowprops=dict(
#                     arrowstyle="->",
#                     color="black",
#                 )
#             )
            
#             if title:
#                 ax.set_title(title)
#             else:
#                 cond1, cond2 = cond.split(join_by)
#                 ax.set_title(f"{cond1} vs {cond2}")

#             ax.tick_params(axis='x', rotation=xticks_rotation)

#             if save:
#                 plt.savefig(save, dpi=dpi, bbox_inches="tight")
#             plt.show()


