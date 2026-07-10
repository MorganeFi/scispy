import dask.dataframe as dd

def compute_gene_compartment_percentages(
    ddf: dd.DataFrame,
    sample: str,
    gene_col: str = "feature_name",
    cell_col: str = "cell_id",
    nucleus_col: str = "overlaps_nucleus",
    unassigned_label: str = "UNASSIGNED",
    # condition = None,
):
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