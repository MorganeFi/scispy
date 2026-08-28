from ._constants import XeniumKeys, MerscopeKeys, CosmxKeys, GENE_EXCLUDE_PATTERN

def filter_xenium(
    df, 
    genes: list | None = None, 
    cells: list | None = None, 
    qv:int = 20, 
    only_in_cell: bool = True, 
    only_outside: bool = False,
    gene_exclude_pattern=GENE_EXCLUDE_PATTERN,
):
    mask = df[XeniumKeys.QV_KEY] >= qv
        
    if XeniumKeys.IS_GENE in df.columns:
        mask &= df[XeniumKeys.IS_GENE]

    if genes:
        mask &= df[XeniumKeys.FEATURE_KEY].isin(genes)

    if cells:
        mask &= df[XeniumKeys.CELL_ID].isin(cells)

    df = df[mask].dropna(subset=[XeniumKeys.FEATURE_KEY])

    if only_in_cell and only_outside:
        raise ValueError("Invalid combination: cannot be both 'only_in_cell' and 'only_outside'.")

    elif only_in_cell:
        df = df[df[XeniumKeys.CELL_ID] != XeniumKeys.UNASSIGNED_CELL_ID]

    elif only_outside:
        df = df[df[XeniumKeys.CELL_ID] == XeniumKeys.UNASSIGNED_CELL_ID]

    return df[~df[XeniumKeys.FEATURE_KEY].str.contains(
        gene_exclude_pattern, regex=True, na=True)]



def filter_merscope(
    df, 
    genes: list | None = None, 
    cells: list | None = None, 
    qv=None, 
    only_in_cell: bool = True,
    only_outside: bool = False, 
    gene_exclude_pattern=None,
):  
    df = df.dropna(subset=[MerscopeKeys.FEATURE_KEY])

    if genes:
        df = df[df[MerscopeKeys.FEATURE_KEY].isin(genes)]
    if cells:
        df = df[df[MerscopeKeys.CELL_ID].isin(cells)]
    
    if only_in_cell and only_outside:
        raise ValueError("Invalid combination: cannot be both 'only_in_cell' and 'only_outside'.")

    elif only_in_cell:
        df = df[(df[MerscopeKeys.CELL_ID] != MerscopeKeys.UNASSIGNED_CELL_ID) & (~df[MerscopeKeys.CELL_ID].isna())]

    elif only_outside:
        df = df[(df[MerscopeKeys.CELL_ID] == MerscopeKeys.UNASSIGNED_CELL_ID) | (df[MerscopeKeys.CELL_ID].isna())]

    return df[~df[MerscopeKeys.FEATURE_KEY].str.contains(
        gene_exclude_pattern, regex=True, na=True)]
    


def filter_cosmx(
    df, 
    genes: list | None = None, 
    cells: list | None = None, 
    qv=None, 
    only_in_cell: bool = True,
    only_outside: bool = False, 
    gene_exclude_pattern=None,
):  
    df = df.dropna(subset=[CosmxKeys.FEATURE_KEY])

    if genes:
        df = df[df[CosmxKeys.FEATURE_KEY].isin(genes)]
    if cells:
        df = df[df[CosmxKeys.CELL_ID].isin(cells)]

    if only_in_cell and only_outside:
        raise ValueError("Invalid combination: cannot be both 'only_in_cell' and 'only_outside'.")

    elif only_in_cell:
        df = df[df[CosmxKeys.CELL_ID] != CosmxKeys.UNASSIGNED_CELL_ID]

    elif only_outside:
        df = df[df[CosmxKeys.CELL_ID] == CosmxKeys.UNASSIGNED_CELL_ID]

    return df[~df[CosmxKeys.FEATURE_KEY].str.contains(
        gene_exclude_pattern, regex=True, na=True)]