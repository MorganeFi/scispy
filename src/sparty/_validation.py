# from __future__ import annotations
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData

from ._constants import SUPPORTED_TECHNOLOGIES

# ---------------------------------------------------------------------------
# Generic type validation (used internally by the domain-specific
# assertions below, rarely called directly from Sparty's public API)
# ---------------------------------------------------------------------------

def check_type(
    value: Any,
    expected_types: type | tuple[type, ...],
    param_name: str,
    *,
    allow_none: bool = False,
) -> None:
    """Check that `value` is of the expected type(s).

    Parameters
    ----------
    value
        Value to check.
    expected_types
        Type or tuple of allowed types.
    param_name
        Parameter name, used in the error message.
    allow_none
        If True, `None` is accepted even if absent from `expected_types`.

    Raises
    ------
    TypeError
        If `value` is not of the expected type.
    """
    if value is None:
        if allow_none:
            return
        raise TypeError(f"`{param_name}` cannot be None.")

    if not isinstance(value, expected_types):
        if isinstance(expected_types, type):
            expected_str = expected_types.__name__
        else:
            expected_str = " or ".join(t.__name__ for t in expected_types)
        raise TypeError(
            f"`{param_name}` must be of type {expected_str}, "
            f"got {type(value).__name__}."
        )


# ---------------------------------------------------------------------------
# SpatialData / AnnData structure
# ---------------------------------------------------------------------------

def _assert_dict_of_spatialdata(sdatas: dict, param_name: str = "sdatas") -> None:
    check_type(sdatas, dict, param_name)
    for k, v in sdatas.items():
        check_type(k, str, f"{param_name} key")
        _assert_spatialdata(v, f"{param_name}['{k}']")

def _assert_spatialdata(sdata: Any, param_name: str = "sdata") -> None:
    """Check that `sdata` is indeed a SpatialData object."""
    check_type(sdata, SpatialData, param_name)


def _assert_table_in_sdata(sdata: SpatialData, table_key: str) -> AnnData:
    """Check that a table exists in `sdata.tables` and return it.

    Raises
    ------
    KeyError
        If `table_key` does not exist in `sdata.tables`.
    """
    if table_key not in sdata.tables:
        available = list(sdata.tables.keys())
        raise KeyError(
            f"Table `{table_key}` not found in `sdata.tables`. "
            f"Available tables: {available}."
        )
    return sdata.tables[table_key]


# def _assert_element_in_sdata(sdata: SpatialData, element_key: str) -> None:
#     """Check that a spatial element (points, shapes, images...) exists."""
#     if element_key not in sdata:
#         available = list(sdata.keys())
#         raise KeyError(
#             f"Element `{element_key}` not found in `sdata`. "
#             f"Available elements: {available}."
#         )
def _assert_element_in_sdata(container, element_key: str) -> None:
    """Check that `element_key` exists in a SpatialData element container
    (e.g. `sdata.points`, `sdata.shapes`, `sdata.images`, `sdata.labels`).
    """
    if element_key not in container:
        raise KeyError(
            f"Element `{element_key}` not found. "
            f"Available elements: {list(container.keys())}."
        )
 

# ---------------------------------------------------------------------------
# Technologies (Xenium / Merscope / CosMx)
# ---------------------------------------------------------------------------

def _assert_technology_supported(
    technology: str,
    allowed: Sequence[str] = SUPPORTED_TECHNOLOGIES,
) -> str:
    """Check that the requested ST technology is supported, and return
    its normalized (stripped, lowercase) form.
    """
    check_type(technology, str, "technology")
    technology = technology.strip().lower()

    if technology not in allowed:
        raise ValueError(
            f"Technology `{technology}` is not supported. "
            f"Choose one of: {list(allowed)}."
        )
    return technology


def _assert_technology_matches_sdata(
    sdata: SpatialData,
    technology: str,
    table_key: str = "table",
) -> str:
    technology = _assert_technology_supported(technology)
    adata = _assert_table_in_sdata(sdata, table_key)

    stored = adata.uns.get("technology")
    if stored is None:
        raise KeyError(
            f"No `technology` field found in `sdata.tables['{table_key}'].uns`. "
            "Was this dataset loaded through a Sparty reader?"
        )
    if stored.strip().lower() != technology:
        raise ValueError(
            f"Requested technology `{technology}` does not match the "
            f"technology stored in the dataset (`{stored}`)."
        )
    return technology


# ---------------------------------------------------------------------------
# Genes / panel
# ---------------------------------------------------------------------------


def _assert_gene_in_panel(
    adata: AnnData,
    gene: str | Iterable[str],
) -> None:
    """Check that a gene (or list of genes) is present in `adata.var_names`.

    Raises
    ------
    TypeError
        If `gene` is neither a string nor an iterable of strings.
    KeyError
        If one or more genes are missing from the panel.
    """
    if isinstance(gene, str):
        genes = [gene]
    elif isinstance(gene, Iterable):
        genes = list(gene)
        for g in genes:
            check_type(g, str, "gene")
    else:
        raise TypeError(
            f"`gene` must be a str or an iterable of str, got {type(gene).__name__}."
        )

    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(
            f"Gene(s) not found in panel: {missing}. "
            f"Panel contains {adata.n_vars} genes."
        )


# ---------------------------------------------------------------------------
# obs / AnnData columns (sample_key, condition_key, cluster/cell type key...)
# ---------------------------------------------------------------------------


def _assert_key_in_obs(
    adata: AnnData,
    key: str,
    *,
    categorical: bool = False,
) -> None:
    """Check that a column exists in `adata.obs`, optionally categorical.

    Useful for validating `sample_key`, `condition_key`, `cluster_key`,
    etc. following Sparty's scverse conventions.

    Raises
    ------
    KeyError
        If `key` is not in `adata.obs`.
    TypeError
        If `categorical=True` and the column is not of categorical dtype.
    """
    check_type(key, str, "key")
    if key not in adata.obs.columns:
        raise KeyError(
            f"`{key}` not found in `adata.obs`. "
            f"Available columns: {list(adata.obs.columns)}."
        )
    if categorical and not isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
        raise TypeError(
            f"`adata.obs['{key}']` must be categorical, "
            f"got dtype `{adata.obs[key].dtype}`. "
            f"Consider `adata.obs['{key}'] = adata.obs['{key}'].astype('category')`."
        )


def _assert_key_in_obsm(adata: AnnData, key: str) -> None:
    """Check that a key exists in `adata.obsm` (e.g. spatial coordinates)."""
    check_type(key, str, "key")
    if key not in adata.obsm:
        raise KeyError(
            f"`{key}` not found in `adata.obsm`. "
            f"Available keys: {list(adata.obsm.keys())}."
        )


def _assert_key_in_uns(adata: AnnData, key: str) -> None:
    """Check that a key exists in `adata.uns`."""
    check_type(key, str, "key")
    if key not in adata.uns:
        raise KeyError(
            f"`{key}` not found in `adata.uns`. "
            f"Available keys: {list(adata.uns.keys())}."
        )


# ---------------------------------------------------------------------------
# Numeric values (bandwidth, radius, thresholds...)
# ---------------------------------------------------------------------------

def _assert_str_or_list_of_str(value, name: str) -> list[str]:
    """Normalize a `str | list[str] | None` parameter into a list[str].

    Returns an empty list if `value` is None.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    check_type(value, list, name)
    for v in value:
        check_type(v, str, f"{name} item")
    return value

def _assert_positive(value: float | int, name: str, *, strict: bool = True) -> None:
    """Check that a numeric value is positive (or non-negative if strict=False)."""
    check_type(value, (int, float, np.integer, np.floating), name)
    if strict and value <= 0:
        raise ValueError(f"`{name}` must be strictly positive, got {value}.")
    if not strict and value < 0:
        raise ValueError(f"`{name}` must be non-negative, got {value}.")


def _assert_in_range(
    value: float | int,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """Check that a numeric value lies within [minimum, maximum]."""
    check_type(value, (int, float, np.integer, np.floating), name)
    if minimum is not None and value < minimum:
        raise ValueError(f"`{name}` must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"`{name}` must be <= {maximum}, got {value}.")


def _assert_one_of(value: Any, name: str, allowed: Sequence[Any]) -> None:
    """Check that `value` belongs to a set of allowed values.

    Useful for parameters such as `method="kde"` vs `method="histogram"`.
    """
    if value not in allowed:
        raise ValueError(f"`{name}` must be one of {list(allowed)}, got `{value}`.")



# ---------------------------------------------------------------------------
# Groups / conditions resolution (e.g. for pseudobulk-style analyses)
# ---------------------------------------------------------------------------
 
 
def _resolve_groups(
    adata: AnnData,
    groups_key: str,
    groups: list[str] | None = None,
) -> list[str]:
    """Resolve the list of groups (e.g. cell types) to work with.
 
    Checks that `groups_key` is a valid categorical column in `adata.obs`,
    and defaults `groups` to all its categories if not provided.
 
    Parameters
    ----------
    adata
        AnnData object.
    groups_key
        obs column holding the groups (e.g. cell types).
    groups
        Subset of groups to keep. If None, all categories of `groups_key`
        are used.
 
    Returns
    -------
    list[str]
        Resolved list of groups.
 
    Raises
    ------
    KeyError
        If `groups_key` is missing from `adata.obs`, or if any value in
        `groups` is not a category of `groups_key`.
    TypeError
        If `groups_key` is not categorical.
    """
    _assert_key_in_obs(adata, groups_key, categorical=True)
 
    if groups is None:
        return adata.obs[groups_key].cat.categories.tolist()
 
    available = set(adata.obs[groups_key].cat.categories)
    missing = [g for g in groups if g not in available]
    if missing:
        raise KeyError(
            f"Group(s) not found in `adata.obs['{groups_key}']`: {missing}. "
            f"Available: {sorted(available)}."
        )
    return groups
 