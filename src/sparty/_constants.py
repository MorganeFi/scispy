from dataclasses import dataclass


@dataclass(frozen=True)
class XeniumKeys:
    FEATURE_KEY = "feature_name"
    TRANSCRIPT_KEY = "transcripts"
    CELL_ID = "cell_id"
    NUCLEUS_ID = "overlaps_nucleus"
    UNASSIGNED_CELL_ID = "UNASSIGNED"
    QV_KEY = "qv"
    IS_GENE = "is_gene"
    # GENE_EXCLUDE_PATTERN = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*"

    @staticmethod
    def is_unassigned(cell_id_series):
        return cell_id_series.eq(XeniumKeys.UNASSIGNED_CELL_ID)
    
@dataclass(frozen=True)
class MerscopeKeys:
    FEATURE_KEY = "gene"
    TRANSCRIPT_KEY = "transcripts" # TO MODIFIED
    CELL_ID = "cell_id"
    NUCLEUS_ID = "overlaps_nucleus"

    UNASSIGNED_CELL_ID = -1
    # GENE_EXCLUDE_PATTERN = "Blank-*"

    @staticmethod
    def is_unassigned(cell_id_series):
        return cell_id_series.eq(MerscopeKeys.UNASSIGNED_CELL_ID) 
    
@dataclass(frozen=True) 
class CosmxKeys:
    FEATURE_KEY = "target"
    TRANSCRIPT_KEY = "points" # NUMBER FOR EACH IMAGE
    CELL_ID = "cell_ID"
    NUCLEUS_ID = "overlaps_nucleus"
    UNASSIGNED_CELL_ID = 0
    # GENE_EXCLUDE_PATTERN = "NegPrb*"

    @staticmethod
    def is_unassigned(cell_id_series):
        return cell_id_series.eq(CosmxKeys.UNASSIGNED_CELL_ID)

    
PADJ_COLUMN = "padj"
BASE_MEAN_COLUMN = "baseMean"
PCT_COLUMN = "pct"
LOGFC_COLUMN = "log2FoldChange"
# FEATURE_KEY = "gene"
PACKAGE_KEY = "sparty"
SUPPORTED_TECHNOLOGIES = ("xenium", "merscope", "cosmx")

GENE_EXCLUDE_PATTERN = "Unassigned.*|Deprecated.*|Intergenic.*|Neg.*|Blank-*|NegPrb*|BLANK_*"

# GENE_EXCLUDE_PATTERN = "nan|<NA>|.*control.*|blank.*|antisense.*|unassigned.*|deprecated.*|intergenic.*|false.*|neg.*"
# VALID_DIMENSIONS = ("c", "y", "x")
# LOW_AVERAGE_COUNT = 0.01
# ATTRS_KEY = "spatialdata_attrs"