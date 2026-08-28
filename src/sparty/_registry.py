from ._filters import filter_xenium, filter_merscope, filter_cosmx
from ._constants import XeniumKeys, MerscopeKeys, CosmxKeys, SUPPORTED_TECHNOLOGIES

TECHNO_REGISTRY = {
    "xenium": {
        "keys": XeniumKeys,
        "filter_fn": filter_xenium,
        # "reader_fn": read_xenium,
        # "qc_fn": qc_xenium,            
    },
    "merscope": {
        "keys": MerscopeKeys,
        "filter_fn": filter_merscope,
        # "reader_fn": read_xenium,
        # "qc_fn": qc_xenium,            
    },
    "cosmx": {
        "keys": CosmxKeys,
        "filter_fn": filter_cosmx,
        # "reader_fn": read_xenium,
        # "qc_fn": qc_xenium,            
    },
}


assert set(TECHNO_REGISTRY) == set(SUPPORTED_TECHNOLOGIES), (
    "TECHNO_REGISTRY keys and SUPPORTED_TECHNOLOGIES are out of sync."
)