from dataclasses import dataclass

@dataclass(frozen=True)
class FilterThresholds:
    padj: float = 0.05
    logFC: float = 1.0
    baseMean: float = 0.0
    pct: float = 0.0

