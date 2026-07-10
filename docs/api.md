# API

```{eval-rst}
.. module:: sparty
```

Import Sparty as:

```python
import sparty as spt
```

## Input/Output

```{eval-rst}
.. currentmodule:: sparty.io

.. automodule:: sparty.io
    :members:

```

## Preprocessing

```{eval-rst}
.. currentmodule:: sparty.pp

.. autosummary::
    :toctree: generated

    pp.run_scanpy
    pp.subsetSamples
    pp.metrics_summary
    pp.subset_transcripts
    pp.compute_gene_compartment_percentages
```


## Tools

```{eval-rst}
.. currentmodule:: sparty.tl

.. autofunction:: pseudobulk
.. autofunction:: alpha_shape_optimal
.. autofunction:: centerline
.. autofunction:: unassigned_RNA
```


## Plotting

```{eval-rst}
.. currentmodule:: sparty.pl

.. autosummary::
    :toctree: generated

    pl.barplotDE
    pl.stripPlotDE
    pl.plot_DE
    pl.gene_heatmaps
    pl.colocalization
    pl.plot_density
    pl.scis_prop
```
