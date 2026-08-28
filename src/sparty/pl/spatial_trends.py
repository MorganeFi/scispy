import matplotlib.pyplot as plt
import numpy as np


def plot_param_by_sample(
    df_dict, 
    column, 
    ncols=4, 
    cmap='viridis',
    shared_scale=True, 
    figsize_per_ax=(4, 4),
    markersize=1, 
    **plot_kwargs
):
    """
    Affiche un subplot par sample pour visualiser une colonne donnée
    présente dans chaque GeoDataFrame du dico.

    Parameters
    ----------
    df_dict : dict[str, GeoDataFrame]
    column : str
        Colonne à visualiser (ex: 'dst_along_norm')
    ncols : int
        Nombre de colonnes dans la grille de subplots
    shared_scale : bool
        Si True, utilise un vmin/vmax commun à tous les samples
    """
    samples = list(df_dict.keys())
    n = len(samples)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_ax[0]*ncols,
                                                      figsize_per_ax[1]*nrows))
    axes = np.atleast_1d(axes).ravel()

    if shared_scale:
        vmin = min(df[column].min() for df in df_dict.values())
        vmax = max(df[column].max() for df in df_dict.values())
    else:
        vmin = vmax = None

    for ax, samp in zip(axes, samples):
        df_dict[samp].plot(
            column=column, ax=ax, cmap=cmap,
            vmin=vmin, vmax=vmax,
            markersize=markersize,
            legend=False,
            **plot_kwargs
        )
        ax.set_title(samp, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # cacher les axes vides
    for ax in axes[n:]:
        ax.axis('off')

    # une seule colorbar partagée
    if shared_scale:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        fig.colorbar(sm, ax=axes[:n].tolist(), shrink=0.6, label=column)

    return fig, axes