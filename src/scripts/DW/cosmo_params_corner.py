import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import paths
import seaborn as sns

if __name__ == '__main__':
    trace = az.from_netcdf(op.join(paths.data, 'trace_cosmo.nc'))

    axes_labels = {'h': r'$h$', 
                   'Om': r'$\Omega_M$',
                   'w': r'$w$',
                   'mpisn': r'$m_\mathrm{PISN}$',
                   'mbhmax': r'$m_\mathrm{BH,max}$',
                   'sigma': r'$\sigma$'}
    df = pd.DataFrame({axes_labels[k]: trace.posterior[k].values.flatten() for k in axes_labels.keys()})

    pg = sns.PairGrid(df, diag_sharey=False)
    pg.map_diag(sns.kdeplot)
    pg.map_lower(sns.kdeplot)
    pg.map_upper(sns.scatterplot)

    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'cosmo_params_corner.pdf'))