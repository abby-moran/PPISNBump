import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import paths
import scipy.stats as ss
import seaborn as sns

if __name__ == '__main__':
    sns.set_palette('colorblind')
    trace = az.from_netcdf(op.join(paths.data, 'trace_cosmo.nc'))

    df = pd.DataFrame({r'$h$': trace.posterior.h.values.flatten()})
    sns.kdeplot(df, label='Posterior')

    x = np.linspace(0.35, 1.4, 1024)
    d = ss.norm(loc=0.7, scale=0.2)
    plt.plot(x, d.pdf(x)/(d.cdf(1.4)-d.cdf(0.35)), color='k', label='Prior')

    plt.xlim(0.35, 1.4)

    plt.xlabel(r'$h$')

    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'h_zoomin.pdf'))

