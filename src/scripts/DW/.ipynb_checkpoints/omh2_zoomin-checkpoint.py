import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import paths
import scipy.stats as ss
import seaborn as sns

if __name__ == '__main__':
    rng = np.random.default_rng(19465866273830024359462994947182329018)

    sns.set_palette('colorblind')
    trace = az.from_netcdf(op.join(paths.data, 'trace_cosmo.nc'))

    trace.posterior['Omh2'] = trace.posterior.Om * np.square(trace.posterior.h)

    hp = []
    Om = []
    while len(hp) < 4000:
        h = np.random.normal(loc=0.7, scale=0.2)
        O = np.random.normal(loc=0.3, scale=0.15)

        if h < 0.35 or h > 1.4 or O < 0 or O > 1:
            continue
        hp.append(h)
        Om.append(O)
    hp = np.array(hp)
    Om = np.array(Om)
    omh2 = Om*np.square(hp)

    sns.kdeplot(trace.posterior.Omh2.values.flatten(), label='Posterior')
    sns.kdeplot(omh2, label='Prior', color='k')

    plt.xlim(0, 0.5)

    plt.xlabel(r'$\omega_M \equiv \Omega_M h^2$')
    plt.legend()

    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'omh2_zoomin.pdf'))