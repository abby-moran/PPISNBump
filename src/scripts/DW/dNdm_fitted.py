import arviz as az
from intensity_models import coords
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import seaborn as sns

if __name__ == '__main__':
    sns.set_palette('colorblind')

    trace = az.from_netcdf(op.join(paths.data, 'trace.nc'))

    d = ['chain', 'draw']
    dN = trace.posterior.mdNdmdVdt_fixed_qz 
    m = dN.median(dim=d)
    l = dN.quantile(0.16, dim=d)
    h = dN.quantile(0.84, dim=d)
    ll = dN.quantile(0.025, dim=d)
    hh = dN.quantile(0.975, dim=d)

    x = coords['m_grid']

    line, = plt.plot(x[1:], m[1:])
    plt.fill_between(x[1:], h[1:], l[1:], color=line.get_color(), alpha=0.25)
    plt.fill_between(x[1:], hh[1:], ll[1:], color=line.get_color(), alpha=0.25)

    plt.xlabel(r'$m_1 / M_\odot$')
    plt.ylabel(r'$\left. m_1 \mathrm{d}N/\mathrm{d}m_1 \mathrm{d} q \mathrm{d} V \mathrm{d} t \right|_{q=1,z=0} / \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}$')

    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(op.join(paths.figures, 'dNdm_fitted.pdf'))
