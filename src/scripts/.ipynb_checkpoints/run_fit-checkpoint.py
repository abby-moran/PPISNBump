import arviz as az
import intensity_models
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import os.path as op
import pandas as pd
import paths

nmcmc = 1000
nchain = 4
ndevice = 8
random_seed = 3281922803

if __name__ == '__main__':
    numpyro.set_host_device_count(ndevice) # I guess this will work remotely, too?

    pe_samples = pd.read_hdf(op.join(paths.data, 'pe-samples.h5'), 'samples')
    sel_samples = pd.read_hdf(op.join(paths.data, 'selection-samples.h5'), 'samples')

    evts = pe_samples.groupby('evt')
    m1s = []
    qs = []
    zs = []
    pdraws = []
    for (n, e) in evts:
        m1s.append(e['m1'])
        qs.append(e['q'])
        zs.append(e['z'])
        pdraws.append(e['wt'])

    m1s, qs, zs, pdraws = map(np.array, [m1s, qs, zs, pdraws])

    kernel = NUTS(intensity_models.pop_model)
    mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain)
    mcmc.run(jax.random.PRNGKey(random_seed), 
             m1s, qs, zs, pdraws, 
             sel_samples['m1'], sel_samples['q'], sel_samples['z'], sel_samples['pdraw'], sel_samples['ndraw'].iloc[0])
    
    trace = az.from_numpyro(mcmc)
    az.to_netcdf(trace, op.join(paths.data, 'trace.nc'))
