import os
import numpyro
import arviz as az
from astropy.cosmology import Planck18
import astropy.units as u
import sys
sys.path.append('../src/scripts/')
import intensity_models
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS, SA
import os.path as op
import pandas as pd
import paths
from utils import get_priors_from_file
from importlib import reload
from intensity_models import coords
import scipy.stats as ss

chain_id = int(sys.argv[1])
print(chain_id)
nmcmc = 500 #1000
#nchain = 4
random_seed = 1652819403

prior = get_priors_from_file("gwtc3_evolution.prior")

pe_samples = pd.read_hdf('pe_samples.h5', 'samples')
pe_samples['pdraw_cosmo']=pe_samples['prior_m1d_q_dL']

sel_samples = pd.read_hdf('selection_samples.h5', 'samples')
sel_samples['pdraw_cosmo'] = sel_samples['pdraw_m1sqz']*sel_samples['dm1sz_dm1ddl']
sel_samples['m1d'] = sel_samples['m1']*(1+sel_samples['z'])
sel_samples['dl'] = Planck18.luminosity_distance(sel_samples['z'].to_numpy()).to(u.Gpc).value

evts = pe_samples.groupby('evt')
m1s = []
qs = []
dls = []
pdraws = []
for (n, e) in evts:
    m1s.append(e['mass_1'])
    qs.append(e['mass_ratio'])
    dls.append(e['luminosity_distance_Gpc'])
    pdraws.append(e['pdraw_cosmo'])

m1s, qs, dls, pdraws = map(np.array, [m1s, qs, dls, pdraws])
ndraw = float(sel_samples['ndraw'][0])

kernel = NUTS(intensity_models.pop_cosmo_model)
mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=1,  progress_bar=True)
mcmc.run(jax.random.PRNGKey(random_seed),
         m1s, qs, dls, pdraws,
         sel_samples['m1d'], sel_samples['q'], sel_samples['dl'], sel_samples['pdraw_cosmo'], ndraw, prior)

trace = az.from_numpyro(mcmc)
trace = trace.assign_coords(chain=[chain_id])
az.to_netcdf(trace, f"trace_chain{chain_id}.nc")
print(f"Chain {chain_id} done")