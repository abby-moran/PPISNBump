from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as lalsim
import numpy as np
import os.path as op
import sys
import pandas as pd
import paths
from tqdm import tqdm, trange
import weighting
import scipy.integrate as sint
import intensity_models
import jimFisher
from jimFisher.Fisher import FisherSamples
import jax.numpy as jnp
from bilby.gw.conversion import component_masses_to_chirp_mass

def next_pow_2(x):
    np2 = 1
    while np2 < x:
        np2 = np2 << 1
    return np2

def compute_snrs(d, detectors = ['H1', 'L1'], sensitivity = 'aLIGO', fmin = 20, fmax = 2048, psdstart = 20):
    psdstop = 0.95*fmax
    fref = fmin

    fishers = dict()
    snrs = []
    for dur in [4, 8, 16, 32, 64, 128, 256, 512]:
        for det in detectors:
            name=f"fisher_{det}_{dur}s"
            fishers[name] = FisherSamples(name=name, fmin = 20, fmax = 2048, sensitivity=sensitivity, location=det, duration=dur,trigger_time=0,waveform="IMRPhenomD", f_ref=fref)

    for _, r in tqdm(d.iterrows(), total=len(d)):
        m2s = r.m1*r.q
        m1d = r.m1*(1+r.z)
        m2d = m2s*(1+r.z)

        a1 = np.sqrt(r.s1x*r.s1x + r.s1y*r.s1y + r.s1z*r.s1z)
        a2 = np.sqrt(r.s2x*r.s2x + r.s2y*r.s2y + r.s2z*r.s2z)


        T = max(4, next_pow_2(lalsim.SimInspiralChirpTimeBound(fmin, m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, a1, a2)))

        sn = []
        for det in detectors:
            fisher = fishers[f"fisher_{det}_{T}s"]
            params_here = {"mass_1": m1d, "mass_2": m2d, "s1_z": r.s1z, "s2_z": r.s2z, 
              "luminosity_distance": r.dL * 1e3, "phase_c": 0., "cos_iota": np.cos(r.iota), "ra": r.ra,
              "sin_dec": np.sin(r.dec), 'psi': r.psi, "t_c": 0., "s1_x": r.s1x, "s1_y": r.s1y, "s2_x": r.s2x, "s2_y": r.s2y}
            params_here['chirp_mass'] = component_masses_to_chirp_mass(params_here['mass_1'], params_here['mass_2'])
            
            params_here['mass_ratio'] = params_here['mass_2'] / params_here['mass_1']
            
            snr = fisher.get_snr(params_here)
            sn.append(snr)
        sn = jnp.array(sn)
        snrs.append(jnp.sqrt(jnp.sum(jnp.square(sn))))

    return snrs
