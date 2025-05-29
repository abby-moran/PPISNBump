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
from fisher_snrs import compute_snrs

SENSITIVITIES = {'aLIGO': lalsim.SimNoisePSDaLIGODesignSensitivityP1200087,
                'aplus': lalsim.SimNoisePSDaLIGOAPlusDesignSensitivityT1800042,
                'CE': lalsim.SimNoisePSDCosmicExplorerP1600143}

def next_pow_2(x):
    np2 = 1
    while np2 < x:
        np2 = np2 << 1
    return np2

def compute_snrs_old(d, detectors = ['H1', 'L1'], sensitivity = 'aLIGO', fmin = 20, fmax = 2048, psdstart = 20):
    psdstop = 0.95*fmax
    snrs = []
    for _, r in tqdm(d.iterrows(), total=len(d)):
        m2s = r.m1*r.q
        m1d = r.m1*(1+r.z)
        m2d = m2s*(1+r.z)

        a1 = np.sqrt(r.s1x*r.s1x + r.s1y*r.s1y + r.s1z*r.s1z)
        a2 = np.sqrt(r.s2x*r.s2x + r.s2y*r.s2y + r.s2z*r.s2z)
        dl = r.dL * 1e9*lal.PC_SI

        fref = fmin

        T = next_pow_2(lalsim.SimInspiralChirpTimeBound(fmin, m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, a1, a2))
        df = 1/T

        Nf = int(round(fmax/df)) + 1
        fs = np.linspace(0, fmax, Nf)
        try:
            hp, hc = lalsim.SimInspiralChooseFDWaveform(m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, r.s1x, r.s1y, r.s1z, r.s2x, r.s2y, r.s2z, dl, r.iota, 0.0, 0.0, 0.0, 0.0, df, fmin, fmax, fref, None, lalsim.IMRPhenomD)
        except Exception as e:
            print(e.args)
            snrs.append(0)
            continue

        sn = []
        for det in detectors:
            h = lal.CreateCOMPLEX16FrequencySeries('h', hp.epoch, hp.f0, hp.deltaF, hp.sampleUnits, hp.data.data.shape[0])
            psd = lal.CreateREAL8FrequencySeries("psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0])

            dd = lal.cached_detector_by_prefix[det]
            Fp, Fc = lal.ComputeDetAMResponse(dd.response, r.ra, r.dec, r.psi, r.gmst)

            h.data.data = Fp*hp.data.data + Fc*hc.data.data

            SENSITIVITIES[sensitivity](psd, psdstart)

            sn.append(lalsim.MeasureSNRFD(h, psd, psdstart, psdstop))
        sn = np.array(sn)
        snrs.append(np.sqrt(np.sum(np.square(sn))))

    return snrs

class ZPDF(object):
    def __init__(self, lam, kappa, zp, zmax, cosmo):
        self.lam = lam
        self.kappa = kappa
        self.zp = zp

        self.zmax = zmax
        self.cosmo = cosmo

        self.zinterp = np.expm1(np.linspace(np.log(1), np.log(1+self.zmax), 1024))
        self.norm = 1

        unnorm_pdf = self(self.zinterp)
        
        self.norm = 1/np.trapz(unnorm_pdf, self.zinterp)
        self.pdfinterp = unnorm_pdf * self.norm

        self.cdfinterp = sint.cumtrapz(self.pdfinterp, self.zinterp, initial=0)

    def __call__(self, z):
        if self.cosmo == 'default':
            return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value / (1+z)
        else:
            return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * self.cosmo.dVCdz(z) / (1+z)

    def icdf(self, c):
        return np.interp(c, self.cdfinterp, self.zinterp)
    
class InterpolatedPDF(object):
    def __init__(self, xs, cdfs):
        self.xs = xs
        self.cdfs = cdfs / cdfs[-1]
        self.pdfs = np.diff(cdfs) / np.diff(xs)

    def __call__(self, x):
        x = np.atleast_1d(x)
        i = np.searchsorted(self.xs, x)-1

        return self.pdfs[i]
    
    def icdf(self, c):
        return np.interp(c, self.cdfs, self.xs)

class PowerLawPDF(object):
    def __init__(self, alpha, a, b):
        assert alpha > 1

        self.alpha = alpha
        self.a = a
        self.b = b

        self.norm = (self.a - (self.a/self.b)**self.alpha*self.b)/(self.a*(self.alpha-1))

    def __call__(self, x):
        return (self.a/x)**self.alpha/self.a/self.norm
    
    def icdf(self, c):
        return ((self.a**self.alpha*self.b*c + self.a*self.b**self.alpha*(1-c))/(self.a*self.b)**self.alpha)**(1/(1-self.alpha))


def calc_nex(df_det, default_settings, **kwargs):
    for key in df_det.keys():
        df_det[key] = np.array(df_det[key])
    if default_settings:
        h, Om, w = Planck18.h, Planck18.Om0, -1
        log_dN_func = weighting.default_log_dNdmdqdV
        rate = weighting.default_parameters.R
    else:
        pop_params = {key: kwargs[key] for key in ModelParameters().keys()}
        h, Om, w = kwargs['h'], kwargs['Om'], kwargs['w']
        rate = kwargs['R']
        log_dN_func = intensity_models.LogDNDMDQDV(**pop_params)
    if "cosmo" not in kwargs.keys():
        cosmo = intensity_models.FlatwCDMCosmology(h, Om, w)
    else:
        cosmo = kwargs.get("cosmo")
    log_dN = log_dN_func(df_det['m1'], df_det['q'], df_det['z'])
    nex = np.sum(rate*np.exp(log_dN)*cosmo.dVCdz(df_det['z'])*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)
    return nex



if __name__ == '__main__':

    config_file = sys.argv[1]
    outfile = sys.argv[2]

    population_parameters = dict()

    with open(config_file) as param_file:
        for line in param_file:
            (key, val) = line.split('=')
            population_parameters[key.strip()] = val.strip()
            try:
                population_parameters[key.strip()] = float(val.strip())
            except ValueError:
                pass
    
    snr_threshold = population_parameters.pop('snr_threshold', 0)
    ndraw = int(population_parameters.pop('ndraw', 1000000))
    sensitivity = population_parameters.pop('sensitivity', 'aLIGO')
    detectors = population_parameters.pop('detectors', 'H1,L1').split(',')
        
    custom_cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'], population_parameters['w'], population_parameters['zmax'])
    population_parameters['cosmo'] = custom_cosmo
    print("Using the following custom population_parameters: " + str(population_parameters))
    
    zpdf = ZPDF(lam=population_parameters["lam"], kappa=population_parameters["kappa"], zp=population_parameters["zp"], zmax = population_parameters.get("zmax", 20), cosmo=population_parameters["cosmo"])
    mpdf = PowerLawPDF(1.8, population_parameters["mbh_min"], 400)

    #rng = np.random.default_rng(333165393797366967556667466879860422123)
    rng = np.random.default_rng()

    #df = pd.DataFrame(columns = ['m1', 'q', 'z', 'iota', 'ra', 'dec', 'psi', 'gmst', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'pdraw_mqz', 'SNR_H1', 'SNR_L1', 'SNR_V1', 'SNR'])
    print("drawing zs and ms")
    z = zpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))
    m = mpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))
    print("drawing mts")
    mtpdf = PowerLawPDF(2, m+population_parameters['mbh_min'], 2 * m)

    mt = mtpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))

    m2 = mt - m
    q = m2/m

    print("calculating pdraws")
    pdraw = mpdf(m)*(mtpdf(mt)*m)*zpdf(z)

    m1d = m * (1 + z)
    iota = np.arccos(rng.uniform(low=-1, high=1, size=ndraw))

    ra = rng.uniform(low=0, high=2*np.pi, size=ndraw)
    dec = np.arcsin(rng.uniform(low=-1, high=1, size=ndraw))

    # 0 < psi < pi, uniformly distributed
    psi = rng.uniform(low=0, high=np.pi, size=ndraw)
    gmst = rng.uniform(low=0, high=2*np.pi, size=ndraw)

    print("assigning spins")

    s1x, s1y, s1z = 0,0,0#rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))
    s2x, s2y, s2z = 0,0,0#rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))

    print("calculating dLs")

    dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmo=population_parameters['cosmo'])
    dL = population_parameters['cosmo'].dL(z)

    df = pd.DataFrame({
        'm1': m,
        'q': q,
        'z': z,
        'dL': dL,
        'm1d': m1d,
        'iota': iota,
        'ra': ra,
        'dec': dec,
        'psi': psi,
        'gmst': gmst,
        's1x': s1x,
        's1y': s1y,
        's1z': s1z,
        's2x': s2x,
        's2y': s2y,
        's2z': s2z,
        'pdraw_mqz': pdraw,
        'dm1sz_dm1ddl': dm1sz_dm1ddl
    })
    if snr_threshold > 0:
        df['SNR'] = compute_snrs(df, detectors=detectors, sensitivity=sensitivity)
    else:
        df['SNR'] = 10000000
    p_pop_numerator = weighting.pop_wt(np.array(df['m1']), np.array(df['q']), np.array(df['z']), default=False, **population_parameters)

    df['p_pop_weight'] = p_pop_numerator / df['pdraw_mqz']
    df['p_pop_numerator'] = p_pop_numerator

    random_number = rng.uniform(low=0, high=1, size = len(p_pop_numerator))
    sel = random_number < (df['p_pop_weight'] / np.max(df['p_pop_weight']))
    population_samples = df[sel]
    df_det = population_samples[population_samples['SNR'] > snr_threshold]

    print(f"Retained {len(df_det)} samples after rejection sampling and applying snr cut.")

    df_det.to_hdf(outfile, key='true_parameters')

    #nex = np.sum(weighting.default_parameters.R*np.exp(weighting.default_log_dNdmdqdV(df_det['m1'], df_det['q'], df_det['z']))*Planck18.differential_comoving_volume(df_det['z']).to(u.Gpc**3/u.sr).value*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)
    #nex = calc_nex(df_det, default_settings = default, **population_parameters)
    #print('Found {:d} injections with SNR > {:d}'.format(np.sum(df['SNR'] > snr_threshold), snr_threshold))
    #print('Predicting {:.0f} detections per year'.format(nex))
