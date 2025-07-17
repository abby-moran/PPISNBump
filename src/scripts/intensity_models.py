from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import jax.scipy.stats as jsst
import jax.scipy.integrate as jsi
from jax import lax
import numpy as np
import numpyro 
import numpyro.distributions as dist
from utils import jnp_cumtrapz, sample_parameters_from_dict, log_expit
from jax.scipy.ndimage import map_coordinates

@jax.jit
def mean_mbh_from_mco(mco, mpisn, mbhmax):
    """The mean black hole mass from the core-mass to remnant-mass relation.
    
    :param mco: The CO core mass. shape (1, nmco, 1)
    :param mpisn: The BH mass at which the relation starts to turn over. shape (n_z,1,1)
    :param mbhmax: The maximum BH mass achieved by the relation.
    """
    #jax.debug.print('mpisn {}', mpisn.shape)
    #jax.debug.print('mco {}', mco.shape)
    a = 1 / (4*(mpisn - mbhmax))
    mcomax = 2*mbhmax - mpisn
    return jnp.where(mco < mpisn, mco, mbhmax + a*jnp.square(mco - mcomax))
@jax.jit
def largest_mco(mpisn, mbhmax):
    """The largest CO core mass with positive BH masses."""
    mcomax = 2*mbhmax - mpisn
    return mcomax + jnp.sqrt(4*mbhmax*(mbhmax - mpisn))
@jax.jit
def log_dNdmCO(mco, a, b):
    r"""The broken power law CO core mass function.
    
    The power law breaks (smoothly) at :math:`16 \, M_\odot` (i.e. a BH mass of :math:`20 \, M_\odot`).

    :param mco: The CO core mass.
    :param a: The power law slope for small CO core masses.
    :param b: The power law slope for large CO core masses.
    """
    mtr = 20.0
    x = mco/mtr
    return jnp.where(mco < mtr, -a*jnp.log(x), -b*jnp.log(x))
@jax.jit
def smooth_log_dNdmCO(xx, a, b):
    xtr = 20
    delta = 0.05
    return -a * jnp.log(xx / xtr) + delta * (a - b) * jnp.log(0.5 * (1 + (xx/xtr)**(1/delta)))
@jax.jit
def log_smooth_turnon(m, mmin, width=0.05):
    """A function that smoothly transitions from 0 to 1.
    
    :param m: The function argument.
    :param mmin: The location around which the function transitions.
    :param width: (optional) The fractional width of the transition.
    """
    dm = mmin*width

    return np.log(2) - jnp.log1p(jnp.exp(-(m-mmin)/dm))
@jax.jit
def mmin_log_smooth_turnon(m, delta_m, mmin):
    """Log of a function that smoothly transitions from 0 to 1 over the interval [mmin, mmin + delta_m].
    Written to be consistent with Planck taper turnon in powerlaw+peak in LVK population papers
    (Eq. B5-B6 in arXiv:2111.03634). Adapted from https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/models/mass.py#L628"""
    shifted_mass = jnp.nan_to_num((m - mmin) / delta_m, nan=0)
    shifted_mass = jnp.clip(shifted_mass, 1e-6, 1 - 1e-6)
    exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
    exponent = jnp.where(exponent > 87.0, 87.0, exponent)
    window = jax.lax.logistic(-exponent)
    logwindow = jnp.where(m < mmin, -jnp.inf, jnp.log(window))
    return logwindow

@dataclass
class LogDNDMPISN(object):
    r"""
    Function-like object representing the compact-object mass function induced
    by the PISN process.
    
    .. math::
        \frac{\mathrm{d} N}{\mathrm{d} m} = \int \mathrm{d} m_\mathrm{CO} \, \frac{\mathrm{d} N}{\mathrm{d} m_\mathrm{CO}} p\left( m \mid \mu\left( m_\mathrm{CO} \mid m_\mathrm{PISN}, m_\mathrm{BH,max} \right), \sigma \right),

    where the CO-remnant mass relationship is 

    .. math::
        \mu\left( m_\mathrm{CO} \mid m_\mathrm{PISN}, m_\mathrm{BH,max} \right) = \begin{cases}
            m_\mathrm{CO} + 4 & m_\mathrm{CO} \leq m_\mathrm{PISN} - 4 \\
            m_\mathrm{BH,max} - a \left( m_\mathrm{CO} - b \right^2 & \mathrm{otherwise}
        \end{cases},

    with :math:`a` and :math:`b` chosen so that the relation is smooth when
    :math:`\mu = m_\mathrm{PISN}`; :math:`\sigma` is the amount of scatter
    around the mean CO-remnant mass relationship, and the distribution of
    offsets represented by :math:`p` is Gaussian.  

    Parameters `a` and `b` are passed to :func:`log_dNdmCO`.

    The (optional) `n_m` parameter controls how many points are used in the CO
    mass and compact object mass grids to perform the above integral.  The
    default setting works well for PISN/max BH masses around 50 MSun and `sigma`
    parameters larger than ~1 solar mass; if sigma decreases below that value,
    more grid points are needed to resolve the narrow width of the CO
    mass-remnant mass relation, and similarly if the peak mass scale increases.
    """
    a: object
    b: object
    mpisn: object
    mbhmax: object
    sigma: object
    n_m: object = 1800
    mbh_grid: object = dataclasses.field(init=False)
    log_dN_grid: object = dataclasses.field(init=False)
 
    def __post_init__(self):
        min_bh_mass = 3.0
        min_co_mass = 1.0
        max_bh_mass = 100.0
        max_co_mass = 100.0

        log_mbh = jnp.linspace(jnp.log(min_bh_mass), jnp.log(max_bh_mass), self.n_m+2)
        log_mco = jnp.linspace(jnp.log(min_co_mass), jnp.log(max_co_mass), self.n_m)


        # Array dimensions are (<redshift>, <MCO>, <MBH>).
        sigma = self.sigma
        log_mco = log_mco[None,:,None]
        log_mbh = log_mbh[None,None,:]
        mpisn = self.mpisn[:,None,None]
        mbhmax = self.mbhmax[:,None,None]

        mbh = jnp.exp(log_mbh)
        mco = jnp.exp(log_mco)

        mu = mean_mbh_from_mco(mco, mpisn, mbhmax)
        mu_min = 0.1
        mu = jnp.where(mu > 0, mu, mu_min)
        log_mu = jnp.log(mu)

        log_p = -0.5 * jnp.square((log_mbh - log_mu) / sigma) - 0.5*jnp.log(2*jnp.pi) - jnp.log(sigma) - log_mbh
        log_wts = log_dNdmCO(mco, self.a, self.b) + log_p
        log_trapz = np.log(0.5) + jnp.logaddexp(log_wts[:,:-1,:], log_wts[:,1:,:]) + jnp.log(jnp.diff(mco, axis=1))

        self.log_dN_grid = jss.logsumexp(log_trapz, axis=1)
        self.mbh_grid = mbh[0,0,:]


@dataclass
class LogDNDM(object):
    """
    Function-like object representing the full compact object mass function.

    A combination of :class:`LogDNDMPISN` above and a power law that begins at
    :math:`m = m_\mathrm{BH,max}` with relative amplitude `fpl` and slope `c`.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    mbh_min: object
    delta_m: object
    zmax: object = 20
    mref: object = 30.0
    zref: object = 0.001
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.dmbhmax = self.mbhmax - self.mpisn
        self.setup_interp()

    def setup_interp(self):
        self.z_array = jnp.expm1(jnp.linspace(np.log(1), jnp.log(1+self.zmax), 30))
        mpisns = self.mpisn + self.mpisndot * (1 - 1/(1+self.z_array))
        mbhmaxs = mpisns + self.dmbhmax
        self.log_dndm_pisn = LogDNDMPISN(self.a, self.b, mpisns, mbhmaxs, self.sigma)
        self.mbh_grid = self.log_dndm_pisn.mbh_grid
        self.log_dndm_pisn_grid = self.log_dndm_pisn.log_dN_grid.T
        self.mbhmaxs = jnp.asarray(mbhmaxs)
    
    def interp_2d_dndmpisn(self, m, z):
        m_indxs = jnp.searchsorted(self.mbh_grid, m)
        z_indxs = jnp.searchsorted(self.z_array, z)
        
        m_upper = self.mbh_grid[m_indxs]
        m_lower = self.mbh_grid[m_indxs - 1]
        
        z_upper = self.z_array[z_indxs]
        z_lower = self.z_array[z_indxs - 1]
        
        f1 = self.log_dndm_pisn_grid[m_indxs - 1, z_indxs - 1]
        f2 = self.log_dndm_pisn_grid[m_indxs, z_indxs - 1]
        f3 = self.log_dndm_pisn_grid[m_indxs, z_indxs]
        f4 = self.log_dndm_pisn_grid[m_indxs - 1, z_indxs]

        mdiffs = m_upper - m_lower
        zdiffs = z_upper - z_lower

        mdiffs = jnp.where(mdiffs != 0, mdiffs, np.inf)
        zdiffs = jnp.where(zdiffs != 0, zdiffs, np.inf)

        t = jnp.where(m_lower == m_upper, 0, (m - m_lower) / mdiffs)
        u = jnp.where(z_lower == z_upper, 0, (z - z_lower) / zdiffs)

        coefficients = jnp.array([(1 - t) * (1 - u), t * (1 - u), t * u, (1 - t) * u  ])
        fs = jnp.array([f1, f2, f3, f4])
        coefficients = jnp.where(jnp.isinf(fs), 1e-6, coefficients)

        return jnp.einsum('i...,i...',coefficients, fs)
        
    def __call__(self, m, z):
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z) 
        log_dNdm = self.interp_2d_dndmpisn(m, z)

        #log_dNdm = jnp.where(m <= self.log_dndm_pisn.mbh_grid[0], log_dNdm[0], log_dNdm)
        log_dNdm = jnp.where(m >= self.log_dndm_pisn.mbh_grid[-1], -np.inf, log_dNdm)
        
        mbhmax_at_samples = jnp.array(self.mpisn + self.mpisndot * (1 - 1/(1+z)) + self.dmbhmax)

        log_dNdmbhmax_at_samples = self.interp_2d_dndmpisn(mbhmax_at_samples, z)
        log_high_mass_tail = -self.c*jnp.log(jnp.divide(m, mbhmax_at_samples))    
        log_dNdm = jnp.logaddexp(log_dNdm, jnp.log(self.fpl) + log_dNdmbhmax_at_samples + log_high_mass_tail  + log_smooth_turnon(m, mbhmax_at_samples))
        #log_dNdm = jnp.where(m < self.mbh_min, -np.inf, log_dNdm)
        logwindow = mmin_log_smooth_turnon(m, delta_m=self.delta_m, mmin= self.mbh_min)
        log_dNdm = log_dNdm + logwindow
        return log_dNdm 
    
@dataclass
class LogDNDV(object):

    r"""
    Madau-Dickinson-like merger rate density over cosmic time:

    .. math::
        \frac{\mathrm{d} N}{\mathrm{d} V \mathrm{d} t} \propto \frac{\left( 1 + z \right)^\lambda}{1 + \left( \frac{1 + z}{1 + z_p} \right)^\kappa}
    """
    lam: object
    kappa: object
    zp: object
    zref: object = 0.001
    zmax: object = 20
    log_norm: object = 0.0

    def __post_init__(self):
        self.log_norm = -self(self.zref)

    def __call__(self, z):
        z = jnp.array(z)

        return jnp.where(z < self.zmax, self.lam*jnp.log1p(z) - jnp.log1p(((1+z)/(1+self.zp))**self.kappa) + self.log_norm, -np.inf)
    
@dataclass
class LogDNDMDQDV(object):
    r"""
    TODO: Document pairing function, arguments.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    beta: object
    lam: object
    kappa: object
    zp: object
    mref: object = 30.0
    qref: object = 1.0
    zref: object = 0.001
    zmax: object = 20
    mbh_min: object = 5.0
    delta_m: object = 2.5
    log_dndm: object = dataclasses.field(init=False)
    log_dndv: object = dataclasses.field(init=False)


    def __post_init__(self):
        self.log_dndm = LogDNDM(self.a, self.b, self.c, self.mpisn, self.mpisndot, self.mbhmax, self.sigma, self.fpl, mref=self.mref, 
                                zmax=self.zmax, zref = self.zref, mbh_min = self.mbh_min, delta_m=self.delta_m)
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref, zmax=self.zmax)
        self._normalize()

    def _normalize(self):
        self.log_norm = 0
        log_dN_ref = self(self.mref, self.qref, self.zref)

        # Want m_1 dN/d(m1)d(q)d(V)d(t) == 1 at reference for normalization (then the `R` parameter is the fitted dN/d(m1)d(q)d(V)d(t) at reference)
        self.log_norm = jnp.log(self.mref) + log_dN_ref

    def __call__(self, m1, q, z):
        m1 = jnp.array(m1)
        q = jnp.array(q)
        z = jnp.array(z)

        m2 = q*m1
        mt = m1+m2

        return self.log_dndm(m1, z) + self.log_dndm(m2, z) + self.beta*jnp.log(mt/(self.mref*(1 + self.qref))) + jnp.log(m1) + self.log_dndv(z) - self.log_norm
        
@dataclass
class FlatwCDMCosmology(object):
    """
    Function-like object representing a flat w-CDM cosmology.
    """
    h: object
    Om: object
    w: object
    zmax: object = 20.0
    ninterp: object = 1024
    zinterp: object = dataclasses.field(init=False)
    dcinterp: object = dataclasses.field(init=False)
    dlinterp: object = dataclasses.field(init=False)
    ddlinterp: object = dataclasses.field(init=False)
    vcinterp: object = dataclasses.field(init=False)
    dvcinterp: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.zinterp = jnp.expm1(jnp.linspace(np.log(1), jnp.log(1+self.zmax), self.ninterp))
        self.dcinterp = self.dH*jnp_cumtrapz(1/self.E(self.zinterp), self.zinterp)
        self.dlinterp = self.dcinterp*(1+self.zinterp)
        self.ddlinterp = self.dcinterp + self.dH*(1+self.zinterp)/self.E(self.zinterp)
        self.vcinterp = 4/3*np.pi*self.dcinterp*self.dcinterp*self.dcinterp
        self.dvcinterp = 4*np.pi*jnp.square(self.dcinterp)*self.dH/self.E(self.zinterp)

    @property
    def dH(self):
        return 2.99792 / self.h
    
    @property
    def Ol(self):
        return 1-self.Om
    
    @property
    def om(self):
        return self.Om*jnp.square(self.h)
    
    @property
    def ol(self):
        return self.Ol*jnp.square(self.h)
    
    def E(self, z):
        opz = 1 + z
        opz3 = opz*opz*opz
        return jnp.sqrt(self.Om*opz3 + (1-self.Om)*opz**(3*(1 + self.w)))

    def dC(self, z):
        return jnp.interp(z, self.zinterp, self.dcinterp)
    def dL(self, z):
        return jnp.interp(z, self.zinterp, self.dlinterp)
    def VC(self, z):
        return jnp.interp(z, self.zinterp, self.vcinterp)
    def dVCdz(self, z):
        return jnp.interp(z, self.zinterp, self.dvcinterp)
    
    def ddL_dz(self, z):
        return jnp.interp(z, self.zinterp, self.ddlinterp)

    def z_of_dC(self, dC):
        return jnp.interp(dC, self.dcinterp, self.zinterp)
    def z_of_dL(self, dL):
        return jnp.interp(dL, self.dlinterp, self.zinterp)

coords = {
    'm_grid': np.exp(np.linspace(np.log(1), np.log(150), 128)),
    'q_grid': np.linspace(0, 1, 129)[1:],
    'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(20), 128))
}
@jax.jit
def get_deterministic_parameters(sample):
    kappa = numpyro.deterministic('kappa', sample['lam'] + sample['dkappa'])
    fpl = numpyro.deterministic('fpl', jnp.exp(sample['log_fpl']))
    mbhmax = numpyro.deterministic('mbhmax', sample['mpisn'] + sample['dmbhmax'])   
    return dict(kappa=kappa, fpl=fpl, mbhmax=mbhmax)


def build_population_model(sample):
    return LogDNDMDQDV(
        a=sample['a'], b=sample['b'], c=sample['c'],
        mpisn=sample['mpisn'], mpisndot=sample['mpisndot'],
        mbhmax=sample['mbhmax'], sigma=sample['sigma'], fpl=sample['fpl'],
        beta=sample['beta'], lam=sample['lam'], kappa=sample['kappa'], zp=sample['zp'],
        zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m']
    )

    
def pop_cosmo_model(m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel, Ndraw, priors=None):
    m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel = map(jnp.array, (m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel))

    log_pdraw_sel = jnp.log(pdraw_sel)
    log_pdraw = jnp.log(pdraw)
    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]

    nsel = m1s_det_sel.shape[0]

    sample = sample_parameters_from_dict(priors)
    deterministic_parameters = get_deterministic_parameters(sample)
    sample.update(deterministic_parameters) #sample from hyperparameters, set up cosmology (cosmo) and population model (dN)

    cosmo = FlatwCDMCosmology(sample['h'], sample['Om'], sample['w'], zmax=sample['zmax'])
    log_dN = build_population_model(sample)  # ideally made outside if sample is static
  
    #log_dN = LogDNDMDQDV(a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'], mpisndot=sample['mpisndot'], 
     #                   mbhmax=sample['mbhmax'], sigma=sample['sigma'], fpl=sample['fpl'], beta=sample['beta'], 
      #                  lam=sample['lam'], kappa=sample['kappa'], zp=sample['zp'], zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m'])

    zs = cosmo.z_of_dL(dls)
    m1s = m1s_det / (1 + zs) # convert to source-frame masses

    # event-wise log weight for deteced events = log of pop density (model's differential merger rate density) 
    #      + Jacobian from source to detector frame 
    #      + cosmology factors (for volume element, D_L to z) + log_pdraw (corrects for non-uniform prior over parameter space)
    log_wts = log_dN(m1s, qs, zs) - 2*jnp.log1p(zs) + jnp.log(cosmo.dVCdz(zs)) - jnp.log(cosmo.ddL_dz(zs)) - log_pdraw
    log_like = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)
    log_like = jnp.nan_to_num(jnp.nan_to_num(jnp.sum(log_like), nan=-np.inf))
    
    #log_like_per_event = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)  # shape (nobs,)
    #log_like_per_event = jnp.nan_to_num(log_like_per_event, nan=-np.inf)
    #_=numpyro.deterministic("loglike", log_like_per_event)

    _ = numpyro.factor('loglike', log_like)

    zs_sel = cosmo.z_of_dL(dls_sel) 
    m1s_sel = m1s_det_sel / (1 + zs_sel)

    # now get weights for injected events as with log_wts
    log_sel_wts = log_dN(m1s_sel, qs_sel, zs_sel) - 2*jnp.log1p(zs_sel) + jnp.log(cosmo.dVCdz(zs_sel)) - jnp.log(cosmo.ddL_dz(zs_sel)) - log_pdraw_sel
    log_mu_sel = jss.logsumexp(log_sel_wts) - jnp.log(Ndraw)
    _ = numpyro.factor('selfactor', jnp.nan_to_num(jnp.nan_to_num(-nobs*log_mu_sel, nan=-np.inf)))

    log_mu2 = jss.logsumexp(2*log_sel_wts) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_mu_sel - jnp.log(Ndraw) - log_mu2))
    _ = numpyro.deterministic('neff_sel', jnp.exp(2*log_mu_sel - log_s2))
    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs/mu_sel + jnp.sqrt(nobs)/mu_sel*R_unit)

    # get some derived quantities for later
    # effective sample size for weights
    _ = numpyro.deterministic('neff', jnp.exp(2*jss.logsumexp(log_wts, axis=1) - jss.logsumexp(2*log_wts, axis=1)))

    # differential merger rate with respect to mass, at fixed reference values of mass ratio, z
    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)))

    # now varying q at fixed mass, z and then varying z at fixed mass, q (redshift evolution of merger rate)
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)))
    _ = numpyro.deterministic('dNdVdt_fixed_mq', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])))

    # dimensionless Hubble parameter at z
    _ = numpyro.deterministic('hz', cosmo.h*cosmo.E(coords['z_grid']))
