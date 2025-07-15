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
from pprint import pprint

#@jax.jit
def mean_mbh_from_mco(mco, mpisn, mbhmax):
    """The mean black hole mass from the core-mass to remnant-mass relation.
    
    :param mco: The CO core mass. should be of shape (shape 1, nmco, 1)
    :param mpisn: The BH mass at which the relation starts to turn over. shape (shape n_z,1,1)
    :param mbhmax: The maximum BH mass achieved by the relation. 
    """
    jax.debug.print('mpisn {}', mpisn.shape)
    jax.debug.print('mco {}', mco.shape)
    a = 1 / (4*(mpisn - mbhmax))
    mcomax = 2*mbhmax - mpisn
    return jnp.where(jnp.asarray(mco) < jnp.asarray(mpisn), mco, mbhmax + a*jnp.square(jnp.asarray(mco) - jnp.asarray(mcomax)))

#@jax.jit
def largest_mco(mpisn, mbhmax):
    """The largest CO core mass with positive BH masses."""
    mcomax = 2*mbhmax - mpisn
    return mcomax + jnp.sqrt(4*mbhmax*(mbhmax - mpisn))

#@jax.jit
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

#@jax.jit
def smooth_log_dNdmCO(xx, a, b):
    xtr = 20
    delta = 0.05
    return -a * jnp.log(xx / xtr) + delta * (a - b) * jnp.log(0.5 * (1 + (xx/xtr)**(1/delta)))

#@jax.jit
def log_smooth_turnon(m, mmin, width=0.05):
    """A function that smoothly transitions from 0 to 1.
    
    :param m: The function argument.
    :param mmin: The location around which the function transitions.
    :param width: (optional) The fractional width of the transition.
    """
    dm = mmin*width

    return np.log(2) - jnp.log1p(jnp.exp(-(m-mmin)/dm))
#@jax.jit
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

#@jax.jit
def log_dNdm_pisn(a, b, mpisn_arr, mbhmax_arr, sigma, m_grid):
    """
    Vectorized over redshift samples (mpisn, mbhmax) of shape (n_z,).
    Returns array of shape (n_z, len(m_grid)).
    """
    n_m = 1800
    min_co_mass, max_co_mass = 1.0, 100.0
    min_bh_mass, max_bh_mass = 3.0, 100.0

    log_mco = jnp.linspace(jnp.log(min_co_mass), jnp.log(max_co_mass), n_m)
    log_mbh = jnp.linspace(jnp.log(min_bh_mass), jnp.log(max_bh_mass), n_m + 2)

    mco = jnp.exp(log_mco)  # (n_m,)
    mbh = jnp.exp(log_mbh)  # (n_m+2,)
    dmco = jnp.diff(mco)    # (n_m-1,)

    mco = mco[None, :, None]  # (1, n_m, 1) — will broadcast with (n_z, 1, 1)

    # Reshape mpisn and mbhmax to (n_z, 1, 1)
    mpisn_arr = mpisn_arr[:, None, None]
    mbhmax_arr = mbhmax_arr[:, None, None]

    mu = mean_mbh_from_mco(mco, mpisn_arr, mbhmax_arr)  # (n_z, n_m, 1)
    mu = jnp.where(mu > 0, mu, 0.1)
    log_mu = jnp.log(mu)  # (n_z, n_m, 1)

    log_mbh = jnp.log(mbh)[None, None, :]  # (1, 1, n_m+2)

    log_p = (
        -0.5 * ((log_mbh - log_mu) / sigma) ** 2
        - 0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(sigma)
        - log_mbh
    )  # (n_z, n_m, n_m+2)

    log_dNdmCO_vals = log_dNdmCO(mco.squeeze(), a, b)  # (n_m,)
    log_dNdmCO_vals = log_dNdmCO_vals[None, :, None]   # (1, n_m, 1)

    log_weights = log_dNdmCO_vals + log_p  # (n_z, n_m, n_m+2)

    # Integrate over mco
    lw1 = log_weights[:, :-1, :]
    lw2 = log_weights[:, 1:, :]
    integrand = jnp.log(0.5) + jnp.logaddexp(lw1, lw2) + jnp.log(dmco)[None, :, None]  # (n_z, n_m-1, n_m+2)
    log_integral = jss.logsumexp(integrand, axis=1)  # (n_z, n_m+2)

    # Interpolate onto requested m_grid
    return jax.vmap(lambda logint: jnp.interp(m_grid, mbh, logint, left=-jnp.inf, right=-jnp.inf))(log_integral)

#@jax.jit
def log_dN_dV_dt(z, lam, kappa, zp, zmax=20, zref=0.001):
    # Normalize at zref
    def f(z_):
        return jnp.where(z_ < zmax, lam*jnp.log1p(z_) - jnp.log1p(((1+z_)/(1+zp))**kappa), -jnp.inf)
    log_norm = -f(zref)
    return f(z) + log_norm

#@jax.jit
def log_dNdm(m, z, a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl,
             mbh_min, delta_m, zmax=20., mref=30.0, zref=0.001):
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)

    # Set redshift grid
    z_grid = jnp.expm1(jnp.linspace(jnp.log(1), jnp.log(1 + zmax), 30))  # (nz,)
    n_z = z_grid.shape[0]

    # Redshift evolution of PISN and MBHmax
    mpisn_grid = mpisn + mpisndot * (1 - 1 / (1 + z_grid))                # (nz,)
    mbhmax_grid = mpisn_grid + (mbhmax - mpisn)                           # (nz,)

    # Evaluate log_dNdm_pisn over (z_grid, m)
    def log_dNdm_pisn_single(mpisn_i, mbhmax_i):
        n_m = 1800
        log_mco = jnp.linspace(jnp.log(1.0), jnp.log(100.0), n_m)
        log_mbh = jnp.linspace(jnp.log(3.0), jnp.log(100.0), n_m + 2)

        mco = jnp.exp(log_mco)
        mbh = jnp.exp(log_mbh)
        dmco = jnp.diff(mco)

        mco = mco[None, :, None]
        mbh = mbh[None, None, :]

        mu = mean_mbh_from_mco(mco, mpisn_i[None, None], mbhmax_i)
        mu = jnp.clip(mu, 0.1)
        log_mu = jnp.log(mu)
        log_mbh_arr = jnp.log(mbh)

        log_p = (
            -0.5 * ((log_mbh_arr - log_mu) / sigma) ** 2
            - 0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(sigma)
            - log_mbh_arr
        )

        log_dNdmCO_vals = log_dNdmCO(mco.squeeze(), a, b)[:, None]
        log_weights = log_dNdmCO_vals + log_p

        integrand = (
            jnp.log(0.5)
            + jnp.logaddexp(log_weights[:-1], log_weights[1:])
            + jnp.log(dmco[:, None])
        )
        log_integral = jss.logsumexp(integrand, axis=0)  # (n_m+2,)

        return jnp.interp(m, jnp.exp(log_mbh), log_integral, left=-jnp.inf, right=-jnp.inf)

    # Vectorize over redshift grid
    log_vals_grid = jax.vmap(log_dNdm_pisn_single)(mpisn_grid, mbhmax_grid)  # (nz, len(m))

    # Interpolate over redshift for each z
    log_vals_grid_T = log_vals_grid.T  # (nm, nz)
    log_dNdm_vals = jax.vmap(lambda logv: jnp.interp(z, z_grid, logv, left=-jnp.inf, right=-jnp.inf))(log_vals_grid_T)  # (nm, nz)

    # Smooth taper below mbh_min
    log_window = mmin_log_smooth_turnon(m, delta_m=delta_m, mmin=mbh_min)  # (nm,)
    log_dNdm_vals = jnp.where(m[:, None] >= 100.0, -jnp.inf, log_dNdm_vals)

    # Add power-law tail above mbhmax(z)
    z_bcast = z[None, :]
    m_bcast = m[:, None]
    mpisn_z = mpisn + mpisndot * (1 - 1/(1 + z_bcast))
    mbhmax_z = mpisn_z + (mbhmax - mpisn)

    log_dN_at_mbhmax = jax.vmap(lambda mmax, zval: log_dNdm_pisn_single(mmax[None], mmax)[0])(mbhmax_z.flatten(), z.flatten())
    log_dN_at_mbhmax = log_dN_at_mbhmax[None, :]

    log_tail = -c * jnp.log(m_bcast / mbhmax_z)
    log_tail += jnp.log(fpl) + log_dN_at_mbhmax
    log_tail += log_smooth_turnon(m_bcast, mbhmax_z)

    # Combine PISN and tail
    log_dNdm_total = jnp.logaddexp(log_dNdm_vals, log_tail)

    # Apply turn-on window
    return log_dNdm_total + log_window[:, None]  # shape (nm, nz)




#@jax.jit
def log_dN_dM_dq_dV_dt( m1, q, z, a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl,
    beta, lam, kappa, zp, mref=30.0, qref=1.0, zref=0.001, mbh_min=5.0, delta_m=2.5, zmax=20):
    # Inner function: unnormalized log rate
    def unnorm_log_rate(m1_, q_, z_):
        m2 = q_ * m1_
        mtot = m1_ + m2
        
        log_dN_m1 = log_dNdm(m1_, z_, a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl, mbh_min, delta_m, zmax)
        log_dN_m2 = log_dNdm(m2, z_, a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl, mbh_min, delta_m, zmax)
        log_dN_v = log_dN_dV_dt(z_, lam, kappa, zp, zmax, zref)
        
        return (log_dN_m1 + log_dN_m2 + beta * jnp.log(mtot / (mref * (1 + qref))) + jnp.log(m1_) + log_dN_v)
    
    # Compute unnormalized values for input
    val = unnorm_log_rate(m1, q, z)
    
    # Compute reference value at (mref, qref, zref)
    log_ref = unnorm_log_rate(mref, qref, zref)
    
    # Normalize by subtracting reference value and add log(mref) for consistent normalization
    return val - log_ref + jnp.log(mref)

    
def E(z, Om, w):
    opz = 1 + z
    return jnp.sqrt(Om * opz**3 + (1 - Om) * opz**(3*(1 + w)))

def dH(h):
    return 2.99792 / h

def dc_integrand(z, Om, w):
    return 1 / E(z, Om, w)

def dC(z, h, Om, w):
    # Integrate from 0 to z numerically using cumtrapz approximation
    # For speed, use fixed quadrature or jax.scipy.integrate.odeint or jax.scipy.integrate.quad if available
    # Here use jax.scipy.integrate.cumtrapz like function or fixed quadrature:
    from jax.scipy.integrate import quad
    
    def integrand(z_):
        return 1 / E(z_, Om, w)
    
    # Use quad:
    val, _ = jsi.quad(integrand, 0.0, z, args=(), rtol=1e-5)
    return dH(h) * val

def dL(z, h, Om, w):
    return dC(z, h, Om, w) * (1 + z)

def VC(z, h, Om, w):
    dc = dC(z, h, Om, w)
    return 4/3 * jnp.pi * dc**3

def dVCdz(z, h, Om, w):
    dc = dC(z, h, Om, w)
    return 4 * jnp.pi * dc**2 * dH(h) / E(z, Om, w)

def ddL_dz(z, h, Om, w):
    # derivative of dL w.r.t z
    dc = dC(z, h, Om, w)
    ddl = dc + dH(h) * (1 + z) / E(z, Om, w)
    return ddl

def z_of_dL(dL_val, h, Om, w, zmax=20):
    # root finding, e.g. binary search for z so that dL(z) = dL_val
    # Implement a bisection or jax-friendly root finder:
    def func(z_):
        return dL(z_, h, Om, w) - dL_val
    
    # Bisection between 0 and zmax
    def bisection(f, a, b, tol=1e-5, maxiter=50):
        def cond(state):
            _, a, b, i = state
            return jnp.logical_and(i < maxiter, (b - a) > tol)
        def body(state):
            f, a, b, i = state
            mid = 0.5*(a+b)
            f_mid = f(mid)
            a_new = jnp.where(f_mid * f(a) < 0, a, mid)
            b_new = jnp.where(f_mid * f(a) < 0, mid, b)
            return (f, a_new, b_new, i+1)
        f, a, b, i = lax.while_loop(cond, body, (f, a, b, 0))
        return 0.5*(a+b)
    return bisection(func, 0.0, zmax)


coords = {
    'm_grid': np.exp(np.linspace(np.log(1), np.log(150), 128)),
    'q_grid': np.linspace(0, 1, 129)[1:],
    'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(20), 128))
}

mref: object = 30.0
zref: object = 0.001
qref: object = 1.0

#@jax.jit
def get_deterministic_parameters(sample):
    kappa = numpyro.deterministic('kappa', sample['lam'] + sample['dkappa'])
    fpl = numpyro.deterministic('fpl', jnp.exp(sample['log_fpl']))
    mbhmax = numpyro.deterministic('mbhmax', sample['mpisn'] + sample['dmbhmax'])   
    return dict(kappa=kappa, fpl=fpl, mbhmax=mbhmax)


def cumtrapz(y, x):
    # Simple trapezoidal cumulative integral (like jnp_cumtrapz)
    dx = jnp.diff(x)
    mid = (y[:-1] + y[1:]) / 2
    cumsum = jnp.concatenate([jnp.array([0]), jnp.cumsum(mid * dx)])
    return cumsum

def precompute_cosmo_arrays(zmax, ninterp, Om, h, w):
    zinterp = jnp.expm1(jnp.linspace(jnp.log(1), jnp.log(1+zmax), ninterp))
    dH_val = dH(h)
    invE = 1/E(zinterp, Om, w)
    dcinterp = dH_val * cumtrapz(invE, zinterp)
    dlinterp = dcinterp * (1 + zinterp)
    ddlinterp = dcinterp + dH_val * (1 + zinterp) / E(zinterp, Om, w)
    vcinterp = 4/3 * jnp.pi * dcinterp**3
    dvcinterp = 4 * jnp.pi * dcinterp**2 * dH_val / E(zinterp, Om, w)
    return zinterp, dcinterp, dlinterp, ddlinterp, vcinterp, dvcinterp

def interp_1d(x, xp, fp):
    # safe linear interp
    return jnp.interp(x, xp, fp, left=-jnp.inf, right=-jnp.inf)

def z_of_dL(dL, zinterp, dlinterp):
    return jnp.interp(dL, dlinterp, zinterp, left=0.0, right=zinterp[-1])

def dVCdz(z, zinterp, dvcinterp):
    return interp_1d(z, zinterp, dvcinterp)

def ddL_dz(z, zinterp, ddlinterp):
    return interp_1d(z, zinterp, ddlinterp)

def log_dndmdqdv(m1, q, z, 
                 a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl, beta, lam, kappa, zp,
                 mref=30., qref=1., zref=0.001, zmax=20., mbh_min=5., delta_m=2.5):
    # Compute dmbhmax(z)
    dmbhmax = mbhmax - mpisn
    mpisns = mpisn + mpisndot * (1 - 1/(1+z))
    mbhmaxs = mpisns + dmbhmax
    
    m2 = q * m1
    mt = m1 + m2
    val = (log_dNdm(m1, z, a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl, mbh_min, delta_m,) + 
           log_dNdm(m2, z, a, b, c, mpisn, mpisndot, mbhmax, sigma, fpl, mbh_min, delta_m,) + beta*jnp.log(mt/(mref*(1 + qref))) +
           jnp.log(m1) + lam*jnp.log1p(z) - jnp.log1p(((1+z)/(1+zp))**kappa))
    return val


def pop_cosmo_model(m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel, Ndraw, priors):
    m1s_det, qs, dls, pdraw = map(jnp.array, (m1s_det, qs, dls, pdraw))
    m1s_det_sel, qs_sel, dls_sel, pdraw_sel = map(jnp.array, (m1s_det_sel, qs_sel, dls_sel, pdraw_sel))

    log_pdraw_sel = jnp.log(pdraw_sel)
    log_pdraw = jnp.log(pdraw)
    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]
    nsel = m1s_det_sel.shape[0]

    # Sample parameters from priors dictionary
    sample = sample_parameters_from_dict(priors)
    # Compute deterministic parameters
    kappa = sample['lam'] + sample['dkappa']
    fpl = jnp.exp(sample['log_fpl'])
    mbhmax = sample['mpisn'] + sample['dmbhmax']

    # Cosmology arrays
    zinterp, dcinterp, dlinterp, ddlinterp, vcinterp, dvcinterp = precompute_cosmo_arrays(
        sample['zmax'], 1024, sample['Om'], sample['h'], sample['w']
    )

    # Convert dL to z for detected events and injections
    zs = z_of_dL(dls, zinterp, dlinterp)
    zs_sel = z_of_dL(dls_sel, zinterp, dlinterp)

    # Convert to source frame
    m1s = m1s_det / (1 + zs)
    m1s_sel = m1s_det_sel / (1 + zs_sel)

    # Compute log population model values
    log_dN = lambda m1, q, z: log_dndmdqdv(
        m1, q, z,
        a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'], mpisndot=sample['mpisndot'],
        mbhmax=mbhmax, sigma=sample['sigma'], fpl=fpl, beta=sample['beta'], lam=sample['lam'], kappa=kappa, zp=sample['zp'],
        mref=30., qref=1., zref=0.001, zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m']
    )

    # Calculate weights for detected events
    log_wts = (log_dN(m1s, qs, zs) - 2*jnp.log1p(zs) +
               jnp.log(dVCdz(zs, zinterp, dvcinterp)) - jnp.log(ddL_dz(zs, zinterp, ddlinterp)) - log_pdraw)
    log_like = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)
    log_like = jnp.nan_to_num(jnp.sum(log_like), nan=-jnp.inf)

    numpyro.factor('loglike', log_like)

    # Calculate weights for injections
    log_sel_wts = (log_dN(m1s_sel, qs_sel, zs_sel) - 2*jnp.log1p(zs_sel) +
                   jnp.log(dVCdz(zs_sel, zinterp, dvcinterp)) - jnp.log(ddL_dz(zs_sel, zinterp, ddlinterp)) - log_pdraw_sel)
    log_mu_sel = jss.logsumexp(log_sel_wts) - jnp.log(Ndraw)
    numpyro.factor('selfactor', jnp.nan_to_num(-nobs * log_mu_sel, nan=-jnp.inf))

    log_mu2 = jss.logsumexp(2*log_sel_wts) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_mu_sel - jnp.log(Ndraw) - log_mu2))
    neff_sel = jnp.exp(2*log_mu_sel - log_s2)
    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs/mu_sel + jnp.sqrt(nobs)/mu_sel * R_unit)

    # Effective sample size for detected events
    neff = jnp.exp(2*jss.logsumexp(log_wts, axis=1) - jss.logsumexp(2*log_wts, axis=1))
    numpyro.deterministic('neff', neff)

    log_dN_mgrid = log_dndmdqdv(
        coords['m_grid'], qref, zref,
        a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'], mpisndot=sample['mpisndot'],
        mbhmax=mbhmax, sigma=sample['sigma'], fpl=fpl, beta=sample['beta'], lam=sample['lam'], kappa=kappa, zp=sample['zp'],
        mref=mref, qref=qref, zref=zref, zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m'])
    numpyro.deterministic('mdNdmdVdt_fixed_qz', coords['m_grid'] * R * jnp.exp(log_dN_mgrid))
    
    log_dN_qgrid = log_dndmdqdv(
        mref, coords['q_grid'], zref,
        a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'], mpisndot=sample['mpisndot'],
        mbhmax=mbhmax, sigma=sample['sigma'], fpl=fpl, beta=sample['beta'], lam=sample['lam'], kappa=kappa, zp=sample['zp'],
        mref=mref, qref=qref, zref=zref, zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m'])
    numpyro.deterministic('dNdqdVdt_fixed_mz', mref * R * jnp.exp(log_dN_qgrid))
    
    log_dN_zgrid = log_dndmdqdv(
        mref, qref, coords['z_grid'],
        a=sample['a'], b=sample['b'], c=sample['c'], mpisn=sample['mpisn'], mpisndot=sample['mpisndot'],
        mbhmax=mbhmax, sigma=sample['sigma'], fpl=fpl, beta=sample['beta'], lam=sample['lam'], kappa=kappa, zp=sample['zp'],
        mref=mref, qref=qref, zref=zref, zmax=sample['zmax'], mbh_min=sample['mbh_min'], delta_m=sample['delta_m'] )
    numpyro.deterministic('dNdVdt_fixed_mq', mref * R * jnp.exp(log_dN_zgrid))
    
    hz = sample['h'] * E(coords['z_grid'], Om=sample['Om'], w=sample['w'])
    numpyro.deterministic('hz', hz)
    numpyro.deterministic('mbhmax', mbhmax)
    numpyro.deterministic('fpl', fpl)
    numpyro.deterministic('kappa', kappa)
    numpyro.deterministic('neff_sel', neff_sel)

