import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def jnp_cumtrapz(ys, xs):
    """Cumulative trapezoidal integral of the function defined by values `ys` at points `xs`."""
    xs = jnp.array(xs)
    ys = jnp.array(ys)

    return jnp.concatenate((jnp.zeros(1), jnp.cumsum(0.5*jnp.diff(xs)*(ys[:-1] + ys[1:]))))

"""
Most of the stuff under this is cloned from Tom Callister's effective-spin-priors package: https://github.com/tcallister/effective-spin-priors
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import spence as PL

def Di(z):

    """
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html
    Inputs
    z: A (possibly complex) scalar or array
    Returns
    Array equivalent to PolyLog[2,z], as defined by Mathematica
    """

    return PL(1.-z+0j)

def chi_effective_prior_from_aligned_spins(xs, mass_ratio, a_max):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, aligned component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)

    if not isinstance(mass_ratio, np.ndarray):
        mass_ratio *= np.ones(len(xs))
    if not isinstance(a_max, np.ndarray):
        a_max *= np.ones(len(xs))

    masks = dict()
    masks["caseA"] = (xs>a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs<=a_max)
    masks["caseB"] = (xs<-a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs>=-a_max)
    masks["caseC"] = (xs>=-a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs<=a_max*(1.-mass_ratio)/(1.+mass_ratio))

    """
    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    """
    functions = dict()
    functions["caseA"] = lambda X, q, aMax: (1.+q)**2.*(aMax-X)/(4.*q*aMax**2)
    functions["caseB"] = lambda X, q, aMax: (1.+q)**2.*(aMax+X)/(4.*q*aMax**2)
    functions["caseC"] = lambda X, q, aMax: (1.+q)/(2.*aMax)
    for case in masks.keys():
        mask = masks[case]
        Xs = xs[mask]
        qs = mass_ratio[mask]
        amaxes = a_max[mask]
        pdfs[mask] = functions[case](X = Xs, q = qs, aMax = amaxes)
    return pdfs

def chi_effective_prior_from_isotropic_spins(xs, mass_ratio, a_max):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(np.abs(xs),-1)

    # Set up various piecewise cases
    pdfs = np.ones(xs.size,dtype=complex)*(-1.)
    if not isinstance(mass_ratio, np.ndarray):
        mass_ratio *= np.ones(len(xs))
    if not isinstance(a_max, np.ndarray):
        a_max *= np.ones(len(xs))

    """
    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    # Select relevant effective spins
    q_A = q[caseA]
    q_B = q[caseB]
    q_C = q[caseC]
    q_D = q[caseD]
    q_E = q[caseE]
    """
    functions = dict()
    functions["caseZ"] = lambda X, q, aMax: (1.+q)/(2.*aMax)*(2.-np.log(q))

    functions["caseA"] = lambda X, q, aMax: (1.+q)/(4.*q*aMax**2)*(
                    q*aMax*(4.+2.*np.log(aMax) - np.log(q**2*aMax**2 - (1.+q)**2*X**2))
                    - 2.*(1.+q)*X*np.arctanh((1.+q)*X/(q*aMax))
                    + (1.+q)*X*(Di(-q*aMax/((1.+q)*X)) - Di(q*aMax/((1.+q)*X)))
                    )

    functions["caseB"] = lambda X, q, aMax: (1.+q)/(4.*q*aMax**2)*(
                    4.*q*aMax
                    + 2.*q*aMax*np.log(aMax)
                    - 2.*(1.+q)*X*np.arctanh(q*aMax/((1.+q)*X))
                    - q*aMax*np.log((1.+q)**2*X**2 - q**2*aMax**2)
                    + (1.+q)*X*(Di(-q*aMax/((1.+q)*X)) - Di(q*aMax/((1.+q)*X)))
                    )

    functions["caseC"] = lambda X, q, aMax: (1.+q)/(4.*q*aMax**2)*(
                    2.*(1.+q)*(aMax-X)
                    - (1.+q)*X*np.log(aMax)**2.
                    + (aMax + (1.+q)*X*np.log((1.+q)*X))*np.log(q*aMax/(aMax-(1.+q)*X))
                    - (1.+q)*X*np.log(aMax)*(2. + np.log(q) - np.log(aMax-(1.+q)*X))
                    + q*aMax*np.log(aMax/(q*aMax-(1.+q)*X))
                    + (1.+q)*X*np.log((aMax-(1.+q)*X)*(q*aMax-(1.+q)*X)/q)
                    + (1.+q)*X*(Di(1.-aMax/((1.+q)*X)) - Di(q*aMax/((1.+q)*X)))
                    )

    functions["caseD"] = lambda X, q, aMax: (1.+q)/(4.*q*aMax**2)*(
                    -X*np.log(aMax)**2
                    + 2.*(1.+q)*(aMax-X)
                    + q*aMax*np.log(aMax/((1.+q)*X-q*aMax))
                    + aMax*np.log(q*aMax/(aMax-(1.+q)*X))
                    - X*np.log(aMax)*(2.*(1.+q) - np.log((1.+q)*X) - q*np.log((1.+q)*X/aMax))
                    + (1.+q)*X*np.log((-q*aMax+(1.+q)*X)*(aMax-(1.+q)*X)/q)
                    + (1.+q)*X*np.log(aMax/((1.+q)*X))*np.log((aMax-(1.+q)*X)/q)
                    + (1.+q)*X*(Di(1.-aMax/((1.+q)*X)) - Di(q*aMax/((1.+q)*X)))
                    )

    functions["caseE"] = lambda X, q, aMax: (1.+q)/(4.*q*aMax**2)*(
                    2.*(1.+q)*(aMax-X)
                    - (1.+q)*X*np.log(aMax)**2
                    + np.log(aMax)*(
                        aMax
                        -2.*(1.+q)*X
                        -(1.+q)*X*np.log(q/((1.+q)*X-aMax))
                        )
                    - aMax*np.log(((1.+q)*X-aMax)/q)
                    + (1.+q)*X*np.log(((1.+q)*X-aMax)*((1.+q)*X-q*aMax)/q)
                    + (1.+q)*X*np.log((1.+q)*X)*np.log(q*aMax/((1.+q)*X-aMax))
                    - q*aMax*np.log(((1.+q)*X-q*aMax)/aMax)
                    + (1.+q)*X*(Di(1.-aMax/((1.+q)*X)) - Di(q*aMax/((1.+q)*X)))
                    )

    functions["caseF"] = lambda X, q, aMax: 0.

    masks = dict()
    masks["caseZ"] = (xs==0)
    masks["caseA"] = (xs>0)*(xs<a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs<mass_ratio*a_max/(1.+mass_ratio))
    masks["caseB"] = (xs<a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs>mass_ratio*a_max/(1.+mass_ratio))
    masks["caseC"] = (xs>a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs<mass_ratio*a_max/(1.+mass_ratio))
    masks["caseD"] = (xs>a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs<a_max/(1.+mass_ratio))*(xs>=mass_ratio*a_max/(1.+mass_ratio))
    masks["caseE"] = (xs>a_max*(1.-mass_ratio)/(1.+mass_ratio))*(xs>a_max/(1.+mass_ratio))*(xs<a_max)
    masks["caseF"] = (xs>=a_max)

    for case in masks.keys():
        mask = masks[case]
        Xs = xs[mask]
        qs = mass_ratio[mask]
        amaxes = a_max[mask]
        pdfs[mask] = functions[case](X = Xs, q = qs, aMax = amaxes)

    # Deal with spins on the boundary between cases
    if np.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(mass_ratio = mass_ratio[boundary], a_max = a_max[boundary], xs = xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(mass_ratio = mass_ratio[boundary], a_max = a_max[boundary], xs = xs[boundary]+1e-6))

    return np.real(pdfs)

def chi_p_prior_from_isotropic_spins(xs, mass_ratio, a_max):

    """
    Function defining the conditional priors p(chi_p|q) corresponding to
    uniform, isotropic component spin priors.
    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_p value or values at which we wish to compute prior
    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = np.reshape(xs,-1)

    # Set up various piecewise cases
    pdfs = np.zeros(xs.size)

    masks = dict()
    masks["caseA"] = xs<mass_ratio*a_max*(3.+4.*mass_ratio)/(4.+3.*mass_ratio)
    masks["caseB"] = (xs>=mass_ratio*a_max*(3.+4.*mass_ratio)/(4.+3.*mass_ratio))*(xs<a_max)

    """
    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    """

    functions = dict()
    functions["caseA"] = lambda X, q, aMax: (1./(aMax**2*q))*((4.+3.*q)/(3.+4.*q))*(
                    np.arccos((4.+3.*q)*X/((3.+4.*q)*q*aMax))*(
                        aMax
                        - np.sqrt(aMax**2-X**2)
                        + X*np.arccos(X/aMax)
                        )
                    + np.arccos(X/aMax)*(
                        aMax*q*(3.+4.*q)/(4.+3.*q)
                        - np.sqrt(aMax**2*q**2*((3.+4.*q)/(4.+3.*q))**2 - X**2)
                        + X*np.arccos((4.+3.*q)*X/((3.+4.*q)*aMax*q))
                        )
                    )
                    
    functions["caseB"] = lambda X, q, aMax: (1./aMax)*np.arccos(X/aMax)

    for case in masks.keys():
        mask = masks[case]
        Xs = xs[mask]
        qs = mass_ratio[mask]
        amaxes = a_max[mask]
        pdfs[mask] = functions[case](X = Xs, q = qs, aMax = amaxes)
    return pdfs

def joint_prior_from_isotropic_spins(q,aMax,xeffs,xps,ndraws=10000,bw_method='scott'):

    """
    Function to calculate the conditional priors p(xp|xeff,q) on a set of {xp,xeff,q} posterior samples.
    INPUTS
    q: Mass ratio
    aMax: Maximimum spin magnitude considered
    xeffs: Effective inspiral spin samples
    xps: Effective precessing spin values
    ndraws: Number of draws from the component spin priors used in numerically building interpolant
    RETURNS
    p_chi_p: Array of priors on xp, conditioned on given effective inspiral spins and mass ratios
    """

    # Convert to arrays for safety
    xeffs = np.reshape(xeffs,-1)
    xps = np.reshape(xps,-1)
    
    # Compute marginal prior on xeff, conditional prior on xp, and multiply to get joint prior!
    p_chi_eff = chi_effective_prior_from_isotropic_spins(mass_ratio = q, a_max = aMax,xs = xeffs)
    p_chi_p_given_chi_eff = np.array([chi_p_prior_given_chi_eff_q(q,aMax,xeffs[i],xps[i],ndraws,bw_method) for i in range(len(xeffs))])
    joint_p_chi_p_chi_eff = p_chi_eff*p_chi_p_given_chi_eff

    return joint_p_chi_p_chi_eff

def chi_p_prior_given_chi_eff_q(q,aMax,xeff,xp,ndraws=10000,bw_method='scott'):

    """
    Function to calculate the conditional prior p(xp|xeff,q) on a single {xp,xeff,q} posterior sample.
    Called by `joint_prior_from_isotropic_spins`.
    INPUTS
    q: Single posterior mass ratio sample
    aMax: Maximimum spin magnitude considered
    xeff: Single effective inspiral spin sample
    xp: Single effective precessing spin value
    ndraws: Number of draws from the component spin priors used in numerically building interpolant
    RETURNS
    p_chi_p: Prior on xp, conditioned on given effective inspiral spin and mass ratio
    """

    # Draw random spin magnitudes.
    # Note that, given a fixed chi_eff, a1 can be no larger than (1+q)*chi_eff,
    # and a2 can be no larger than (1+q)*chi_eff/q
    a1 = np.random.random(ndraws)*aMax
    a2 = np.random.random(ndraws)*aMax

    # Draw random tilts for spin 2
    cost2 = 2.*np.random.random(ndraws)-1.

    # Finally, given our conditional value for chi_eff, we can solve for cost1
    # Note, though, that we still must require that the implied value of cost1 be *physical*
    cost1 = (xeff*(1.+q) - q*a2*cost2)/a1  

    # While any cost1 values remain unphysical, redraw a1, a2, and cost2, and recompute
    # Repeat as necessary
    while np.any(cost1<-1) or np.any(cost1>1):   
        to_replace = np.where((cost1<-1) | (cost1>1))[0]   
        a1[to_replace] = np.random.random(to_replace.size)*aMax
        a2[to_replace] = np.random.random(to_replace.size)*aMax
        cost2[to_replace] = 2.*np.random.random(to_replace.size)-1.    
        cost1 = (xeff*(1.+q) - q*a2*cost2)/a1   
            
    # Compute precessing spins and corresponding weights, build KDE
    # See `Joint-ChiEff-ChiP-Prior.ipynb` for a discussion of these weights
    Xp_draws = chi_p_from_components(a1,a2,cost1,cost2,q)
    jacobian_weights = (1.+q)/a1
    prior_kde = gaussian_kde(Xp_draws,weights=jacobian_weights,bw_method=bw_method)

    # Compute maximum chi_p
    if (1.+q)*np.abs(xeff)/q<aMax:
        max_Xp = aMax
    else:
        max_Xp = np.sqrt(aMax**2 - ((1.+q)*np.abs(xeff)-q)**2.)

    # Set up a grid slightly inside (0,max chi_p) and evaluate KDE
    reference_grid = np.linspace(0.05*max_Xp,0.95*max_Xp,50)
    reference_vals = prior_kde(reference_grid)

    # Manually prepend/append zeros at the boundaries
    reference_grid = np.concatenate([[0],reference_grid,[max_Xp]])
    reference_vals = np.concatenate([[0],reference_vals,[0]])
    norm_constant = np.trapz(reference_vals,reference_grid)

    # Interpolate!
    p_chi_p = np.interp(xp,reference_grid,reference_vals/norm_constant)
    return p_chi_p

def chi_p_from_components(a1,a2,cost1,cost2,q):

    """
    Helper function to define effective precessing spin parameter from component spins
    INPUTS
    a1: Primary dimensionless spin magnitude
    a2: Secondary's spin magnitude
    cost1: Cosine of the primary's spin-orbit tilt angle
    cost2: Cosine of the secondary's spin-orbit tilt
    q: Mass ratio
    RETRUNS
    chi_p: Corresponding precessing spin value
    """

    sint1 = np.sqrt(1.-cost1**2)
    sint2 = np.sqrt(1.-cost2**2)
    
    return np.maximum(a1*sint1,((3.+4.*q)/(4.+3.*q))*q*a2*sint2)

def decode_string_into_args_kwargs(expression):
    argumentslist = expression.split(',')
    argumentslist = [arg.strip() for arg in argumentslist]

    args = []
    kwargstr = ""
    for arg in argumentslist:
        if "=" in arg:
            kwargstr += arg + ","
        else:
            if arg.startswith("np."):
                arg_float = eval(arg)
            else:
                arg_float = float(arg)
            args.append(arg_float)
    kwargs = eval(f"dict({kwargstr})")

    return args, kwargs

def get_numpyro_dist_from_string(expression):
    funcstring, inputstring = expression.split("(", 1)
    distfunc = getattr(dist, funcstring.strip())
    argstring = inputstring.strip()[:-1]
    args, kwargs = decode_string_into_args_kwargs(argstring)

    return distfunc(*args, **kwargs)

def parse_input_line(line):
    variable, functionstr = line.split("=", 1)
    variable = variable.strip()

    try:
        if functionstr.startswith("np."):
            float_arg = eval(functionstr)
        else:
            float_arg = float(functionstr)
        return variable, float_arg

    except ValueError:
        pass
    function = get_numpyro_dist_from_string(functionstr)
    return variable, function

def get_priors_from_file(filename):
    prior = dict()
    with open(filename, "r") as priorfile:
        lines = priorfile.read().splitlines()
    for line in lines:
        param, func = parse_input_line(line)
        prior[param] = func
    return prior

def sample_parameters_from_dict(prior):
    samples = dict()
    for param in prior:
        if isinstance(prior[param], float):
            samples[param] = numpyro.deterministic(param, prior[param])
        else:
            samples[param] = numpyro.sample(param, prior[param])
    return samples

def log_expit(exponent):
    epsilon = 1e-8
    expres = jnp.exp(-exponent)
    result = -jnp.log1p(expres+epsilon)
    return result