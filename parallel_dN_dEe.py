# Imports
import numpy as np
import scipy as sp
# for integration and interpolation
from scipy.integrate import quad
from scipy.interpolate import interp1d
# for parrallelism
from multiprocessing import Pool

from constants_module import c, hbar_GeV, me, k, h
import b_loss as b

''' Import perameters from config.py '''
from config import params

''' conversions '''
conv_pc_to_cm = 3.086e+18       # conversion factor from [pc] to [cm]
conv_cm_to_pc = 3.24e-19        # conversion factor from [cm] to [pc]
conv_s_yr = 3.171e-8         # conversion factor from [sec] to [yr]
conv_yr_sec = 3.154e+7       # conversion factor from [yr] to [sec]
conv_rad_degree = 57.2958    # conversion factor from [rad] to [deg]
c = 2.998e+10                # speed of light, in [cm/s]

def Q_injected(Ee):
    ''' 
    Describes the source term for the transport equation describing the 
    electron motion throughout the TeV halo. 
    
    - params - 
    Ee: electron energy (GeV)
    Qnorm: normalization factor for the power law

    - returns -
    injected electron power law spectrum
    '''
    alpha = params.alpha
    Ecut = params.Ecut 

    return Ee**(-alpha) * np.exp(-Ee / Ecut)

def B_anti(Ex, Emin=me):    
    '''
    Computes the antiderivative of 1/btot at a given energy to then later
    solve for E0 (the intial electron energy) as a function of Ee and t.
    Utilizes logspace to get the integral to work out properly.

    - params - 
    Ex: some electron energy value to evaluate at
    Emin: minimum value for the integral, set to 0.1 GeV ( for now )

    - returns - 
    the antiderivative of 1/btot evaluated at Ex
    '''
    E_test = np.logspace(np.log10(me), np.log10(Ex), 50)

    def integrand_B(Ex):
        return 1 / b.b_tot(Ex)
    
    # Old one with points = E_test
    result, error = quad(integrand_B, Emin, Ex, points = E_test, 
                        epsabs=1e-12, epsrel=1e-10, limit = 1000)
    
    # result, error = quad(integrand_B, Emin, Ex,
    #                     epsabs=1e-12, epsrel=1e-10, limit = 1000)
    
    return result

def spline_to_E0s(Emin = me, Emax = 1e40):
    n_points = 200000

    Es = np.logspace(np.log10(Emin), np.log10(Emax), n_points)

    Bs_at_Es = []
    for i in range(n_points):
        Bs_at_Es.append(B_anti(Es[i]))

    inv_spline = interp1d(Bs_at_Es, Es, bounds_error=False, fill_value='extrapolate')

    return inv_spline

''' global variable '''
# spline for interpolating E0
E0_spline = spline_to_E0s()

def E0_val(Ee, tau):
    '''
    Computes the inital photon energy as a function of final
    photon energy and time difference between observation and 
    injection(tstar). Utilizes a inverse function spline to find
    the E0 that solves the necessary equation for a given Ee and
    tau.

    - params -
    Ee: final photon energy
    tau: tobs - tinj (Defined in the hooper paper)

    - returns - 
    value of E0 at given Ee and tau
    maximum value of Ee for wich the function is defined, above that 
    there isn't any well defined values
    '''
    value = B_anti(Ee) + tau
    return E0_spline(value)

def diffusion_hoop(Ee, r):
    ''' 
    Diffusion model from the hooper paper, uses a two zone diffusion model
    with a factor rh that describes where the model changes from one zone
    to the next. Spherical diffusion is required by the results of the 
    Luque et. al. paper. 

    - params -
    r: distance from center of pulsar halo
    '''
    # rh: radius when the diffusion model 'switches zones, set as 30 pc in hooper
    # adjusted for outsdie halo numbers
    rh = params.rh
    if r < rh: 
        # coefficient for r < rh
        D0 = 1e26 # cm^2 / s
        del_exponent = params.del_exponent 

        # (Ee/1 GeV) below is just 1 because we are already in GeV
        return D0*(Ee/1)**del_exponent
    elif r >= rh:
        Dism = 4e28 # cm^2 / s
        del_ism = 1/3

        return Dism*(Ee/1)**del_ism
    else:
        print(f'Error in reading in radius for diffusion')
        return 0 

def diffusion_len(E0, Ee, r):
    '''
    Uses the diffusion coefficient to find the diffusion length.

    - params - 
    Ee: final electron energy (GeV)
    E0: intial electron energy (GeV)

    - returns - 
    diffusion length (cm)
    '''
    def diffusion_integrand(Ee, r):
        return diffusion_hoop(Ee, r)/b.b_tot(Ee)

    '''
    Positron flux and γ-ray emission from Geminga pulsar and pulsar wind nebula
    --> mentions to use this for the bounds of integration
    '''
    E0_max = 1e9 #maximum energy electron that is produced by the halo (GeV)

    result = quad(diffusion_integrand, Ee, min(E0, E0_max), r)[0]

    if result < 0:
        print(f"diffu_len < 0: E0 = {E0}, Ee = {Ee}, r = {r}")

    return np.sqrt(4*result)

def H_one_zone(r, Ee, E0, rs = params.rs):
    '''
    One zone model found in Mauro et al. and various other papers. 
    This is what hooper uses.
    '''
    del_r = np.abs(r - rs)*conv_pc_to_cm

    dl = diffusion_len(E0, Ee, del_r)*conv_pc_to_cm

    term1 = 1/(np.pi**(3/2) * dl**3)
    term2 = np.exp(-(del_r**2)/(dl**2))

    result = term1*term2

    return result

def H(r, Ee, E0, rs = params.rs):
    '''
    Defined in Di Mauro et. al. paper
    
    - params - 
    r: position in galactic coordinates
    '''

    # rs is the source position
    del_r = np.abs(r - rs)

    # diffusion lengths
    l_0 = diffusion_len(E0, Ee, 0)*conv_pc_to_cm # setting r to 0 pc so it uses D0
    l_ism = diffusion_len(E0, Ee, 100)*conv_pc_to_cm # setting r to 100 pc so it uses Dism

    # Defining perameters in paper
    D0 = 1e26 # cm^2 / s
    Dism = 4e28 # cm^2 / s

    # rh: radius when the diffusion model 'switches zones, set as 30 pc in hooper
    rh = 30 # parsecs

    xi = np.sqrt(D0/Dism)

    # epsilon definition
    ep = rh/l_0
    erf = sp.special.erf(ep)
    erfc = sp.special.erfc(ep)

    # Implementing the function
    bot00 = (np.pi*(l_0**2))**(3/2)
    bot01 = 2*(xi**2)*erf
    bot02 = xi*(xi-1)*erf
    bot03 = 2*erfc
    term0 = (xi*(xi+1)) / (bot00)*(bot01 - bot02 + bot03) 

    # r dependence
    if r < rh:
        term1 = np.exp( - (del_r**2) / (l_0**2) )
        term21 = (xi-1)/(xi + 1)
        term22 = (2*rh)/(r) - 1
        term23 = np.exp( - ((del_r-2*rh)**2) / (l_0**2) )
        term2 = term21*term22*term23

        result = term0*(term1+term2)

    else:
        term01 = (2*xi)/(xi + 1)
        term02 = np.exp( - (((del_r - rh)/l_ism) + rh/l_0 )**2 )
        term1 = rh/r
        term2 = xi*(1-rh/r)

        result = term0*term01*term02*(term1 + term2)

    return result

def L_mauro(t):
    L0_unscaled = 1
    tau0 = params.tau0
    return L0_unscaled/((1 + t/tau0)**2)

def full_Ne_mauro(Ee, r, t, rs = params.rs):
    '''
    Combines everything from above to make a full electron spectrum 
    from the pulsar halo. 

    - params -
    Ee: electron energy
    r: position in galactic coordinates
    rs: location of source in galactic coordinates
    t: time (in seconds)

    - returns - 
    value at a specified Electron energy
    '''

    tobs = params.tobs
    tdiff = tobs - t

    E0 = E0_val(Ee, tdiff)

    # if E0 < 0 or E0 > 1e6:
    #     return 0

    b0 = b.b_tot(E0)
    be = b.b_tot(Ee)

    term_b = b0/be

    lum = L_mauro(t)

    term_Q = lum*Q_injected(E0)

    # TWO DIFFERENT THINGS
    result = term_b * term_Q * H_one_zone(r, Ee, E0)
    # result = term_b * term_Q * H(r, Ee, E0, rs = 1)

    return result

def dN_dEe_r(Ee, t):
    '''
    Calculates the differential number density at a given time for
    all of space.

    - params - 
    t: time to integrate to
    Ee: electron energy 
    r: position in galactic coordinates
    rs: position of source in galactic coordinates

    - returns - 
    differential number density at a given time
    '''
    r_min = 30
    r_max = 100
    rs = np.linspace(r_min, r_max, 12)
    
    def dN_integrand(r, args):
        return full_Ne_mauro(args[0], r, args[1], args[2])*4*np.pi*r**2

    # result, error = quad(dN_integrand, r_min, r_max, args = [Ee, t, 0], points = rs, limit = 50)

    # REMOVED POINTS
    result, error = quad(dN_integrand, r_min, r_max, args = [Ee, t, 0], limit = 50)

    return result

def full_dN_dEe(Ee, t_age):
    '''
    Full dN_dE over time and spaaaace.

    - params - 
    Ee: electron energy [GeV]
    t_age: age of the pulsar
    
    - returns - 
    total N of the pulsar as at a given Ee
    '''
    t_min = 0
    t_max = t_age
    ts = np.linspace(t_min, t_max, 12)

    def r_integrand(t, args):
        return dN_dEe_r(args[0], t)
    
    # result, error = quad(r_integrand, t_min, t_max, args = [Ee], points = ts, limit = 50)
    
    # REMOVED POINTS
    result, error = quad(r_integrand, t_min, t_max, args = [Ee], limit = 50)

    # IMPORTANT SCALING FACTOR THAT WAS TAKEN FROM L0 IN LUMINOSITY TERM
    L0 = (1+params.t_age/params.tau0)
    Edot = params.Edot 
    SCALING = L0*Edot

    return SCALING*result

def point_flux_dN_dEe(Ee, t_age):
    '''
    Calculates flux at a certain radius

    - params - 
    Ee: electron energy [GeV]
    t_age: age of the pulsar
    
    - returns - 
    total N of the pulsar as at a given Ee
    '''
    t_min = 0
    t_max = t_age
    ts = np.linspace(t_min, t_max, 12)

    d_mono = 288 # distance of monogem in parsecs
    
    def integrand(t, args):
        return full_Ne_mauro(args[0], args[1], t,  args[2])
    
    # result, error = quad(r_integrand, t_min, t_max, args = [Ee], points = ts, limit = 50)
    
    # REMOVED POINTS
    result, error = quad(integrand, t_min, t_max, args = [Ee, d_mono, 1], limit = 50)

    # IMPORTANT SCALING FACTOR THAT WAS TAKEN FROM L0 IN LUMINOSITY TERM
    L0 = (1+params.t_age/params.tau0)
    Edot = params.Edot 
    SCALING = L0*Edot

    return SCALING*result

# Parallelized version
def compute_single_energy(args):
    """Helper function for parallel computation"""
    Ee, t_age = args

    # CHOOSE WHAT TO CALCULATE
    # result = full_dN_dEe(Ee, t_age)
    result = point_flux_dN_dEe(Ee, t_age)

    return result

def full_dN_dEe_spec_parallel(t_age, n_processes=None):
    """
    Parallelized version using multiprocessing
    """
    n_points = 8
    Ees = np.logspace(np.log10(0.1), np.log10(1e6), n_points)
    
    # Prepare arguments for passing into pool
    args_list = [(Ee, t_age) for Ee in Ees]
    
    # Use all available cores if not specified
    if n_processes is None:
        n_processes = min(len(args_list), 8)  # Cap at 8 processes
    
    with Pool(n_processes) as pool:
        dNs = pool.map(compute_single_energy, args_list)
    
    return dNs, Ees

def main():
    
    dNs, Ees = full_dN_dEe_spec_parallel(params.t_age, n_processes = 8)

    # Save the data to a file
    np.savetxt("electron_spectrum_data.txt", np.column_stack((Ees, dNs)), 
               header="Ees (GeV)    dNs", fmt="%.6e")
    print("Data saved to 'electron_spectrum_data.txt'")
    

if __name__ == "__main__":
    main()