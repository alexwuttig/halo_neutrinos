# Imports
import numpy as np
import scipy as sp
# for integration and interpolation
from scipy.integrate import quad

# My functions
from diffusion import diffusion_len
from config import params

from halo_neutrinos.conv_and_const_module import *

def H_one_zone(r, Ee, E0, rs=params.rs):
    '''
    One zone model found in various papers. 
    Pulling from: https://doi.org/10.1093/mnras/stz268

    - params -
    r: position in PARSECS
    Ee: current electron energy
    E0: injected electron energy
    rs: position of the source
    '''
    del_r_cm = np.abs(r - rs) * conv_pc_to_cm  # Convert to cm
    
    dl_cm = diffusion_len(Ee, E0, del_r_cm)  # Returns cm
    
    if dl_cm == 0:
        print(f"ERROR: diffusion length is zero! r={r}, Ee={Ee}, E0={E0}, dl={dl_cm} ")
        return 0
    
    term1 = 1/(np.pi**(3/2) * dl_cm**3)
    term2 = np.exp(-(del_r_cm**2)/(dl_cm**2))
    
    result = term1 * term2
    
    if np.isnan(result) or np.isinf(result):
        print(f"ERROR: H_one_zone = {result}, r={r}, Ee={Ee}, E0={E0}, dl={dl_cm}")
        return 0
    
    return result

def H_mauro(r, Ee, E0, rs = params.rs, rh = params.rh):
    '''
    H funciton as defined in Di Mauro et. al. paper
    Link: https://doi.org/10.1103/PhysRevD.100.123015

    - params - 
    r: position in parsecs
    Ee: current electron energy
    E0: injected electron energy
    rs: position of the source


    Note: In the H function there is the use of r. Unsure if this 
    is supposed to be del_r or just r. 
    '''

    # rs is the source position
    del_r = np.abs(r - rs) * conv_pc_to_cm

    # diffusion lengths
    l_0 = diffusion_len(Ee, E0,  0) # setting r to 0 pc so it uses D0
    l_ism = diffusion_len(Ee, E0, 100 * conv_pc_to_cm) # setting r to 100 pc so it uses Dism

    # Defining perameters in paper
    D0 = 1e26 # cm^2 / s
    Dism = 4e28 # cm^2 / s

    xi = np.sqrt(D0/Dism)

    # epsilon definition
    ep = rh/l_0
    erf = sp.special.erf(ep)
    erf2 = sp.special.erf(ep*2)
    erfc = sp.special.erfc(ep)

    # Implementing the function
    bot00 = (np.pi*(l_0**2))**(3/2)
    bot01 = 2*(xi**2)*erf
    bot02 = xi*(xi-1)*erf2
    bot03 = 2*erfc
    term0 = (xi*(xi+1)) / (bot00)*(bot01 - bot02 + bot03) 

    # r dependence
    if r < rh:
        term1 = np.exp( - (del_r**2) / (l_0**2) )
        term21 = (xi-1)/(xi + 1)
        term22 = (2*rh)/(r) - 1
        term23 = np.exp( - ((del_r-2*rh)**2) / (l_0**2) )
        term2 = term21*term22*term23

        result = term0*(term1 + term2)

    else:
        term01 = (2*xi)/(xi + 1)
        term02 = np.exp( - (((del_r - rh)/l_ism) + rh/l_0 )**2 )
        term1 = rh/r
        term2 = xi*(1-rh/r)

        result = term0 * term01 * term02 * (term1 + term2)

    return result


# Dont use this equation as is, but it exists as a possible improvement over mauro
def H_schroer(r, E, t):
    """
    Link to paper: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.123020 

    Compute H(r,E,t) by numerically integrating over psi from 0 to infinity.
    Uses piecewise form depending on r < r0 or r >= r0.
    """
    r0 = 50 #pc

    # lam = diffusion_len(E, t, 0)
    lam = 50 # pc
    
    # xi = np.sqrt(params.D0 / params.D1)
    xi = np.sqrt(1/100)
    
    def integrand(psi):
        # Avoid division by zero at psi=0
        if psi == 0:
            return 0.0
        chi = 2.0 * np.sqrt(psi) * r0 / lam
        # Compute A(psi) according to given formula
        A = (xi * np.cos(chi) * np.cos(xi * chi) +
             np.sin(chi) * np.sin(xi * chi) +
             (1.0 - xi) / chi * np.sin(chi) * np.cos(xi * chi))
        # Compute B(psi) as given
        B = (np.sin(chi) - A * np.sin(xi * chi)) / np.cos(xi * chi)
        # Prefactor common to both cases
        prefac = xi * np.exp(-psi) / (np.pi**2 * lam**2 * (A**2 + B**2))
        if r < r0:
            return prefac * (1.0/r) * np.sin(2.0 * np.sqrt(psi) * r / lam)
        else:
            return (prefac * (1.0/r) *
                    (A * np.sin(2.0 * np.sqrt(psi) * r * xi / lam) +
                     B * np.cos(2.0 * np.sqrt(psi) * r * xi / lam)))
    
    # Perform the integration from 0 to +inf (approximated by a finite upper limit or quad's inf handling)
    value, error = quad(integrand, 0, np.inf, limit = 100)
    return value
