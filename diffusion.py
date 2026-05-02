# Imports
import numpy as np
# for integration and interpolation
from scipy.integrate import quad

''' import file functionality '''
from conv_and_const_module import *
from config import params
import b_loss as b
import energy_loss as E_loss

def L_mauro(t):
    '''
    Luminosity function from: Mauro et al
    https://iopscience.iop.org/article/10.3847/1538-4357/ad738e/pdf
    '''

    L0_val = L0()
    tau0 = params.tau0
    return (params.eta*L0_val)/((1 + t/tau0)**2)

def L0 ():
    '''
    Findining the normalization factor L0 from the spin down rate
    over time. 

    - returns - 
    L0 normalization factor in GeV
    '''
    # finding the spindown power based on the age of the pulsar(T)
    d_cm = conv_pc_to_cm * params.d
    T = params.tobs + d_cm/c
    tau0 = params.tau0
    Edot = params.Edot

    # Spin down power
    W0 = tau0 * Edot * ( 1 + T/tau0 )**2

    # Solving for L0
    Etot = W0 * params.eta

    def L_mauro_unscaled(t):
        tau0 = params.tau0
        return params.eta/((1 + t/tau0)**2)

    integrand, _ = quad(L_mauro_unscaled,0,T)

    L0_val = Etot / integrand

    return L0_val * conv_erg_to_GeV


def L_xaiping (t):
    '''
    Luminosity function from: Xaiping et al
    https://academic.oup.com/mnras/article/484/3/3491/5301417#update310119_update300119_update300119_equ15 
    '''
    Lt_val = params.eta * params.Edot * ( (1 + params.t_age/params.tau0)**2/((1 + t/params.tau0)**2) )
    return  Lt_val * conv_erg_to_GeV

def Q_injected(t, Ee):
    ''' 
    Describes the source term for the transport equation describing the 
    electron motion throughout the TeV halo. 

    Specific function descriped in Hooper et al and Mauro et al. 
    
    - params - 
    t: time of injection
    Ee: electron energy (GeV)

    - returns -
    injected electron power law spectrum
    '''
    alpha = params.alpha
    Ecut = params.Ecut 

    return L_mauro(t) * Ee**(-alpha) * np.exp(-Ee / Ecut)

def diffusion_len_approx(Ee, E0, r):
    '''
    Analytic approximation to the diffusion length using:
        b(E)  = b0 * E^2  (quadratic energy loss)
        D(E)  = D0 * E^(1/3)  (inner zone, r < rh)
    
    Result is in cm. Multiply by conv_cm_pc to get parsecs.

    - params -
    Ee:  observed electron energy (GeV)
    tau: tobs - tinj (s)
    r:   distance from pulsar (pc) — only affects which D0 is used
    '''
    b0 = 1.4e-16   # GeV^-1 s^-1

    if r < params.rh:
        D = params.D0
    else:
        D = params.Dism

    delta = params.del_exponent

    integral = (D/b0) * (1/(delta - 1)) * (E0**(delta - 1) - Ee**(delta  -1))

    return np.sqrt(4 * integral)

# ── Diffusion coefficient ──────────────────────────────────────────────────────
def D(Ee, r):
    """Two-zone diffusion coefficient(from Hooper)."""
    rh  = params.rh
    D0  = params.D0 # cm^2/s  (inner)
    Dism = params.Dism # cm^2/s  (outer)
    delta = params.del_exponent
    coeff = D0 if r < rh else Dism
    return coeff * (Ee / 1.0)**delta

# ── Diffusion length ───────────────────────────────────────────────────────────
def diffusion_len(Ee, E0, r):
    ''' 
    Function for diffusion length. General formula given a diffusion equation 
    D and an energy loss function b.
    '''

    # Checks if E0 value is nonphysical, sets to 0 if so
    if E0 <= Ee:
        return 0.0

    pts = np.logspace(np.log10(Ee), np.log10(E0), 300)
    result, _ = quad((lambda E: D(E, r) / b.b_tot(E)) , Ee, E0,
                     points=pts, epsabs=1e-12, epsrel=1e-10, limit=500)

    if np.sqrt(4 * result) > 0:
        diff_len = np.sqrt(4 * result)
    else:
        print(f"diffusion length set to 0")
        diff_len = 0

    return diff_len


def main():
    L0_result = L_mauro(0)
    print(f"L(t = 0) value for Mauro is: {L0_result}")

    L0_result1 = L_xaiping(0)
    print(f"L(t = 0) value for Xaiping is: {L0_result1}")

if __name__ == "__main__":
    main()