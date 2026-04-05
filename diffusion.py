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





''' - - - - - - - - - - - - - - - - - - - - - - [ Checkpoint ] - - - - - - - - - - - - - - - - - - - - - - - - - '''


def diffusion_len_approx(Ee, tau, r=0):
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
    rh = 30        # pc

    if r < rh:
        D0 = 1e26  # cm^2/s
    else:
        D0 = 4e28  # cm^2/s

    # Analytic E0
    B_ee = (1/me - 1/Ee) / b0
    inv_E0 = 1/me - b0 * (B_ee + tau)
    
    if inv_E0 <= 0:
        return 0.0  # electron couldn't have reached Ee in time tau

    E0 = 1.0 / inv_E0

    # Analytic diffusion length integral
    integral = (3 * D0) / (2 * b0) * (Ee**(-2/3) - E0**(-2/3))

    if integral <= 0:
        return 0.0

    return np.sqrt(4 * integral)


# ── Diffusion coefficient ──────────────────────────────────────────────────────
def D(Ee, r):
    """Two-zone diffusion coefficient(from Hooper)."""
    rh  = 30    # pc
    D0  = 1e26  # cm^2/s  (inner)
    Dism = 4e28 # cm^2/s  (outer)
    delta = 1/3 
    coeff = D0 if r < rh else Dism
    return coeff * (Ee / 1.0)**delta

# ── Diffusion length ───────────────────────────────────────────────────────────
def diffusion_len(Ee, tau, r=0):
    E0 = E_loss.E0_val(Ee, tau)

    # Checks if E0 value is nonphysical, sets to 0 if so
    if E0 <= Ee:
        return 0.0

    pts = np.logspace(np.log10(Ee), np.log10(E0), 300)
    result, _ = quad((lambda E: D(E, r) / b.b_tot(E)) , Ee, E0,
                     points=pts, epsabs=1e-12, epsrel=1e-10, limit=500)

    return np.sqrt(4 * result) if result > 0 else 0.0


''' - - - - - - - - - - - - - - - - - - - - - - [ Old code below ] - - - - - - - - - - - - - - - - - - - - - - - - - '''

def diffusion_hoop(Ee, r_cm):
    ''' 
    Diffusion coefficient
    - params -
    r_cm: distance from center of pulsar halo (IN CM)
    '''
    rh_cm = params.rh  # Should already be in cm from params
    
    if r_cm < 0:
        print(f"ERROR: negative radius {r_cm}")
        return 0
    
    if r_cm < rh_cm: 
        D0 = 1e26
        del_exponent = params.del_exponent 
        return D0*(Ee)**del_exponent
    else:
        Dism = 4e28
        del_ism = 1/3
        return Dism*(Ee)**del_ism

def diffusion_len(Ee, E0, r_cm):
    '''
    Diffusion model from the hooper paper, uses a two zone diffusion model
    with a factor rh that describes where the model changes from one zone
    to the next. Spherical diffusion is required by the results of the 
    Luque et. al. paper. 

    Diffusion length λ(Ee, E0, r):
        λ^2 = 4 ∫_{Ee}^{E0} [ D(E) / b(E) ] dE

    - params - 
    Ee: final electron energy (GeV)
    E0: intial electron energy (GeV)

    - returns - 
    diffusion length (cm)
    '''
    E0_max = 1e9
    E_upper = min(E0, E0_max)

    if E_upper <= Ee:
        print(f"WARNING: E_upper ({E_upper}) <= Ee ({Ee}), returning minimum diffusion length")
        return 1e10  # 1e10 cm = ~0.3 pc, reasonable minimum
    
    if abs(E_upper - Ee) / Ee < 1e-6:
        print(f"WARNING: E_upper ≈ Ee, returning minimum diffusion length")
        return 1e10
    
    def diffusion_integrand(E, r_cm):
        D_val = diffusion_hoop(E, r_cm)
        b_val = b.b_tot(E)
        if b_val == 0 or np.isnan(b_val):
            print(f"ERROR: b_tot({E}) = {b_val}")
            return 0
        return D_val / b_val
    
    try:
        result = quad(diffusion_integrand, Ee, E_upper, args=(r_cm,))[0]
    except Exception as e:
        print(f"ERROR in diffusion_len quad: {e}")
        return 1e10
    
    if result <= 0:
        print(f"WARNING: diffusion integral = {result} <= 0, Ee={Ee}, E0={E0}, r={r_cm/conv_pc_to_cm:.2f} pc")
        return 1e10
    
    if np.isnan(result) or np.isinf(result):
        print(f"ERROR: diffusion integral = {result}")
        return 1e10
    
    return np.sqrt(4*result)



def main():
    L0_result = L_mauro(0)
    print(f"L(t = 0) value for Mauro is: {L0_result}")

    L0_result1 = L_xaiping(0)
    print(f"L(t = 0) value for Xaiping is: {L0_result1}")

if __name__ == "__main__":
    main()