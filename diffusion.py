# Imports
import numpy as np
# for integration and interpolation
from scipy.integrate import quad

''' import file functionality '''
from conv_and_const_module import *
from config import params
import b_loss as b

def L_mauro(t):
    tau0 = params.tau0
    return params.eta/((1 + t/tau0)**2)


def L0 ():
    '''
    Findining the normalization factor L0 from the spin down rate
    over time. 

    '''
    # finding the spindown power based on the age of the pulsar(T)
    d_cm = conv_pc_to_cm * params.d
    T = params.tobs + d_cm/c
    tau0 = params.tau0
    Edot = params.Edot
    W0 = tau0 * Edot * ( 1 + T/tau0 )**2

    # Solving for L0
    Etot = W0 * params.eta

    L0_val = Etot / quad(L_mauro,0,T)

    return L0_val


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

def diffusion_hoop(Ee, r_cm):
    ''' 
    Diffusion coefficient
    - params -
    r_cm: distance from center of pulsar halo IN CM
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
    L0 = L0()
    print("In main!")
    print(f"L0 value is: {L0}")



if __name__ == "__main__":
    main()