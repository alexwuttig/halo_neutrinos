# Imports
import numpy as np
import scipy as sp
# for integration and interpolation
from scipy.integrate import quad
from scipy.interpolate import interp1d
# for parrallelism
from multiprocessing import Pool

from halo_neutrinos.conv_and_const_module import c, hbar_GeV, me, k, h
import b_loss as b
from config import params
from energy_loss import E0Calculator

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

def H_one_zone(r, Ee, E0, rs=params.rs):
    '''
    One zone model
    - params -
    r: position in PARSECS
    '''
    del_r_cm = np.abs(r - rs) * conv_pc_to_cm  # Convert to cm
    
    dl_cm = diffusion_len(Ee, E0, del_r_cm)  # Returns cm
    
    if dl_cm == 0:
        print(f"ERROR: diffusion length is zero!")
        return 0
    
    term1 = 1/(np.pi**(3/2) * dl_cm**3)
    term2 = np.exp(-(del_r_cm**2)/(dl_cm**2))
    
    result = term1 * term2
    
    if np.isnan(result) or np.isinf(result):
        print(f"ERROR: H_one_zone = {result}, r={r}, Ee={Ee}, E0={E0}, dl={dl_cm}")
        return 0
    
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
    l_0 = diffusion_len(Ee, E0,  0) # setting r to 0 pc so it uses D0
    l_ism = diffusion_len(Ee, E0, 100) # setting r to 100 pc so it uses Dism

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


def full_Ne_mauro(Ee, r, t, e0_calc, rs=params.rs):
    '''
    Combines everything from above to make a full electron spectrum 
    from the pulsar halo. 

    - params -
    Ee: electron energy
    r: position in galactic coordinates
    t: time (in seconds)
    e0_calc: E0Calculator instance
    rs: location of source in galactic coordinates

    - returns - 
    value at a specified Electron energy
    '''
    tdiff = params.tobs - t

    E0 = e0_calc.get_E0(Ee, tdiff)
    if np.isnan(E0):
        return 0

    b0 = b.b_tot(E0)
    be = b.b_tot(Ee)

    if be == 0 or np.isnan(be):
        return 0

    term_b = b0/be
    lum = L_mauro(t)
    term_Q = lum*Q_injected(E0)
    H_val = H_one_zone(r, Ee, E0)

    result = term_b * term_Q * H_val
    return result

def dN_dEe_r(Ee, t, e0_calc):
    '''Calculates the differential number density at a given time'''
    r_min = 0
    r_max = 100
    
    def dN_integrand(r, args):
        Ee, t, rs, calc = args
        return full_Ne_mauro(Ee, r, t, calc, rs) * 4 * np.pi * r**2

    result, error = quad(dN_integrand, r_min, r_max, args=[Ee, t, 0, e0_calc], limit=50)
    return result

def point_flux_dN_dEe(Ee, t_age, e0_calc):
    '''
    Calculates flux at a certain radius

    - params - 
    Ee: electron energy [GeV]
    t_age: age of the pulsar
    e0_calc: E0Calculator instance
    
    - returns - 
    total N of the pulsar as at a given Ee
    '''
    # Integration limits: from injection at t=0 to observation at tobs
    t_min = 0
    t_max = t_age

    if t_max <= t_min:
        print(f"ERROR: tobs ({params.tobs}) <= 0")
        return 0

    d_mono = 288  # distance of monogem in parsecs
    
    def integrand(t, args):
        Ee, d, rs, calc = args
        return full_Ne_mauro(Ee, d, t, calc, rs)
    
    result, error = quad(integrand, t_min, t_max, args=[Ee, d_mono, 1, e0_calc], limit=50)

    L0 = (1 + params.t_age/params.tau0)
    Edot = params.Edot 
    SCALING = L0 * Edot

    return SCALING * result

def compute_single_energy(args):
    """Helper function for parallel computation"""
    Ee, t_age, e0_calc = args
    result = point_flux_dN_dEe(Ee, t_age, e0_calc)
    return result

def full_dN_dEe_spec_parallel(t_age, e0_calc, n_processes=None):
    """Parallelized version using multiprocessing"""
    n_points = 8
    Ees = np.logspace(np.log10(0.1), np.log10(1e6), n_points)
    
    # Include calculator in each argument tuple
    args_list = [(Ee, t_age, e0_calc) for Ee in Ees]
    
    if n_processes is None:
        n_processes = min(len(args_list), 8)
    
    with Pool(n_processes) as pool:
        dNs = pool.map(compute_single_energy, args_list)
    
    return dNs, Ees

def main():
    print(f"Pulsar age: {params.t_age:.3e} s")
    print(f"Observation time: {params.tobs:.3e} s")
    print(f"tau0: {params.tau0:.3e} s")
    print()
    # Create calculator once
    e0_calc = E0Calculator(Emin=me, Emax=1e100, n_points=10000)


    n_points = 3
    Ees = np.logspace(np.log10(0.1), np.log10(1e6), n_points)
    dNs = []
    for i in range(len(Ees)):
        t_test = params.t_age/2
        d_test = 5  # distance of monogem in parsecs
        
        result = full_Ne_mauro(Ees[i], d_test, t_test, e0_calc)

        L0 = (1 + params.t_age/params.tau0)
        Edot = params.Edot 
        SCALING = L0 * Edot

        print(f"result is: {result}")
        print(f"scaling is: {SCALING}")
        

        dNs.append(result*SCALING)


    # SOMETHING TO DO!!!

    # GET TO THE POINT WHERE NONE OF THE NUMBERS ARE A SURPRISE TO ME
    # FLUX IS AROUND 10^(-6) for 5 PARSECS AWAY FOR A GIVEN TIME FOR NO ENERGY LOSSES
    # AT 1 TeV electron

    print(Ees)
    print(dNs)

    # # Pass it to the parallel function
    # dNs, Ees = full_dN_dEe_spec_parallel(params.t_age, e0_calc, n_processes=8)

    # np.savetxt("electron_spectrum_data.txt", np.column_stack((Ees, dNs)), 
    #            header="Ees (GeV)    dNs", fmt="%.6e")
    # print("Data saved to 'electron_spectrum_data.txt'")

if __name__ == "__main__":
    main()




# def full_Ne_mauro(Ee, r, t, rs = params.rs):
#     '''
#     Combines everything from above to make a full electron spectrum 
#     from the pulsar halo. 

#     - params -
#     Ee: electron energy
#     r: position in galactic coordinates
#     rs: location of source in galactic coordinates
#     t: time (in seconds)

#     - returns - 
#     value at a specified Electron energy
#     '''

#     tobs = params.tobs
#     tdiff = tobs - t

#     E0 = E0_val(Ee, tdiff)
#     if np.isnan(E0):
#         print(f"NaN in E0: Ee={Ee}, t={t}, tdiff={tdiff}")
#         return 0

#     b0 = b.b_tot(E0)
#     be = b.b_tot(Ee)

#     if be == 0 or np.isnan(be):
#         print(f"Invalid be: {be}")
#         return 0

#     term_b = b0/be
#     lum = L_mauro(t)
#     term_Q = lum*Q_injected(E0)

#     # TWO DIFFERENT MODELS
#         # one zone model
#     H_val = H_one_zone(r, Ee, E0)
#         # two zone model
#     # H_val = H(r, Ee, E0, rs = 1)

#     result = term_b * term_Q * H_val

#     if np.isnan(result):
#         print(f"NaN result: Ee={Ee}, r={r}, t={t}, E0={E0}, H={H_val}")

#     return result

# def dN_dEe_r(Ee, t):
#     '''
#     Calculates the differential number density at a given time for
#     all of space.

#     - params - 
#     t: time to integrate to
#     Ee: electron energy 
#     r: position in galactic coordinates
#     rs: position of source in galactic coordinates

#     - returns - 
#     differential number density at a given time
#     '''
#     r_min = 0
#     r_max = 100
#     rs = np.linspace(r_min, r_max, 12)
    
#     def dN_integrand(r, args):
#         return full_Ne_mauro(args[0], r, args[1], args[2])*4*np.pi*r**2

#     # result, error = quad(dN_integrand, r_min, r_max, args = [Ee, t, 0], points = rs, limit = 50)

#     # REMOVED POINTS
#     result, error = quad(dN_integrand, r_min, r_max, args = [Ee, t, 0], limit = 50)

#     return result

# def full_dN_dEe(Ee, t_age):
#     '''
#     Full dN_dE over time and spaaaace.

#     - params - 
#     Ee: electron energy [GeV]
#     t_age: age of the pulsar
    
#     - returns - 
#     total N of the pulsar as at a given Ee
#     '''
#     t_min = 0
#     t_max = t_age
#     ts = np.linspace(t_min, t_max, 12)

#     def r_integrand(t, args):
#         return dN_dEe_r(args[0], t)
    
#     # result, error = quad(r_integrand, t_min, t_max, args = [Ee], points = ts, limit = 50)
    
#     # REMOVED POINTS
#     result, error = quad(r_integrand, t_min, t_max, args = [Ee], limit = 50)

#     # IMPORTANT SCALING FACTOR THAT WAS TAKEN FROM L0 IN LUMINOSITY TERM
#     L0 = (1+params.t_age/params.tau0)
#     Edot = params.Edot 
#     SCALING = L0*Edot

#     return SCALING*result

# def point_flux_dN_dEe(Ee, t_age):
#     '''
#     Calculates flux at a certain radius

#     - params - 
#     Ee: electron energy [GeV]
#     t_age: age of the pulsar
    
#     - returns - 
#     total N of the pulsar as at a given Ee
#     '''
#     t_min = 0
#     t_max = t_age
#     ts = np.linspace(t_min, t_max, 12)

#     d_mono = 288 # distance of monogem in parsecs
    
#     def integrand(t, args):
#         return full_Ne_mauro(args[0], args[1], t,  args[2])
    
#     # result, error = quad(r_integrand, t_min, t_max, args = [Ee], points = ts, limit = 50)
    
#     # REMOVED POINTS
#     result, error = quad(integrand, t_min, t_max, args = [Ee, d_mono, 1], limit = 50)

#     # IMPORTANT SCALING FACTOR THAT WAS TAKEN FROM L0 IN LUMINOSITY TERM
#     L0 = (1+params.t_age/params.tau0)
#     Edot = params.Edot 
#     SCALING = L0*Edot

#     return SCALING*result

# # Parallelized version
# def compute_single_energy(args):
#     """Helper function for parallel computation"""
#     Ee, t_age = args

#     # CHOOSE WHAT TO CALCULATE
#     # result = full_dN_dEe(Ee, t_age)
#     result = point_flux_dN_dEe(Ee, t_age)

#     return result

# def full_dN_dEe_spec_parallel(t_age, n_processes=None):
#     """
#     Parallelized version using multiprocessing
#     """
#     n_points = 8
#     Ees = np.logspace(np.log10(0.1), np.log10(1e6), n_points)
    
#     # Prepare arguments for passing into pool
#     args_list = [(Ee, t_age) for Ee in Ees]
    
#     # Use all available cores if not specified
#     if n_processes is None:
#         n_processes = min(len(args_list), 8)  # Cap at 8 processes
    
#     with Pool(n_processes) as pool:
#         dNs = pool.map(compute_single_energy, args_list)
    
#     return dNs, Ees


# # def test_spline():
# #     """Test if E0 spline is working correctly"""
# #     print("Testing E0 spline...")
# #     test_Ees = [0.1, 1, 10, 100, 1000]
# #     test_taus = [1e10, 1e12, 1e14]
    
# #     for Ee in test_Ees:
# #         for tau in test_taus:
# #             E0 = E0_val(Ee, tau)
# #             print(f"Ee={Ee:>6.1f} GeV, tau={tau:.1e} s -> E0={E0:.3e} GeV")
# #             if np.isnan(E0):
# #                 print("  ^^^ NaN detected!")

# # # Call this before main()
# # if __name__ == "__main__":
# #     test_spline()
# #     # main()


# def main():
#     # Calculate E0 values
#     print("Initializing E0 calculator...")
#     initialize_E0_calculator(Emin=me, Emax=1e100, n_points=10000)
#     print("E0 calculator ready!")
    
#     # Actual computation
#     dNs, Ees = full_dN_dEe_spec_parallel(params.t_age, n_processes = 1)

#     # Save the data to a file
#     np.savetxt("electron_spectrum_data.txt", np.column_stack((Ees, dNs)), 
#                header="Ees (GeV)    dNs", fmt="%.6e")
#     print("Data saved to 'electron_spectrum_data.txt'")
    

# if __name__ == "__main__":
#     main()