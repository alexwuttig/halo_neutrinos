# Imports
import numpy as np
import scipy as sp
# for integration and interpolation
from scipy.integrate import quad
import matplotlib.pyplot as plt
# for parrallelism
from multiprocessing import Pool

from conv_and_const_module import *
import b_loss as b
from config import params
import energy_loss as E_loss
import diffusion
import H_functions as H_funcs


def full_Ne_mauro(Ee, r, t, rs=params.rs):
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







    ''' CHECK THIS VALUE!!!!!!!! CHECK SIGN!!!!!'''
    tau = params.tobs - t







    # get E0 for current Ee and t
    E0 = E_loss.get_E0(Ee, tau)
    if np.isinf(E0) or np.isnan(E0):
        return 0  # no physical solution exists for this (Ee, tau)

    b0 = b.b_tot(E0)
    be = b.b_tot(Ee)

    if be == 0 or np.isnan(be):
        return 0

    term_b = b0/be

    term_Q = diffusion.Q_injected(t, E0)

    H_val = H_funcs.H_one_zone(r, Ee, E0)

    result = term_b * term_Q * H_val
    return result

def dN_dEe_r(Ee, t):
    '''Calculates the differential number density at a given time'''
    r_min = 0
    r_max = 30
    
    def dN_integrand(r, args):
        Ee, t = args
        return full_Ne_mauro(Ee, r, t) * 4 * np.pi * r**2

    result, error = quad(dN_integrand, r_min, r_max, args=[Ee, t], limit=50)
    return result

def point_flux_dN_dEe(Ee):
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
    t_max = min(params.t_age, params.tobs) # CHECK IF THIS IS PHYSICAL --> 

    if t_max <= t_min:
        print(f"ERROR: tobs ({params.tobs}) <= 0")
        return 0
    
    def integrand(t, args):
        Ee, d = args
        return full_Ne_mauro(Ee, d, t)
    
    result, _ = quad(integrand, t_min, t_max, args=[Ee, params.d], limit=50)

    return result

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

    n_points = 40
    Ees = np.logspace(np.log10(0.1), np.log10(1e6), n_points)
    dNs_E_sq = []
    for i in range(len(Ees)):
        dNs_E_sq.append(point_flux_dN_dEe(Ees[i]) * Ees[i]**3)
        

    plt.figure()
    plt.loglog(Ees, dNs_E_sq, marker='o', markersize = 2)
    plt.xlabel("Ee (GeV)")
    plt.ylabel("dN/dEe * Ee^3")
    plt.title("Point Flux Spectrum")
    plt.grid(True, which="both", ls="--")

    plt.savefig("spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()

    np.savetxt("spectrum.txt", np.column_stack((Ees, dNs_E_sq)),
           header="Ee dN/dEe")



    # SOMETHING TO DO!!!

    # GET TO THE POINT WHERE NONE OF THE NUMBERS ARE A SURPRISE TO ME
    # FLUX IS AROUND 10^(-6) for 5 PARSECS AWAY FOR A GIVEN TIME FOR NO ENERGY LOSSES
    # AT 1 TeV electron

if __name__ == "__main__":
    main()