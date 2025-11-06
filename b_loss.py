# Imports
import numpy as np

from constants_module import alpha, c, hbar_GeV, me, k, h

def rest_sig_T():
    
    r0 = 2.82e-13 # cm
    sig = (8/3)*np.pi*(r0**2)

    return sig

def b_sync(Ee):
    ''' 
    Energy losses due to synchrotron radiation.

    - perams - 
    Ee: electron energy (GeV)

    - returns - 
    energy loss due to synchrotron radiation
    '''
    sig_T = rest_sig_T() # cm^2
    B = 3e-6 # Gauss

    u_B = 0.22*(1e-9) # eV/cm

    #u_B = B**2 / (8 * np.pi)  # energy density of magnetic field in Gauss^2
    b_syn = (4/3) * sig_T * c * u_B * (Ee / me)**2
    
    return b_syn

def Si(Ee, Ti):
    ''' 
    Defines the suppression factor in the KN regime. 
    Approximate result from arXiv:2010.13825. 
    '''
    # boltzman constant in GeV
    k_GeV = 8.617*10**(-14)

    Ai = (45 * me**2)/(64 * (np.pi**2) * k_GeV * Ti**2)

    return Ai/(Ai + (Ee/me)**2)

def b_ics_i(Ee, ui, T):
    '''
    Inverse comton energy losses.

    - perams - 
    Ee: Electron energy
    ui: energy density
    T: temperature of blackbody

    - returns -
    losses due to inverse compton scattering in given regime
    '''

    sig_T = rest_sig_T() # cm^2

    top = 4*sig_T*ui*Si(Ee, T) * c # --> Likely right but I am not sure
    bot = 3 # *(c**3) # --> USURE ABOUT THIS, couldn't find it in other sources

    return (top/bot)*(Ee/me)**2

def b_tot(Ee):
    ''' 
    Combines all ICS component losses:
        CMB: u = 0.26 eV/cm^3, T = 2.75 K
        IR: u = 0.60 eV/cm^3, T = 20 K
        Starlight: u = 0.60 eV/cm^3, T = 5000 K
    '''
    #losses due to cmb photons
    u_cmb = 0.26*(1e-9) # GeV/cm^3
    T_cmb = 2.75 # K
    b_cmb = b_ics_i(Ee, u_cmb, T_cmb)

    # losses due to ir photons
    u_ir = 0.60*(1e-9) # GeV/cm^3
    T_ir = 20 # K
    b_ir = b_ics_i(Ee, u_ir, T_ir)

    # losses due to starlight photons
    u_star = 0.60*(1e-9) # GeV/cm^3
    T_star = 5000 # K
    b_star = b_ics_i(Ee, u_star, T_star)

    # total inverse compton scattering losses
    b_ics =  b_cmb + b_ir + b_star

    #losses from synchrotron radiation
    b_syn = b_sync(Ee)

    # total energy loss
    b_tot = b_ics + b_syn

    return b_tot
    