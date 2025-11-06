import matplotlib.pyplot as plt
import numpy as np
import random

from constants_module import alpha, c, hbar_GeV, me, k, h
import Lorentz as lz
import PlotHelp as ph

from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp1d

def dSig_dE(Ei_GeV, Ef_GeV):
    ''' 
    Computes the differential cross section for compton scattering for a given incident 
    photon energy and final energy. All in the electron rest frame

    - arguments -
    Ei_GeV: Initial photon energy
    Ef_GeV: Final photon energy
    
    - returns - 
        sigE: differential cross section
    '''

    # Photon energies in GeV
    Ei = Ei_GeV
    Ef = Ef_GeV

    # mass in kg
    m = 0.511*10**(-3) # GeV / c**2

    # c is in terms of cm/s
    prefactor = (((hbar_GeV*c)**2)*np.pi*alpha**2)/(m*Ei**2)

    # differential cross section 
    sigE = prefactor*( (Ef/Ei) + (Ei/Ef) + 2*m*(1/Ei + 1/Ef) + (m**2)*(1/Ei + 1/Ef)**2 )

    return sigE

def angle_scatter(Ei_rest, Ef_rest):
    ''' 
    Returns the scattering angle in electron rest frame from the 
    initial and final energy in the electyron rest frame. 
    '''
    Ef_min, Ef_max = E_bounds(Ei_rest)

    if Ef_rest < Ef_min:
        Ef_rest = Ef_min + 1e-18

    if Ef_rest > Ef_max:
        Ef_rest = Ef_max - 1e-18

    return np.arccos((me/Ei_rest)*(1 - (Ei_rest/Ef_rest)) + 1)

def angle_finalRest(As_r, Ai_r):
    '''
    Calculates the final scattering angle in the electron rest frame.
    Takes in the rest initial and scattering angles
    '''
    thea_tot =  Ai_r + As_r
    return thea_tot

def E_bounds(Ei):
    '''
    Calculates the energy bounds for the outgoing energy photon post 
    scattering. 
    '''
    Ef_min = Ei/(1+2*(Ei/(me)))
    Ef_max = Ei
    return Ef_min, Ef_max

def Ef_rest_mc(Ei):
    ''' 
    Chooses a single final energy given a photon energy in the electron
    rest frame. Does this by recreating the cross section at a given energy
    then choosing a random point from that distribution.

    - perams -
    Ei: photon energy in electron rest frame

    - returns - 
    random final photon energy after scattering based for given Ei
    '''
    Emin, Emax = E_bounds(Ei)
    n_points = 100

    Ef = np.logspace(np.log10(Emin), np.log10(Emax), n_points)
    
    pdf_values = dSig_dE(Ei, Ef)
    
    # okay to use trapazoidal rule here, doesn't improve much at all with quad and
    # is much slower
    cdf_values = cumtrapz(pdf_values, Ef, initial=0)
    
    # Normalize the CDF
    cdf_values = cdf_values / cdf_values[-1]
        
    # Create inverse CDF interpolation (flipped cdf_values and Ef to get inverse)
    inverse_cdf = interp1d(cdf_values, Ef, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')

    # Generate random final energy
    u = np.random.uniform(0, 1)
    Ef_rest = inverse_cdf(u)
    
    return Ef_rest

def Ef_rest_mc_slow(Ei):
    ''' 
    Slower version of the above function that uses quad instead of the 
    trapezoidal rule. At least an order of magnitude slower. 
    '''
    Emin, Emax = E_bounds(Ei)
    n_points = 1000

    Ef = np.logspace(np.log10(Emin), np.log10(Emax), n_points)
    
    # Create interpolation function
    pdf = interp1d(Ef, dSig_dE(Ei, Ef), kind='cubic')
    
    cdf_values = []
    for E in Ef:
        integral_val, error = quad(pdf, min(Ef), E)
        cdf_values.append(integral_val)

    # Normalize the CDF
    cdf_values = [x / cdf_values[-1] for x in cdf_values]
        
    inverse_cdf = interp1d(cdf_values, Ef, kind='cubic', bounds_error=False, fill_value='extrapolate')

    # Generate random samples
    u = np.random.uniform(0, 1)
    Ef_rest = inverse_cdf(u)
    
    return Ef_rest


def pseudo_iso_mc(Ei_lab, gamma):
    ''' 
    Main function of this file. Takes in an inital photon energy and electron energy(via gamma)
    and computes a single final photon energy of the spectrum. Makes the approximation that 
    gamma is much greater than 1 in order to assume relatavistic beaming reduces the angular 
    dependence between the inital photon and electron. 

    - perams - 
    Ei_lab: photon energy in the lab frame
    gamma: gamma from the incident electron

    - returns - 
    final energy of photon in th elab frame
    
    '''
    # calculate beta from gamma
    beta = lz.gammaToBeta(gamma)
    # Generate random incident angle

    # APPROXIMATION THAT GAMMA IS HIGH (not random angle but beaming makes collision head on
    # in the electron rest frame)
    Ai_l = np.pi #lz.rand_theta()

    Ai_r = lz.angle_labToRest(Ai_l, beta, gamma)

    # Use intial angle, initial photon energy, and beta to translate into electron rest frame
    Ei_rest = lz.labToRest(Ei_lab, beta, gamma, Ai_l)

    # Using incident energy in rest frame, generate a final energy in rest frame via M.C.
    Ef_rest = Ef_rest_mc(Ei_rest)

    # Find the final angle for the transform from Angle in lab + pi - scatter
    As_r = angle_scatter(Ei_rest, Ef_rest)
    Af_r = angle_finalRest(As_r, Ai_r)
    
    # Useing the Final energy in the electron rest frame generate final energy in lab frame
    Ef_lab = lz.restToLab(Ef_rest, Af_r, beta, gamma)

    return Ef_lab

''' BELOW IS FOR SCALING THE MC BY TOTAL CROSS SECTIONAL AREA '''

def rest_sig_KN(Ei):
    ''' 
    Rest frame TOTAL cross sectional area for a given Ei.
    '''
    E = (Ei)/(me)
    r0 = 2.82e-13 # cm
    pre = 2*np.pi*(r0**2)

    term1_1 = (1 + E)/(E**3)
    term1_2 = (2*E*(1 + E))/(1 + 2*E) - np.log(1+2*E)
    term1 = term1_1 * term1_2

    term2 = (np.log(1 + 2*E))/(2*E)

    term3 = (1 + 3*E)/((1 + 2*E)**2)

    sig = pre *(term1 + term2 - term3)

    return sig

def rest_sig_T():
    '''
    Thompson cross section. 
    '''
    
    r0 = 2.82e-13 # cm
    sig = (8/3)*np.pi*(r0**2)

    return sig

def IC_scaled_mc(Ei_lab, gamma):
    ''' 
    Same as pseudo_iso_mc but includes a scaling factor for each energy equivalent
    to the total cross section in the electron rest frame. This is to reflect the 
    liklihood of interaction. 
    '''
    # calculate beta from gamma
    beta = lz.gammaToBeta(gamma)
    # Generate random incident angle

    Ai_l = np.pi #lz.rand_theta()

    Ai_r = lz.angle_labToRest(Ai_l, beta, gamma)

    # Use random angle, initial photon energy, and beta to translate into electron rest frame
    Ei_rest = lz.labToRest(Ei_lab, beta, gamma, Ai_l)

    if Ei_rest < 1e-6:
        weight = rest_sig_T()
    else:
        weight = rest_sig_KN(Ei_rest)

    # Using incident energy in rest frame, generate a final energy in rest frame via M.C.
    Ef_rest = Ef_rest_mc(Ei_rest)

    # Find the final angle for the transform from Angle in lab + pi - scatter
    As_r = angle_scatter(Ei_rest, Ef_rest)
    Af_r = angle_finalRest(As_r, Ai_r)
    
    # Useing the Final energy in the electron rest frame generate final energy in lab frame
    Ef_lab = lz.restToLab(Ef_rest, Af_r, beta, gamma)

    return Ef_lab, weight


''' ------------------------------------- WORKS ---------------------------------------------- '''

def justTransform(Ei_lab, gamma):
    ''' 
    Function that just lorentz transforms a given energy back between the two frames. Used as a 
    check to make sure that transforms aren't changing energies. 
    '''
    beta = lz.gammaToBeta(gamma)
    Ai_l = lz.rand_theta()
    print(f'random theta: {Ai_l}')

    Ai_r = lz.angle_labToRest(Ai_l, beta, gamma)
    print(f'rest theta: {Ai_r}')
    print()

    # Use random angle, initial photon energy, and beta to translate into electron rest frame
    Ei_rest = lz.labToRest(Ei_lab, beta, gamma, Ai_l)
    print(f'Rest Ei: {Ei_rest}')

    Ei_lab_post = lz.restToLab(Ei_rest, Ai_r, beta, gamma)
    print(f'Rest Ei: {Ei_lab_post}')



