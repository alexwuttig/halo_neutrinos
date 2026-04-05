# """
# Energy loss and E0 calculation module
# Handles electron energy loss calculations and initial energy determination
# """
from conv_and_const_module import *
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from conv_and_const_module import me
import b_loss as b


def get_E0_approx(Ee, tau):
    b0 = 1.4e-16
    inv_E0 = 1/Ee - b0 * tau
    if inv_E0 <= 0:
        return np.inf
    return 1 / inv_E0

def B(Ex, Emin=me):
    """Integral of 1/b_tot from Emin to Ex."""
    pts = np.logspace(np.log10(Emin), np.log10(Ex), 200)
    # Lambda function optimization --> cool
    result, _ = quad((lambda E: 1.0 / b.b_tot(E)), Emin, Ex,
                     points=pts, epsabs=1e-12, epsrel=1e-10, limit=1000)
    return result

def build_E0_spline(Emin=me, Emax=1e8, n_points=10000):
    """
    Build inverse spline: B_value → E.
    Emax capped at 1e7 GeV — B(E) saturates beyond this,
    making inversion numerically unstable.
    """
    Es = np.logspace(np.log10(Emin), np.log10(Emax), n_points)
    Bs = np.array([B(E) for E in Es])
    return interp1d(Bs, Es, kind='cubic', bounds_error=False, fill_value='extrapolate')

E0_spline = build_E0_spline()

def get_E0(Ee, tau):
    """Initial energy of an electron observed at Ee after time tau."""
    return float(E0_spline(B(Ee) + tau))


# import numpy as np
# from scipy.integrate import quad
# from scipy.interpolate import interp1d

# from constants_module import me
# import b_loss as b

# class E0Calculator:
#     """
#     Handles calculation of initial electron energy E0 from final energy Ee and time tau.
#     Uses spline interpolation for efficient computation.
#     """
    
#     def __init__(self, Emin=me, Emax=1e100, n_points=10000):
#         """
#         Initialize the E0 calculator by building the interpolation spline.
        
#         Parameters:
#         -----------
#         Emin : float
#             Minimum energy for spline (GeV)
#         Emax : float
#             Maximum energy for spline (GeV)
#         n_points : int
#             Number of points for spline interpolation
#         """
#         self.Emin = Emin
#         self.Emax = Emax
#         self.n_points = n_points
        
#         print(f"Building E0 spline with {n_points} points from {Emin} to {Emax} GeV...")
#         self.spline = self._build_spline()
#         print("E0 spline built successfully!")
    
#     def B_anti(self, Ex, Emin=None):
#         """
#         Computes the antiderivative of 1/btot at a given energy.
        
#         Parameters:
#         -----------
#         Ex : float
#             Energy to evaluate at (GeV)
#         Emin : float, optional
#             Minimum energy for integration (GeV)
        
#         Returns:
#         --------
#         float : Antiderivative value
#         """
#         if Emin is None:
#             Emin = self.Emin
        
#         if Ex <= Emin:
#             print(f"ERROR in B_anti: Ex ({Ex}) <= Emin ({Emin})")
#             return 0
        
#         E_test = np.logspace(np.log10(Emin), np.log10(Ex), 200)

#         def integrand_B(Ex):
#             b_val = b.b_tot(Ex)
#             if b_val == 0 or np.isnan(b_val):
#                 print(f"ERROR: b_tot({Ex}) = {b_val}")
#                 return 0
#             return 1 / b_val
        
#         try:
#             result, error = quad(integrand_B, Emin, Ex, points=E_test, 
#                                 epsabs=1e-12, epsrel=1e-10, limit=1000)
            
#             if np.isnan(result) or np.isinf(result):
#                 print(f"ERROR: B_anti returned {result} for Ex={Ex}")
#                 return 0
            
#             return result
#         except Exception as e:
#             print(f"ERROR in B_anti integration: {e}, Ex={Ex}")
#             return 0
    
#     def _build_spline(self):
#         """
#         Build the inverse spline for E0 calculation.
#         Maps B_anti(E) -> E
        
#         Returns:
#         --------
#         scipy.interpolate.interp1d : Interpolation function
#         """
#         Es = np.logspace(np.log10(self.Emin), np.log10(self.Emax), self.n_points)
        
#         Bs_at_Es = []
#         for i, E in enumerate(Es):
#             if i % 1000 == 0:
#                 print(f"  Progress: {i}/{self.n_points}")
            
#             B_val = self.B_anti(E)
            
#             if np.isnan(B_val) or np.isinf(B_val) or B_val < 0:
#                 print(f"WARNING: Invalid B_anti at E={E}: {B_val}")
            
#             Bs_at_Es.append(B_val)
        
#         # Check monotonicity
#         Bs_array = np.array(Bs_at_Es)
#         diffs = np.diff(Bs_array)
#         if np.any(diffs <= 0):
#             non_monotonic = np.where(diffs <= 0)[0]
#             print(f"WARNING: B_anti not monotonic at {len(non_monotonic)} points!")
#             print(f"First few: {non_monotonic[:5]}")
        
#         # Create inverse spline: B -> E
#         inv_spline = interp1d(Bs_at_Es, Es, bounds_error=False, 
#                              fill_value='extrapolate', kind='linear')
        
#         return inv_spline
    
#     def get_E0(self, Ee, tau, approximate=True):
#         """
#         Computes the inital photon energy as a function of final
#         photon energy and time difference between observation and 
#         injection(tstar). Utilizes a inverse function spline to find
#         the E0 that solves the necessary equation for a given Ee and
#         tau.
        
#         Parameters:
#         -----------
#         Ee : Final electron energy (GeV)
#         tau : Time difference tobs - tinj (seconds)
#         approximate : Whether to use approximations for edge cases
        
#         Returns:
#         --------
#         float : Initial energy E0 (GeV)
#         """
#         E0_max = 1e9  # Maximum energy electron produced by halo (GeV)
        
#         # Check inputs
#         if tau < 0:
#             print(f"ERROR: tau is negative: {tau}")
#             return E0_max
        
#         if Ee <= 0:
#             print(f"ERROR: Ee is non-positive: {Ee}")
#             return E0_max
        
#         # Calculate B_anti(Ee) + tau
#         try:
#             B_Ee = self.B_anti(Ee)
#         except Exception as e:
#             print(f"ERROR in B_anti(Ee): {e}, Ee={Ee}")
#             return E0_max
        
#         value = B_Ee + tau
        
#         if np.isnan(value) or np.isinf(value):
#             print(f"ERROR: B_anti(Ee) + tau = {value}, Ee={Ee}, tau={tau}")
#             return E0_max
        
#         # Use spline to get E0
#         E0_result = self.spline(value)
        
#         if np.isnan(E0_result) or np.isinf(E0_result):
#             print(f"ERROR: Spline returned {E0_result}, Ee={Ee}, tau={tau}, value={value}")
#             return E0_max
        
#         # Apply approximations for numerical stability
#         if approximate:
#             if E0_result < 0:
#                 print(f"WARNING: E0 negative ({E0_result}), setting to {E0_max}")
#                 E0_result = E0_max
#             elif E0_result < Ee:
#                 # If E0 < Ee, energy loss would be negative (unphysical)
#                 # Set E0 slightly above Ee
#                 print(f"WARNING: E0 ({E0_result:.3e}) < Ee ({Ee:.3e}), adjusting to Ee*1.01")
#                 E0_result = Ee * 1.01
#             elif E0_result > E0_max:
#                 print(f"WARNING: E0 ({E0_result:.3e}) > E0_max, capping at {E0_max}")
#                 E0_result = E0_max
        
#         return E0_result
    
#     def test(self):
#         """Run diagnostic tests on the E0 calculator"""
#         print("\n" + "="*60)
#         print("Testing E0 Calculator")
#         print("="*60)
        
#         test_Ees = [0.1, 1, 10, 100, 1000]
#         test_taus = [1e10, 1e12, 1e14, 1e16]
        
#         print(f"\n{'Ee (GeV)':<12} {'tau (s)':<12} {'E0 (GeV)':<12} {'Status'}")
#         print("-" * 60)
        
#         for Ee in test_Ees:
#             for tau in test_taus:
#                 E0 = self.get_E0(Ee, tau)
#                 status = "OK" if not np.isnan(E0) and E0 >= Ee else "PROBLEM"
#                 print(f"{Ee:<12.2e} {tau:<12.2e} {E0:<12.3e} {status}")
        
#         print("\n" + "="*60 + "\n")

# # For testing
# if __name__ == "__main__":
#     # Test the calculator
#     calc = E0Calculator(Emin=me, Emax=1e100, n_points=1000)  # Fewer points for testing
#     calc.test()


# """
# Energy loss and E0 calculation module
# Handles electron energy loss calculations and initial energy determination
# """

# import numpy as np
# from scipy.integrate import quad
# from scipy.interpolate import interp1d

# from conv_and_const_module import me
# import b_loss as b

# class E0Calculator:
#     """
#     Handles calculation of initial electron energy E0 from final energy Ee and time tau.
#     Uses spline interpolation for efficient computation.
#     """
    
#     def __init__(self, Emin=me, Emax=1e100, n_points=10000, verbose=True):
#         """
#         Initialize the E0 calculator by building the interpolation spline.
        
#         - parameters - 
#         Emin : float
#             Minimum energy for spline (GeV)
#         Emax : float
#             Maximum energy for spline (GeV)
#         n_points : int
#             Number of points for spline interpolation
#         verbose : bool
#             Whether to print progress messages
#         """
#         self.Emin = Emin
#         self.Emax = Emax
#         self.n_points = n_points
#         self.verbose = verbose
        
#         if self.verbose:
#             print(f"Building E0 spline with {n_points} points from {Emin} to {Emax} GeV...")
#         self.spline = self._build_spline()
#         if self.verbose:
#             print("E0 spline built successfully!")
    
#     def B_anti(self, Ex, Emin=None):
#         """
#         Computes the antiderivative of 1/btot at a given energy.
        
#         - parameters - 
#         Ex : float
#             Energy to evaluate at (GeV)
#         Emin : float, optional
#             Minimum energy for integration (GeV)
        
#         - returns - 
#         float : Antiderivative value
#         """
#         if Emin is None:
#             Emin = self.Emin
        
#         if Ex <= Emin:
#             if self.verbose:
#                 print(f"ERROR in B_anti: Ex ({Ex}) <= Emin ({Emin})")
#             return 0
        
#         E_test = np.logspace(np.log10(Emin), np.log10(Ex), 200)

#         def integrand_B(Ex):
#             b_val = b.b_tot(Ex)
#             if b_val == 0 or np.isnan(b_val):
#                 if self.verbose:
#                     print(f"ERROR: b_tot({Ex}) = {b_val}")
#                 return 0
#             return 1 / b_val
        
#         result, error = quad(integrand_B, Emin, Ex, points=E_test, 
#                             epsabs=1e-12, epsrel=1e-10, limit=1000)
        
#         if np.isnan(result) or np.isinf(result):
#             if self.verbose:
#                 print(f"ERROR: B_anti returned {result} for Ex={Ex}")
#             return 0
        
#         return result

    
#     def _build_spline(self):
#         """
#         Build the inverse spline for E0 calculation.
#         Maps B_anti(E) -> E
        
#         - returns - 
#         scipy.interpolate.interp1d : Interpolation function
#         """
#         Es = np.logspace(np.log10(self.Emin), np.log10(self.Emax), self.n_points)
        
#         Bs_at_Es = []
#         for i, E in enumerate(Es):
#             if self.verbose and i % 1000 == 0:
#                 print(f"  Progress: {i}/{self.n_points}")
            
#             B_val = self.B_anti(E)
            
#             if np.isnan(B_val) or np.isinf(B_val) or B_val < 0:
#                 if self.verbose:
#                     print(f"WARNING: Invalid B_anti at E={E}: {B_val}")
            
#             Bs_at_Es.append(B_val)
        
#         # Check monotonicity
#         Bs_array = np.array(Bs_at_Es)
#         diffs = np.diff(Bs_array)
#         if np.any(diffs <= 0):
#             if self.verbose:
#                 non_monotonic = np.where(diffs <= 0)[0]
#                 print(f"WARNING: B_anti not monotonic at {len(non_monotonic)} points!")
#                 print(f"First few: {non_monotonic[:5]}")
        
#         # Create inverse spline: B -> E
#         inv_spline = interp1d(Bs_at_Es, Es, bounds_error=False, 
#                              fill_value='extrapolate', kind='linear')
        
#         return inv_spline
    
#     def get_E0(self, Ee, tau, approximate=True):
#         """
#         Computes the inital photon energy as a function of final
#         photon energy and time difference between observation and 
#         injection(tstar). Utilizes a inverse function spline to find
#         the E0 that solves the necessary equation for a given Ee and
#         tau.

#         - params -
#         Ee: final photon energy
#         tau: tobs - tinj (Defined in the hooper paper)

#         - returns - 
#         value of E0 at given Ee and tau
#         maximum value of Ee for wich the function is defined, above that 
#         there isn't any well defined values
#         """
#         E0_max = 1e9  # Maximum energy electron produced by halo (GeV)
        
#         # Check inputs
#         if tau < 0:
#             if self.verbose:
#                 print(f"ERROR: tau is negative: {tau}")
#             return E0_max
        
#         if Ee <= 0:
#             if self.verbose:
#                 print(f"ERROR: Ee is non-positive: {Ee}")
#             return E0_max
        
#         # Calculate B_anti(Ee) + tau

#         B_Ee = self.B_anti(Ee)
#         value = B_Ee + tau
        
#         if np.isnan(value) or np.isinf(value):
#             if self.verbose:
#                 print(f"ERROR: B_anti(Ee) + tau = {value}, Ee={Ee}, tau={tau}")
#             return E0_max
        
#         # Use spline to get E0
#         E0_result = self.spline(value)
        
#         if np.isnan(E0_result) or np.isinf(E0_result):
#             if self.verbose:
#                 print(f"ERROR: Spline returned {E0_result}, Ee={Ee}, tau={tau}, value={value}")
#             return E0_max
        
#         # Apply approximations for numerical stability
#         if approximate:
#             if E0_result < 0:
#                 if self.verbose:
#                     print(f"WARNING: E0 negative ({E0_result}), setting to {E0_max}")
#                 E0_result = E0_max
#             elif E0_result < Ee:
#                 if self.verbose:
#                     print(f"WARNING: E0 ({E0_result:.3e}) < Ee ({Ee:.3e}), adjusting to Ee*1.01")
#                 E0_result = Ee * 1.01
#             elif E0_result > E0_max:
#                 if self.verbose:
#                     print(f"E0 ({E0_result:.3e}) > E0_max, capping at {E0_max}")
#                 E0_result = E0_max
        
#         return E0_result
    
#     def test(self):
#         """Run diagnostic tests on the E0 calculator"""
#         print("\n" + "="*60)
#         print("Testing E0 Calculator")
#         print("="*60)
        
#         test_Ees = [0.1, 1, 10, 100, 1000]
#         test_taus = [1e10, 1e12, 1e14, 1e16]
        
#         print(f"\n{'Ee (GeV)':<12} {'tau (s)':<12} {'E0 (GeV)':<12} {'Status'}")
#         print("-" * 60)
        
#         for Ee in test_Ees:
#             for tau in test_taus:
#                 E0 = self.get_E0(Ee, tau)
#                 status = "OK" if not np.isnan(E0) and E0 >= Ee else "PROBLEM"
#                 print(f"{Ee:<12.2e} {tau:<12.2e} {E0:<12.3e} {status}")
        
#         print("\n" + "="*60 + "\n")


# # For testing
# if __name__ == "__main__":
#     calc = E0Calculator(Emin=me, Emax=1e100, n_points=1000)
#     calc.test()