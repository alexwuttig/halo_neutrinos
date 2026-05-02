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

# def build_E0_spline(Emin=me, Emax=1e15, n_points=10000):
#     """
#     Build inverse spline: B_value → E.
#     Emax capped at 1e7 GeV — B(E) saturates beyond this,
#     making inversion numerically unstable.
#     """
#     Es = np.logspace(np.log10(Emin), np.log10(Emax), n_points)
#     Bs = np.array([B(E) for E in Es])
#     return interp1d(Bs, Es, kind='cubic', bounds_error=False, fill_value='extrapolate')


def build_E0_spline(Emin=me, Emax=1e15, n_points=10000):
    Es = np.logspace(np.log10(Emin), np.log10(Emax), n_points)
    Bs = np.array([B(E) for E in Es])
    
    # Find where B stops increasing (saturation) and truncate there
    diffs = np.diff(Bs)
    valid = np.where(diffs <= 0)[0]
    if len(valid) > 0:
        cutoff = valid[0]
        print(f"B(E) saturates at E={Es[cutoff]:.3e} GeV, truncating spline there.")
        Es = Es[:cutoff]
        Bs = Bs[:cutoff]
    
    return interp1d(Bs, Es, kind='cubic', bounds_error=False, fill_value='extrapolate')

# def build_E0_spline(Emin=me, Emax=1e15, n_points=10000):
#     Es = np.logspace(np.log10(Emin), np.log10(Emax), n_points)
#     Bs = np.array([B(E) for E in Es])
    
#     # Remove any non-strictly-increasing B values (handles both saturation and float duplicates)
#     mask = np.concatenate(([True], np.diff(Bs) > 1e-30))
#     Es = Es[mask]
#     Bs = Bs[mask]
    
#     # Also ensure strictly unique to floating point precision
#     _, unique_idx = np.unique(Bs, return_index=True)
#     Es = Es[unique_idx]
#     Bs = Bs[unique_idx]
    
#     print(f"Spline built: {len(Bs)} points, E range [{Es[0]:.3e}, {Es[-1]:.3e}] GeV, B range [{Bs[0]:.3e}, {Bs[-1]:.3e}]")
    
#     return interp1d(Bs, Es, kind='cubic', bounds_error=False, fill_value='extrapolate')

E0_spline = build_E0_spline()

# def get_E0(Ee, tau):
#     """Initial energy of an electron observed at Ee after time tau."""
#     return float(E0_spline(B(Ee) + tau))

def get_E0(Ee, tau):
    B_target = B(Ee) + tau
    B_max = E0_spline.x[-1]  # upper boundary of spline

    if B_target > B_max:
        return np.inf  # electron would need infinite initial energy — unphysical, contributes 0

    E0_val = float(E0_spline(B_target))

    if E0_val < Ee:
        print(f"E0 > Ee for: tau = {tau}, E0 = {E0_val}, Ee = {Ee}, B_target = {B_target} \n")

    if B_target > B_max:
        print(f"tau={tau:.2e} too large: B_target={B_target:.2e} exceeds spline max {B_max:.2e}. Increase Emax in build_E0_spline.\n")

    return E0_val


def main():
    print(f"b_tot(1.0) = {b.b_tot(1.0):.3e}")  # expect ~1e-16
    print(f"b_tot(1e3) = {b.b_tot(1e3):.3e}")  # expect ~1e-10 to 1e-9

    Ee_test = 1.0   # GeV
    tau_test = 1e16 # seconds — try a few values

    B_ee = B(Ee_test)
    B_target = B_ee + tau_test
    E0 = get_E0(Ee_test, tau_test)

    print(f"B(Ee)       = {B_ee:.4e}")
    print(f"B(Ee) + tau = {B_target:.4e}")
    print(f"E0          = {E0:.4e}")
    print(f"E0 > Ee?    = {E0 > Ee_test}")

    # Also check the spline's domain
    Bs_min = B(me)
    Bs_max = B(1e8)
    print(f"Spline B range: [{Bs_min:.3e}, {Bs_max:.3e}]")
    print(f"B_target in range? {Bs_min <= B_target <= Bs_max}")

if __name__ == "__main__":
    main()

    