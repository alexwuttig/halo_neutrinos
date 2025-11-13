''' conversions '''
conv_pc_cm = 3.086e+18       # conversion factor from [pc] to [cm]
conv_cm_pc = 3.24e-19        # conversion factor from [cm] to [pc]
conv_s_yr = 3.171e-8         # conversion factor from [sec] to [yr]
conv_yr_sec = 3.154e+7       # conversion factor from [yr] to [sec]
conv_rad_degree = 57.2958    # conversion factor from [rad] to [deg]
c = 2.998e+10                # speed of light, in [cm/s]


class Params:
    """
    Holds all tunable parameters for the halo model.
    """
    def __init__(self):
        # injection spectrum
        self.alpha = 2.0          # electron power-law index
        self.Ecut = 1e6           # electron cutoff energy (GeV)

        # diffusion parameters
        self.rh = 30*conv_pc_cm       # transition radius [pc]
        self.D0 = 1e26            # cm^2/s (inner zone)
        self.Dism = 4e28          # cm^2/s (outer zone)
        self.del_exponent = 1/3   # spectral index of diffusion coefficient

        # luminosity parameters
        self.tau0 = 12 * conv_yr_sec
        self.Edot = 3.8e34        # erg/s (for Monogem)
        self.tobs = 110e3 * conv_yr_sec  # observed age of pulsar

        # location, age, and distance perameters
        self.rs = 1             # alactic location of source
        self.d = 288            # distance in parsecs
        self.t_age = self.tobs + (self.d * conv_pc_cm)/c # actual age of pulsar(accounting for light travel time)

        # integration resolution
        self.n_points = 8

# Create a single instance of Params
params = Params()