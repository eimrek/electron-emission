
#  # Richardson-Laue-Dushman equation

import scipy as sp
import scipy.constants
import numpy as np

# CONSTANTS #
kb = sp.constants.value("Boltzmann constant in eV/K") # unit: eV/K
qe = sp.constants.value("elementary charge") # unit: C
me = sp.constants.value("electron mass")/qe #unit: eV*s^2/m^2
hp = sp.constants.value("Planck constant in eV s") #unit: eV*s
hbar = hp/(2*sp.pi) #unit: eV*s
eps0 = sp.constants.value("electric constant")*qe #unit: C^2/(eV*m)
a_rld = 4*sp.pi*me*kb**2*qe/hp**3 #unit: C/(K^2*s*m^2)

# Units:
# current density [A/m^2]: j(F[GV/m], phi[eV])
# inputs: field F [GV/m]
#         work function phi [eV]
def j(F, T, phi):
    # Convert field from [GV/m] to [eV/C*m] to match other units
    F = 1e9/qe*F
    b_lowering = np.sqrt(qe**3*F/(4*np.pi*eps0))
    return a_rld*T**2*np.exp((-phi+b_lowering)/(kb*T))
