
#  # Fowler-Nordheim equation

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

# FN constants:
a_fn = qe**3/(8*sp.pi*hp) #unit: C^3/(eV*s)
b_fn = 8*sp.pi*np.sqrt(2*me)/(3*qe*hp)  #unit: 1/(C*eV^0.5*m)

# Units:
# current density [A/m^2]: j(F[GV/m], phi[eV])
# inputs: field F [GV/m]
#         work function phi [eV]
def j(F, phi):

    # Convert field from [GV/m] to [eV/C*m] to match other units
    F = 1e9/qe*F

    fb = 4*sp.pi*eps0*phi**2/qe**3
    nu = 1-F/fb+1/6*F/fb*np.log(F/fb)
    tau = 1+1/9*F/fb*(1-1/2*np.log(F/fb))

    return a_fn*F**2/(phi*tau**2)*np.exp(-nu*b_fn*phi**(3/2)/F)

# Units:
# current density [A/m^2]: j(F[GV/m], phi[eV])
# inputs: field F [GV/m]
#         temperature [K]
#         work function phi [eV]
def j_temp(F, T, phi):
    if T <= 0:
        T = 1
    F_conv = 1e9/qe*F
    dt = 2*F_conv/(3*b_fn*np.sqrt(phi))
    theta = np.pi*kb*T/dt/np.sin(np.pi*kb*T/dt)
    return theta*j(F, phi)
