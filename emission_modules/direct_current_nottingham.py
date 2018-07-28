import scipy as sp
import scipy.constants
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

# CONSTANTS #
kb = sp.constants.value("Boltzmann constant in eV/K") # unit: eV/K
qe = sp.constants.value("elementary charge") # unit: C
me = sp.constants.value("electron mass")/qe*1e-18 #unit: eV*s^2/nm^2 NB: COPPER EFFECTIVE MASS is 1.01*
hp = sp.constants.value("Planck constant in eV s") #unit: eV*s
hbar = hp/(2*sp.pi) #unit: eV*s
eps0 = sp.constants.value("electric constant")*qe*1e-9 #unit: C^2/(eV*nm)
arld = 4*sp.pi*me*kb**2*qe/hp**3 #unit: C/(K^2*s*nm^2)
phi_cu = 4.5 #unit: eV
mu_cu = 7.0 #unit: eV

z_sommer = qe*me/(2*np.pi**2*hbar**3) # units: C/(s*eV^2*nm^2)

# Gauss quadrature
gauss_quad_x = [[-np.sqrt(1/3), np.sqrt(1/3)],
                [-np.sqrt(3/5), 0, np.sqrt(3/5)],
                [-np.sqrt(3/7+2/7*np.sqrt(6/5)), -np.sqrt(3/7-2/7*np.sqrt(6/5)), np.sqrt(3/7-2/7*np.sqrt(6/5)), np.sqrt(3/7+2/7*np.sqrt(6/5))]]
gauss_quad_w = [[1, 1],
                [5/9, 8/9, 5/9],
                [(18-np.sqrt(30))/36, (18+np.sqrt(30))/36, (18+np.sqrt(30))/36, (18-np.sqrt(30))/36]]


# ## Supply function
# The reference energy (the zero energy) is the bottom of the conduction band.
#
# The supply function is the following:
# \begin{equation}
#     N(U,T)= \frac{4 \pi m_e k_B T}{h_P^3} \ln \left( 1+\exp \left(-\frac{U-\mu}{k_B T}\right) \right)
# \end{equation}

def supply(U, T):
    # Unit: 1/(nm^2*eV*s)
    return 4*sp.pi*me*kb*T/hp**3*sp.log1p(sp.exp(np.min([-(U-mu_cu)/(kb*T), 700.0])))


# ## Tunnelling probability

# equation (14) from M&G
def elliptic_k(k):
    return sp.integrate.quad(lambda x: (1-k**2*sp.sin(x)**2)**(-0.5), 0, sp.pi/2)[0]
def elliptic_e(k):
    return sp.integrate.quad(lambda x: (1-k**2*sp.sin(x)**2)**(0.5), 0, sp.pi/2)[0]

def nu(y):
    if y > 1:
        # eq. (15)
        return -sp.sqrt(y/2)*(-2*elliptic_e(sp.sqrt(y-1)/sp.sqrt(2*y))+(y+1)*elliptic_k(sp.sqrt(y-1)/sp.sqrt(2*y)))
    else:
        # eq. (16)
        return sp.sqrt(1+y)*(elliptic_e(sp.sqrt(1-y)/sp.sqrt(1+y))-y*elliptic_k(sp.sqrt(1-y)/sp.sqrt(1+y)))

# Forbes approximation (from Jensen 2007)
def nu_approx(y):
    return 1-y**2+1.0/3*y**2*sp.log(y)

# Murpy and Good tunnelling probability: eq. (17) and (18)
def tunnel(U, F, phi):
    u_lim = mu_cu+phi-sp.sqrt(2)/2*sp.sqrt(qe**2*F/(4*sp.pi*eps0))
    if U >= u_lim:
        return 1.0
    y = sp.sqrt(qe**2*F/(4*sp.pi*eps0))/abs(U-mu_cu-phi)
    inter = 4.0/(3*hbar*F)*sp.sqrt(2*me*abs(U-mu_cu-phi)**3)*nu_approx(y)
    # to suppress the overflow warnings #
    if inter > 150:
        return sp.exp(-inter)
    return (1+sp.exp(inter))**(-1)


# ## Current density

# units:
# in: F (V/nm), T (K), phi (eV)
# out: A/m^2

def j_python(F, T, phi):
    ppoint = mu_cu+phi; # main problem point
    rel_points = np.array([0.01, 0.02, 0.05, 0.1])
    problem_points = np.append([ppoint + rel_points, ppoint - rel_points], [mu_cu, ppoint])
    return qe*sp.integrate.quad(lambda u: supply(u,T)*tunnel(u, F, phi), 0, 20, points=problem_points, epsrel=1e-4, epsabs=0)[0]*1e18

def j_gauss(F, T, phi):
    gauss_rank = 2
    e_regs = [0.0, mu_cu-1.0, mu_cu+phi-0.1, mu_cu+phi+0.2, 20.0]
    n_regs = [16, 64, 64, 16]
    j_accumulation = 0.0
    for i in range(len(n_regs)):
        e_min = e_regs[i]
        e_max = e_regs[i+1]
        de = (e_max-e_min)/(n_regs[i]-1)
        for i in range(n_regs[i]-1):
            e1 = e_min + i*de
            e2 = e_min + (i+1)*de
            gauss_acc = 0.0
            for gi in range(len(gauss_quad_x[gauss_rank])):
                arg = (e2-e1)/2*gauss_quad_x[gauss_rank][gi]+(e2+e1)/2
                gauss_acc += gauss_quad_w[gauss_rank][gi]*qe*supply(arg,T)*tunnel(arg, F, phi)
            j_accumulation += (e2-e1)/2*gauss_acc
    return j_accumulation*1e18

def j(F, T, phi):
    return j_gauss(F, T, phi)

# ## Nottingham

# regarding units: as the integration gives the result in eV, the additional coefficient 1/qe can be dropped
# to convert the result from eV/(s*nm^2) to W/nm^2
# units:
# in: F (V/nm), T (K), phi (eV)
# out: W/m^2

def heat_flux_gauss(F, T, phi):
    gauss_rank = 2
    e_min = 0.0
    e_max = 20.0
    n_div = 256
    de = (e_max-e_min)/(n_div-1)
    hf_accumulation = 0.0
    inner_accumulation = 0.0
    e2_inner = e_min
    for i in range(n_div-1):
        e1 = e_min + i*de
        e2 = e_min + (i+1)*de
        e1_inner = e2_inner
        e2_inner = (e1+e2)/2
        # Inner integration update
        inner_gauss_acc = 0.0
        for gi in range(len(gauss_quad_x[gauss_rank])):
            arg = (e2_inner-e1_inner)/2*gauss_quad_x[gauss_rank][gi]+(e2_inner+e1_inner)/2
            inner_gauss_acc += gauss_quad_w[gauss_rank][gi]*tunnel(arg, F, phi)
        inner_accumulation += (e2_inner-e1_inner)/2*inner_gauss_acc

        # outer integration
        gauss_acc = 0.0
        for gi in range(len(gauss_quad_x[gauss_rank])):
            arg = (e2-e1)/2*gauss_quad_x[gauss_rank][gi]+(e2+e1)/2
            gauss_acc += gauss_quad_w[gauss_rank][gi]*z_sommer*(arg-mu_cu)/(1+np.exp(np.min([(arg-mu_cu)/(kb*T), 700.0])))*inner_accumulation
        hf_accumulation += (e2-e1)/2*gauss_acc
    return hf_accumulation*1e18

def heat_flux(F, T, phi):
    return heat_flux_gauss(F, T, phi)

def deltaE(F, T, phi):
    return heat_flux_gauss(F, T, phi)/j_gauss(F, T, phi)
