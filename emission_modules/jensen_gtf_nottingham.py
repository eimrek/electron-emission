# # Jensen's GTF and nottingham effect

# current density [A/m^2]: j(F[GV/m], T[K], phi[eV])
# delta E (Nottingham's average energy difference) [eV]: deltaE(F[GV/m], T[K], phi[eV])

import scipy as sp
import scipy.constants
import numpy as np

# ## Constants

# CONSTANTS #
kB = sp.constants.value("Boltzmann constant in eV/K") # unit: eV/K
qe = sp.constants.value("elementary charge") # unit: C
me = sp.constants.value("electron mass")*10**(-18)/qe #unit: eV*s^2/nm^2
hP = sp.constants.value("Planck constant in eV s") #unit: eV*s
hbar = hP/(2*sp.pi) #unit: eV*s
eps0 = sp.constants.value("electric constant")*qe*10**(-9) #unit: C^2/(eV*nm)
aRLD = 4*sp.pi*me*kB**2*qe/hP**3*10**18 #unit: C/(K^2*s*m^2)

# Chemical potential of copper !!!
muCu = 7.0 #unit: eV

### --------------------------------------------------------------------
### GTF equations
### --------------------------------------------------------------------

# eq. (16) from Jensen 2008 #
def sigma(x):
    try:
        #return (1+x**2)/(1-x**2) -0.039*x**2*(9.1043+2.7163*x**2+x**4)  # Works until 11 GV/m (at phi=4.5)
        return (1+x**2)/(1-x**2) -0.039*x**2*(9.1043+2.7163*x**2)   # Works until 12 GV/m (at phi=4.5)
    except:
        return np.inf

def N(n,s):
    N = 0.0
    if n==1.0:
        N = sp.exp(-s)*(s+1)
    else:
        N = n**2*sigma(1/n)*sp.exp(-s)+sigma(n)*sp.exp(-n*s)
    return N

def ns(F, T, phiCu):
    Q = qe**2/(16*sp.pi*eps0)
    phi = phiCu-(4*Q*F)**(0.5)
    y = (4*Q*F)**(0.5)/phiCu
    betaT = 1/(kB*T)
    Bq = sp.pi/hbar*phi*sp.sqrt(2*me)*(Q/F**3)**0.25
    nu = 1-y**2+1.0/3*y**2*sp.log(y)
    t = 1.0/9*y**2*(1-sp.log(y))+1
    Bfn = 4.0/(3*hbar*F)*sp.sqrt(2*me*phiCu**3)*nu
    Cfn = 2*phi/(hbar*F)*sp.sqrt(2*me*phiCu)*t
    Tmin = 1/(kB*1/phi*Cfn)
    Tmax = 1/(kB*1/phi*Bq)
    a = -3*(2*Bfn-Bq-Cfn)
    b = 3*(2*Bfn-Bq-Cfn)+Bq-Cfn
    c = Cfn-phi*betaT
    z = -b/(2*a)+sp.sqrt((b/(2*a))**2-c/a)
    s = 0.0
    n = 0.0
    if T>Tmax:
        s = Bq
        n = betaT*phi/Bq
    elif T<Tmin:
        s = Bfn
        n = betaT*phi/(Cfn)
    else:
        s = Bfn+b/2*z**2+2/3*a*z**3
        n = 1.0
    return (n,s)

# F-phi validity (gtf & nottingham):
#     phi(eV) max_F(GV/m)  max_F(GV/m)
#     5.0     15
#     4.5     12           11
#     4.4     11.5
#     4.3     11
#     4.2     10.5         9.5
#     4.1     10
#     4.0     9.5          8.5
#     2.0     2.7
#
# input values:
#    F is electric field in GV/m
#    T is temperature in K
#    phiCu is the work function in eV
# output value:
#    j - current density in A/m^2
def j(F, T, phiCu):
    (n,s) = ns(F,T, phiCu)
    J = aRLD*T**2*N(n,s)
    return J

### --------------------------------------------------------------------
### Nottingham effect
### --------------------------------------------------------------------

# eq. (17) from Jensen 2008 (corrected version)#
def d_sigma(x):
    try:
        #return (-1+4*x**2+x**4)/(1-x**2)**2-0.027066*x**2*(13.118+11.742*x**2+7.205*x**4+x**6) # seemingly worse
        return (-1+4*x**2+x**4)/(1-x**2)**2-0.027066*x**2*(13.118+11.742*x**2+7.205*x**4)
    except:
        return np.inf

# eq. (20) from Jensen 2008 #
def d_N(n, s):
    return n**3*d_sigma(1/n)*sp.exp(-s)+((n*s+1)*sigma(n)-d_sigma(n))*sp.exp(-n*s)

# the division of d_N/N, (is singular at n=1.0, but has a finite limit) #
def d_N_N(n,s):
    if n==1.0:
        return (s**2+2*s+2)/(2*(s+1))
        #return d_N(1.0001,s)/N2(1.0001,s)
    else:
        return d_N(n,s)/N(n,s)

# eq. (19) from Jensen 2008 #
def deltaE(F, T, phiCu):
    (n,s) = ns(F,T, phiCu)
    return kB*T*d_N_N(n,s)

# the field limit of delta E #
def deltaE_field(F, T, phiCu):
    (n,s) = ns(F,T, phiCu)
    return kB*T*n*d_sigma(1/n)/sigma(1/n)
