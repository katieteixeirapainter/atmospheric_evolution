#%%


## Object oriented Coupled Atmospheric Evolution for TRAPPIST-1c.



## Imports ##



import numpy as np
import scipy as sp
import matplotlib as mpl
from scipy.optimize import fsolve
from scipy.optimize import fmin
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline as US
import matplotlib.pyplot as plt
import time
import pandas as pd
import copy
import os
import itertools
import warnings
import concurrent.futures
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import InterpolatedUnivariateSpline
plt.rcParams.update({'font.family':'sans-serif'})
import matplotlib.lines as mlines
from num2tex import num2tex



## Saving Preferences ##



parent_dir = '/Users/ket2264/Documents/Work/Research/Caroline/TRAPPIST_1c'
figures_dir = "/Users/ket2264/Documents/Work/Research/Caroline/TRAPPIST_1c/Figures for Paper"



## Term Lookup ##



DICTIONARY = {'G':"Newton's Gravitational Constant",
              'Rg':'Universal Gas Constant',
              'k_b':"Boltzmann Constant",
              'AU':'Astronomical Unit',
              'year':'Year',
              'Gyr':'Gigayear',
              'm_bar_co2':"Molar Mass of" + r' $\rm{{CO}_2}$',
              'm_molecule_co2':"Molecular Mass of" + r' $\rm{{CO}_2}$',
              'AMU':'Atomic Molecular Unit',
              'P2W':'Power-to-Weight Ratio',
              'tau_':'Half Life',
              'lambd':'Decay Constant',
              'tau':'Half Life',
              'F_melt_flag':'Determines Formula Used in _F_melt',
              'gamma_ad':'Adiabatic Gradient of Melt Subtracted by Adiabatic Gradient of Solid Mantle',
              'weath_demand': "CO2 Drawdown",
              'P_dense':'Maximum Pressure at which Melting Occurs',
              'F_offset':'Net CO2 Input that Will Cause 100 Bar of CO2 to Accumulate in 100 Million Years',
              'M':'Mass',
              'R':'Radius',
              'L':'Luminosity',
              'T_surface':'Surface Temperature',
              'U238':'Abundance of Uranium 238 in the Mantle',
              'U235':'Abundance of Uranium 235 in the Mantle',
              'Th':'Abundance of Thorium in the Mantle',
              'K':'Abundance of Potassium in the Mantle',
              'F_degas':'Mantle Degassing Flux',
              'SA':'Surface Area',
              'g':'Surface Gravitational Acceleration',
              'star':{'age': "Age of Star",
                      'type': "Type of Star",
                      'name': "Name of Star",
                      'L_X': "X-Ray Luminosity of Star"},
              'a':'Semi-Major Axis',
              'M_':'Mass',
              'R_':'Radius',
              'a_':'Semi-Major Axis',
              'name':'Name of Planet',
              'Ev': "Viscosity Activation Energy",
              'alpha': "Thermal Expansion Coefficient",
              'kappa': "Thermal Diffusivity",
              'df_melt_dP': "Change in Melt Fraction for a Change in Melting Pressure",
              'Cp': "Specific Heat Capacity at Constant Pressure",
              'k': "Thermal Conductivity",
              'C1': "Heat Flow Scaling Law Constant",
              'L_m': "Latent Heat",
              'f_C_trapped': "Fraction of Devolatilized Carbon That is Trapped and Still Recycles into Mantle",
              'F_melt_plume': "Volume Flux of Plume Melt",
              'rho_melt': "Melt Density",
              'DT_sol_dep': "Increase in Solidus Temperature Due to Full Depletion",
              'rho_man': "Mantle Density",
              'rho0_man': "Mantle Density at 0 Pressure",
              'dT_dP_man': "Adiabatic Gradient in Mantle",
              'd_man':'Whole Mantle Thickness',
              'mu_ref': "Mantle Reference Viscosity",
              'abundances':'Abundance of Heat-Producing Elements',
              'abundances_':'Abundance of Heat-Producing Elements',
              'loss_rate':"Atmospheric Mass Loss Rate Regime",
              'R_core': "Core Radius",
              'F_XUV': "Incident XUV Flux on Planet",
              'crit_d_crust': "Critical Value of Crust Thickness",
              'mun': "Viscosity Pre-Exponential Factor",
              'd_dense':'Depth at which Melting Stops because Pressure is Too High',
              'sol':'solve_ivp Solution Object',
              't':'Time',
              't_gyr':'Time',
              'T_man':'Mantle Temperature',
              'cumulative_V_crust':'Cumulative Volume of Crust Produced',
              'C_man':'Amount of Carbon in the Mantle',
              'V_crust':'Crust Volume',
              'C_crust':'Amount of Carbon in the Crust',
              'd_lid':'Lithosphere Thickness',
              'Q_U238_crust': "Heat Production from U238 in the Crust",
              'Q_U235_crust': "Heat Production from U235 in the Crust",
              'Q_Th_crust': "Heat Production from Th in the Crust",
              'Q_K_crust': "Heat Production from K in the Crust",
              'Q_U238': "Heat Production from U238 in the Mantle",
              'Q_U235': "Heat Production from U235 in the Mantle",
              'Q_Th': "Heat Production from Th in the Mantle",
              'Q_K': "Heat Production from K in the Mantle",
              'M_atm':'Mass of Atmosphere',
              'Q_man':'Heat Production in the Mantle',
              'Q_crust':'Heat Production in the Crust',
              'P_surface':'Surface Pressure',
              'tidal_heating':'Is Tidal Heating Considered?',
              'outgassing_rate':'Outgassing Rate',
              'V_man':'Mantle Volume',
              'd_crust':'Crust Thickness',
              'SA_man_wo_crust':'Surface Area of the Mantle (Without the Crust)',
              'SA_man_wo_lid':'Surface Area of the Mantle (Without the Lithosphere)',
              'Rai': "Internal Rayleigh Number",
              'theta': "Frank-Kamenetskii Parameter",
              'F_man': "Mantle Convective Heat Flux",
              'T_lid': "Temperature at Base of Lithosphere",
              'x_crust':'Concentration of Heat-Producing Elements in the Crust',
              'x_man': "Concentration of Heat-Producing Elements in the Mantle",
              'T_crust':'Temperature at Base of Crust',
              'P_lid':'Pressure at Base of Lithosphere',
              'd_melt':'Depth at which Melting Begins',
              'P_melt':'Pressure at which Melting Begins',
              'f_melt':'Melt Fraction',
              'F_melt':'Volume Flux of Melt',
              'f_2':'',
              'f_U':'Distribution Coefficient for Uranium',
              'f_Th':'Distribution Coefficient for Thorium',
              'f_K':'Distribution Coefficient for Potassium',
              'd_carb':'Depth at which Decarbonation Occurs',
              'V_volc':'',
              'F_dcarb':'Decarbonation Flux',
              'F_weath_SL':'Flux Threshold When Supply-Limited Weathering Begins'}

U = {'G':"m^3/kg/s^2",
    'Rg':'J/K/mol',
    'k_b':"m^2 kg/s^2/K",
    'AU':'m',
    'year':'s',
    'Gyr':'s',
    'm_bar_co2':'kg',
    'm_molecule_co2':'kg',
    'AMU':'kg',
    'P2W':'W/kg',
    'tau_':'yr',
    'lambd':'',
    'tau':'s',
    'F_melt_flag':'',
    'gamma_ad':'K/Pa',
    'weath_demand': "mol/kg",
    'P_dense':'Pa',
    'F_offset':'mol/s',
    'M':'kg',
    'R':'m',
    'L':'W',
    'T_surface':'K',
    'U238':'PPM',
    'U235':'PPM',
    'Th':'PPM',
    'K':'PPM',
    'F_degas':'mol/s',
    'SA':'m^2',
    'g':'m^2/s^2',
    'star':{'age': "s",
            'type': "",
            'name': "",
            'age_':'Gyr',
            'L_X': "W"},
    'a':'m',
    'M_':r'$M_{\odot}$',
    'R_':r'$R_{\odot}$',
    'a_':'AU',
    'name':'',
    'Ev': "J/mol",
    'alpha': "K^-1",
    'kappa': "m^2/s",
    'df_melt_dP': "Pa^-1",
    'Cp': "J/kg/K",
    'k': "W/m/K",
    'C1': "",
    'L_m': "J/kg",
    'f_C_trapped': "",
    'F_melt_plume': "m^3/s",
    'rho_melt': "kg/m^3",
    'DT_sol_dep': "K",
    'rho_man': "kg/m^3",
    'rho0_man': "kg/m^3",
    'dT_dP_man': "K/Pa",
    'd_man':'m',
    'mu_ref': "Pa s",
    'abundances':'PPM',
    'abundances_':r'$\ocross$',
    'loss_rate':"",
    'R_core': "m",
    'F_XUV': "W/m^2",
    'crit_d_crust': "m",
    'mun': "Pa s",
    'd_dense':'m',
    'sol':'',
    't':'s',
    't_gyr':'Gyr',
    'T_man':'K',
    'cumulative_V_crust':'m^3',
    'C_man':'mol',
    'V_crust':'m^3',
    'C_crust':'mol',
    'd_lid':'m',
    'Q_U238_crust': "W",
    'Q_U235_crust': "W",
    'Q_Th_crust': "W",
    'Q_K_crust': "W",
    'Q_U238': "W",
    'Q_U235': "W",
    'Q_Th': "W",
    'Q_K': "W",
    'M_atm':'kg',
    'Q_man':'W',
    'Q_crust':'W',
    'P_surface':'Pa',
    'tidal_heating':'',
    'outgassing_rate':'kg/s',
    'V_man':'m^3',
    'd_crust':'m',
    'SA_man_wo_crust':'m^2',
    'SA_man_wo_lid':'m^2',
    'Rai': "",
    'theta': "",
    'F_man': "W/m^2",
    'T_lid': "K",
    'x_crust':'PPM/m^3',
    'x_man': "PPM/m^3",
    'T_crust':'K',
    'P_lid':'Pa',
    'd_melt':'m',
    'P_melt':'Pa',
    'f_melt':'',
    'F_melt':'m^3/s',
    'f_2':'',
    'f_U':'',
    'f_Th':'',
    'f_K':'',
    'd_carb':'m',
    'V_volc':'m^3',
    'F_dcarb':'mol/s',
    'F_weath_SL':'mol/s'}

AKA = {'G':"",
    'Rg':'',
    'k_b':"k_boltz",
    'AU':'AU_to_m',
    'year':'[NEW]',
    'Gyr':'[NEW]',
    'm_bar_co2':'',
    'm_molecule_co2':'',
    'AMU':'amu_to_kg',
    'P2W':'Q',
    'tau_':'tau_half',
    'lambd':'lambda',
    'tau':'tau_rad',
    'F_melt_flag':'v_scale_flag',
    'gamma_ad':'',
    'weath_demand': "",
    'P_dense':'[NEW]',
    'F_offset':'[NEW]',
    'M':'M_kg',
    'R':'R_m',
    'L':'',
    'T_surface':'',
    'U238':'',
    'U235':'',
    'Th':'',
    'K':'',
    'F_degas':'Fdegas',
    'SA':'',
    'g':'',
    'star':{'age': "[NEW]",
            'type': "",
            'name': "",
            'age_':'age',
            'L_X': ""},
    'a':'a_m',
    'M_':'M',
    'R_':'R',
    'a_':'a',
    'name':'',
    'Ev': "",
    'alpha': "",
    'kappa': "",
    'df_melt_dP': "dphi_dP",
    'Cp': "",
    'k': "",
    'C1': "",
    'L_m': "",
    'f_C_trapped': "f",
    'F_melt_plume': "plume_vol_flux",
    'rho_melt': "",
    'DT_sol_dep': "",
    'rho_man': "rho",
    'rho0_man': "rho_m",
    'dT_dP_man': "dT_dP",
    'd_man':'d',
    'mu_ref': "",
    'abundances':'[NEW]',
    'abundances_':'[NEW]',
    'loss_rate':"",
    'R_core': "Rc",
    'F_XUV': "",
    'crit_d_crust': "dcr_crit",
    'mun': "",
    'd_dense':'[NEW]',
    'sol':'',
    't':'Time',
    't_gyr':'time',
    'T_man':'Ti',
    'cumulative_V_crust':'crust',
    'C_man':'Rman',
    'V_crust':'crust2',
    'C_crust':'Rcrust',
    'd_lid':'delta',
    'Q_U238_crust': "u238_crust",
    'Q_U235_crust': "u235_crust",
    'Q_Th_crust': "th_crust",
    'Q_K_crust': "K_crust",
    'Q_U238': "u238_man",
    'Q_U235': "u235_man",
    'Q_Th': "th_man",
    'Q_K': "K_man",
    'M_atm':'',
    'Q_man':'[NEW]',
    'Q_crust':'[NEW]',
    'P_surface':'pCO2',
    'tidal_heating':'[NEW]',
    'outgassing_rate':'',
    'V_man':'',
    'd_crust':'delta_c',
    'SA_man_wo_crust':'As',
    'SA_man_wo_lid':'As2',
    'Rai': "",
    'theta': "",
    'F_man': "ql",
    'T_lid': "Tl",
    'x_crust':'x_c',
    'x_man': "x_m",
    'T_crust':'Tc',
    'P_lid':'',
    'd_melt':'',
    'P_melt':'P_melt1',
    'f_melt':'phi',
    'F_melt':'vol_flux',
    'f_2':'frac2',
    'f_U':'fracU',
    'f_Th':'fracTh',
    'f_K':'fracK',
    'd_carb':'z_carb',
    'V_volc':'volc',
    'F_dcarb':'',
    'F_weath_SL':'Fws'}

def Q(q, subattribute=None): 
    name = DICTIONARY[q] if subattribute is None else DICTIONARY[q][subattribute]
    return name
def QU(q, subattribute=None): 
    name = DICTIONARY[q] if subattribute is None else DICTIONARY[q][subattribute]
    units = U[q] if subattribute is None else U[q][subattribute]
    return name + f" [{units}]"



## FORMULAS ##



def _SA(R):
    return 4*pi*R**2

def _V(R):
    return 4/3*pi*R**3

def _g(M, R):
    return G*M/R**2

def _mun(mu_ref, Ev):
    return mu_ref/np.exp(Ev/(Rg*1623))

def _mu_i(mun, Ev, T_man):
    return mun * np.exp(Ev/(Rg*T_man))

def _d_man(M):
    return 2890000*(M/EARTH['M'])**0.28

def _Q(M_man, abundances):

    Q = {}
    for E, abundance in abundances.items(): Q[E] = M_man*P2W[E]*abundance*(1/1e6)*np.exp(4.5e9/lambd[E])

    return Q

def _Rai(rho, g, alpha, Tp, T_surface, d, kappa, mu_i):
    return (rho*g*alpha*d**3*(Tp-T_surface))/(kappa*mu_i)

def _theta(Ev, T_man, T_surface):
    return Ev*(T_man-T_surface)/(Rg*T_man**2)

def _Q_lid(C1, k, T_man, T_surface, d_man, theta, Rai):
    return C1*k*(T_man-T_surface)/d_man*theta**(-4/3)*Rai**(1/3)

def _T_lid(Ev, T_man):
    return T_man-2.5*(Rg*T_man**2)/Ev

def _T_crust(T_surface, T_lid, d_lid, d_crust, x_crust, x_man, k):
    return (T_surface*(d_lid - d_crust) + T_lid*d_crust)/d_lid + (x_crust * d_crust**2 * (d_lid - d_crust))/(2*k*d_lid) + (x_man/k)*((d_lid*d_crust)/2 + (d_crust**3)/(2*d_lid) - d_crust**2)

def _d_melt(T_man, DT_sol_dep, d_crust, crit_d_crust, dT_dP_man, rho0_man, g):
    return ((T_man - (1423 + DT_sol_dep*max(1, d_crust/crit_d_crust)))/(120e-9 - dT_dP_man))/(rho0_man * g)

def _f_melt(P_melt, P_lid, df_melt_dP): # aka phi
    
    P_melt_capped = min(P_melt, P_dense)
    
    return 0.5 * (P_melt - P_lid) * df_melt_dP if (P_melt_capped > P_lid) else 0

def _F_melt(kappa, d_man, Rai, theta, R, f_melt, d_melt, d_dense, d_lid): # aka vol_flux
    
    vi_dim, F_melt = None, None
    d_melt_capped = min(d_melt,d_dense)
    if (F_melt_flag == 1):
        vi_dim = (kappa/d_man) * 0.05 * (Rai/theta)**(2/3)
        F_melt = 17.8 * pi * R**2 * vi_dim * f_melt * (d_melt_capped - d_lid) * (1/d_man)
    elif (F_melt_flag == 2):
        # This if for making the upwelling area 1/2 the radius of the cylindrical convection cell
        vi_dim = (kappa/d_man) * 0.05 * (Rai/theta)**(2/3)
        F_melt = 17.8 * pi * R**2 * vi_dim * f_melt * (d_melt_capped - d_lid) * (1/d_man) * (1/2)
    else:
        vi_dim = (kappa/d_man) * 0.40 * (Rai/theta)**(1/2)
        F_melt = 17.8 * pi * R**2 * vi_dim * f_melt * (d_melt_capped - d_lid) * (1/d_man)
    
    return F_melt

def _d_carb(d_crust, k, T_crust, T_surface, x_crust, A, rho0_man, g, B, d_lid):
    
    d_carb = None
    if (d_crust > 0) and (x_crust > 0):
        d_carb = (d_crust/2) + (k * (T_crust-T_surface))/(d_crust * x_crust) - (A * rho0_man * g * k)/x_crust - (k/x_crust)*np.sqrt(((x_crust * d_crust)/(2*k) + (T_crust-T_surface)/d_crust - A*rho0_man*g)**2 + (2*x_crust)/k*(T_surface - B))
    else:
        d_carb = d_lid + 1000
    if (np.imag(d_carb) != 0): d_carb = d_lid + 1000
    
    return d_carb

def _M_atm(P_surface, SA, g):
    return P_surface*SA/g

def _P_surface(M_atm, SA, g):
    return M_atm*g/SA

def _P(rho, g, h):
    return rho*g*h



## Universal Constants (that don't change from planet to planet) ##



pi = np.pi
G = 6.67e-11
Rg = 8.314
k_b = 1.38e-23
AU = 1.496e11
year = 3600*24*365.2425
Gyr = 1e9*year
m_H = 1.67e-27
m_H2O = 18*m_H
m_EO = 1.42e21
m_bar_co2 = 44/1000 
m_molecule_co2 = 7.3065e-26
AMU = 1.66054e-27
P2W = {'U238':9.17e-5, 'U235':5.75e-4, 'Th':2.56e-5, 'K':2.97e-5}
tau_ = {'U238':4.46e9, 'U235':7.04e8, 'Th':1.4e10, 'K':1.26e9}
lambd = {key:-val/np.log(1/2) for key, val in tau_.items()}
tau = {key:val*year for key, val in lambd.items()}

k_mdwarf = -1.347
Ro_s = 0.155
log_R_s = -4.436

F_melt_flag = 1
gamma_ad = 2e-8
weath_demand = 10*(0.19+0.2+0.145+0.045+0.0017)
P_dense = 10e9
F_offset = 1e14/year

SUN = {'M':1.989e30, 'R':6.957e8, 'L':3.828e26}
EARTH = {'M':5.97e24, 'R':6378100, 'T_surface':285, 'U238':0.022*0.9927, 'U235':0.022*0.0072, 'Th':0.083, 'K':261*(0.0117/100), 'F_degas':6e12/year}
EARTH['SA'] = _SA(EARTH['R'])
EARTH['g'] = _g(EARTH['M'], EARTH['R'])

# Values from Dong et al. (2018)
TRAPPIST_1c_dMdt_atm_species = {'O+':1.54e27*15.999*AMU, 'O2+':1.38e26*31.999*AMU, 'CO2+':1.32e26*44.01*AMU}
TRAPPIST_1c_dMdt_atm = sum(TRAPPIST_1c_dMdt_atm_species.values())

starInputQuantities = ['name','M_','R_','T','age_','type','dMdt_model','dMdt_boost']
inputQuantities = ['name','M_','R_','a_','init_C_man','mu_ref','init_T_man','HPE_','tidal_heating','init_P_surface','init_cumulative_V_crust','init_V_crust','init_C_crust','init_Q_U238_crust','init_Q_U235_crust','init_Q_Th_crust','init_Q_K_crust','Ev','alpha','kappa','df_melt_dP','Cp','k','C1','T_surface','L_m','f_C_trapped','F_melt_plume','rho_melt','DT_sol_dep','rho_man','rho0_man','dT_dP_man']
evolvedQuantities = ['T_man', 'cumulative_V_crust', 'Q_U238_crust', 'C_man', 'V_crust', 'C_crust', 'Q_U238', 'd_lid', 'Q_U235_crust', 'Q_U235', 'Q_Th_crust', 'Q_Th', 'Q_K_crust', 'Q_K', 'M_atm']
helperQuantities = ['V_man', 'd_crust', 'SA_man_wo_crust', 'SA_man_wo_lid', 'V_man_wo_lid', 'Rai', 'theta', 'F_man', 'T_lid', 'Q_crust', 'Q_man', 'x_crust', 'x_man', 'T_crust', 'P_lid', 'd_melt', 'P_melt', 'f_melt', 'F_melt', 'f_2', 'f_U', 'f_Th', 'f_K', 'd_carb', 'V_volc', 'F_dcarb', 'F_degas', 'F_weath_SL', 'P_surface', 'outgassing_rate']

np.random.seed()
warnings.filterwarnings('ignore', message='Creating an ndarray')



## HELPER FUNCTIONS ##



def computeHelperQuantities(planet, *Y):
    
    """
    Computes and returns a dictionary of quantities useful in Planet.evolve()
    planet : Planet object
    Y : Array of values of quantities tracked by differential equation.
    """

    P = planet
    hQ = {}

    T_man = Y[0]
    C_man = Y[3]
    V_crust = Y[4]
    C_crust = Y[5]
    d_lid = Y[7]
    M_atm = Y[14]

    hQ['V_man'] = V_man = _V(P.R) - _V(P.R_core) - V_crust
    hQ['d_crust'] = d_crust = P.R - (P.R**3 - (V_crust*3)/(4*pi))**(1/3)
    
    # Surface area of mantle (excluding crust on top of mantle)
    hQ['SA_man_wo_crust'] = SA_man_wo_crust = _SA(P.R - d_crust)
    
    # Surface area of actively convecting mantle
    hQ['SA_man_wo_lid'] = SA_man_wo_lid = _SA(P.R - d_lid)
    
    # Volume of actively convecting mantle 
    hQ['V_man_wo_lid'] = V_man_wo_lid = _V(P.R - d_lid) - _V(P.R_core)
    
    # Calculate thermal boundary layer thickness
    # Define rayleigh number & theta
    hQ['Rai'] = Rai = _Rai(P.rho_man, P.g, P.alpha, T_man, P.T_surface, P.d_man, P.kappa, _mu_i(P.mun, P.Ev, T_man))
    hQ['theta'] = theta = _theta(P.Ev, T_man, P.T_surface)
    
    # Calculate heat flux from convection to base of lid 
    hQ['F_man'] = F_man = _Q_lid(P.C1, P.k, T_man, P.T_surface, P.d_man, theta, Rai)
    
    # Temp at base of the lid 
    hQ['T_lid'] = T_lid = _T_lid(P.Ev, T_man)
    
    # Concentration of heat producing elements (watts per meter cubed) in crust and mantle
    hQ['Q_crust'] = Q_crust = Y[2] + Y[8] + Y[10] + Y[12]
    hQ['Q_man'] = Q_man = Y[6] + Y[9] + Y[11] + Y[13]
    hQ['x_crust'] = x_crust = 0 if (V_crust==0) else Q_crust/V_crust
    hQ['x_man'] = x_man = Q_man/V_man
    
    # Calculate temperature at base of crust, T_crust
    hQ['T_crust'] = T_crust = _T_crust(P.T_surface, T_lid, d_lid, d_crust, x_crust, x_man, P.k)
    
    hQ['P_lid'] = P_lid = _P(P.rho0_man, P.g, d_lid)
    hQ['d_melt'] = d_melt = _d_melt(T_man, P.DT_sol_dep, d_crust, P.crit_d_crust, P.dT_dP_man, P.rho0_man, P.g)
    hQ['P_melt'] = P_melt = _P(P.rho0_man, P.g, d_melt)
    hQ['f_melt'] = f_melt = _f_melt(P_melt, P_lid, P.df_melt_dP)
    
    hQ['F_melt'] = F_melt = _F_melt(P.kappa, P.d_man, Rai, theta, P.R, f_melt, d_melt, P.d_dense, d_lid)
    
    hQ['f_2'] = f_2 = 1 - (1 - f_melt)**(1/1e-4)
    hQ['f_U'] = f_U = 1 - (1 - f_melt)**(1/0.0012) # Numbers from Beattie 1993
    hQ['f_Th'] = f_Th = 1 - (1 - f_melt)**(1/0.00029) # From Beattie 1993
    hQ['f_K'] = f_K = 1 - (1 - f_melt)**(1/0.0011) # From Hart & Brooks 1974, using 60% Ol & 40%  px   
    
    # This is based on Earth's gravity and is written in terms of depth. Need to convert to deal with changing g
    A = 3.125e-3/(P.rho0_man*9.8)
    B = 835.5
    hQ['d_carb'] = d_carb = _d_carb(d_crust, P.k, T_crust, P.T_surface, x_crust, A, P.rho0_man, P.g, B, d_lid)
    hQ['V_volc'] = V_volc = V_crust if (d_crust<d_carb) else _V(P.R) - _V(P.R-d_carb)
    
    hQ['F_dcarb'] = F_dcarb = (1-P.f_C_trapped) * (C_crust/V_volc) * (F_melt/2) * (np.tanh((d_crust - d_carb) * 20) + 1) if (V_crust>0) else 0
    hQ['F_degas'] = F_degas = (f_2/f_melt) * F_melt * C_man/V_man if (f_melt>0) else 0
    hQ['F_weath_SL'] = F_weath_SL = 0.1 * F_melt * P.rho0_man * weath_demand
    
    hQ['P_surface'] = _P_surface(M_atm, P.SA, P.g)
    hQ['outgassing_rate'] = F_degas*m_bar_co2

    return hQ



def initialize_d_lid(P, T_man):

    init_V_man = _V(P.R) - _V(P.R_core)
    init_M_man = init_V_man * P.rho_man
    init_Q = _Q(init_M_man, P.abundances)
    init_Q_tot = sum(init_Q.values())
    init_mu_i = P.mun*np.exp(P.Ev/(Rg*T_man))
    init_Rai = _Rai(P.rho_man, P.g, P.alpha, T_man, P.T_surface, P.d_man, P.kappa, init_mu_i)
    init_theta = _theta(P.Ev, T_man, P.T_surface)
    init_Q_lid = _Q_lid(P.C1, P.k, T_man, P.T_surface, P.d_man, init_theta, init_Rai)
    init_T_lid = _T_lid(P.Ev, T_man)
    init_x_m = init_Q_tot/init_V_man  
    func = lambda x : init_T_lid - P.T_surface - init_Q_lid*x/P.k - init_x_m*x**2/P.k + init_x_m*x**2/(2*P.k)
    
    return fsolve(func, 1e5)[0]



def initializeTheRest(P):

    """
    Planet properties that are calculated strictly as a consequence of input parameters.
    - P : Planet object initialized with input parameters.
    """

    P.M = P.M_*EARTH['M']
    P.R = P.R_*EARTH['R']
    P.a = P.a_*AU

    P.SA = _SA(P.R)
    P.g = _g(P.M, P.R)
    P.d_man = _d_man(P.M)
    P.R_core = P.R - P.d_man
    P.crit_d_crust = 0.2*(_V(P.R) - _V(P.R_core))/P.SA
    P.mun = _mun(P.mu_ref, P.Ev)
    P.d_dense = P_dense/(P.rho0_man * P.g)

    P.U238 = P.HPE_*EARTH['U238']
    P.U235 = P.HPE_*EARTH['U235']
    P.Th = P.HPE_*EARTH['Th']
    P.K = P.HPE_*EARTH['K']
    P.abundances = {'U238':P.U238, 'U235':P.U235, 'Th':P.Th, 'K':P.K}
    P.abundances_ = {key:val/EARTH[key] for key, val in P.abundances.items()}
    init_V_man = _V(P.R) - _V(P.R_core)
    init_M_man = init_V_man * P.rho_man
    P.init_Q = _Q(init_M_man, P.abundances)
    P.init_Q_U238 = P.init_Q['U238']
    P.init_Q_U235 = P.init_Q['U235']
    P.init_Q_Th = P.init_Q['Th']
    P.init_Q_K = P.init_Q['K']

    P.init_d_lid = initialize_d_lid(P, P.init_T_man)
    P.init_M_atm = _M_atm(P.init_P_surface, P.SA, P.g)

    P.y0 = [P.init_T_man, 
            P.init_cumulative_V_crust, 
            P.init_Q_U238_crust, 
            P.init_C_man, 
            P.init_V_crust, 
            P.init_C_crust, 
            P.init_Q_U238, 
            P.init_d_lid, 
            P.init_Q_U235_crust, 
            P.init_Q_U235, 
            P.init_Q_Th_crust, 
            P.init_Q_Th, 
            P.init_Q_K_crust, 
            P.init_Q_K, 
            P.init_M_atm]

    return



def initializeTheRestStar(S):

    """
    Star properties that are calculated strictly as a consequence of input parameters.
    - S : Star object initialized with input parameters.
    """

    S.M = S.M_*SUN['M']
    S.R = S.R_*SUN['R']
    S.age = S.age_*Gyr
    S.L_X = _L_X(S.age_)

    return



def _L_X(age):
    # Magaudda 2020
    Prot = (age - 0.012)/(0.061)
    if Prot<=0.1:
        L_X = (10**28.54)*0.1**-0.19 ##To prevent complex numbers
    elif Prot<=33.7:
        L_X = (10**28.54)*Prot**-0.19
    else:
        C_unsat = (10**28.54*33.7**-0.19)/(33.7**-3.52)
        L_X = C_unsat*Prot**-3.52
    return L_X/10**7



def _L_EUV(age,mass):
    # Sreejith 2020
    Ro = (age - 0.012)/(0.061*10**(2.33-1.5*mass+0.31*mass**2))
    if Ro<Ro_s:
        log_R_HK = log_R_s
    if Ro>=Ro_s:
        log_R_HK = k_mdwarf*np.log(Ro)+log_R_s-k_mdwarf*np.log(Ro_s)
    L_EUV_to_L_bol = 10**(1.20*log_R_HK+2.23)
    L_bol = 10**(1.87147773*np.log10(mass)+32.54209892) #From Mann 2015, Boudreaux 2022, SIMBAD, and my fit
    L_EUV = L_EUV_to_L_bol*L_bol
    return L_EUV/10**7



def _L_XUV(age,mass):
    return _L_X(age) + _L_EUV(age,mass)



def _dMdt_star(age):
    
    F_X = (_L_X(age)*10**7)/(4*np.pi*(0.1192*SUN['R']*100)**2)
    dMdt = (10**(0.77*np.log10(F_X) + -3.42))*(1.26e9*0.1192**2)
    
    return dMdt



def _dMdt_atm(R_, a_, dMdt_model, t, dMdt_boost=1):
    
    dMdt_star = 2.6e8 # Dong
    if dMdt_model == 'Garraffo': dMdt_star = 1.89e9
    if dMdt_model == 'Variable': dMdt_star = _dMdt_star(t/Gyr)
    
    return TRAPPIST_1c_dMdt_atm * (R_/1.097)**2 * (0.01580/a_)**2 * (dMdt_star/2.6e8) * dMdt_boost



## STAR AND PLANET OBJECTS ##



class Star():



    def __init__(self, filename=None, M_=0.0898, R_=0.1192, T=2566, age_=7.6, name='TRAPPIST_1', **kwargs):
        
        if filename is None:
            
            self.M_ = M_
            self.R_ = R_
            self.T = T
            self.age_ = age_
            self.name = name

            self.dMdt_model = kwargs.pop('dMdt_model', 'Dong')
            self.dMdt_boost = kwargs.pop('dMdt_boost', 1)
            self.type = kwargs.pop('type', 'M8V')
        
        else:
            self.load(filename, **kwargs)
        
        initializeTheRestStar(self)

        return



    def save(self, params=[], folder=parent_dir, **kwargs):
        
        filename = kwargs.pop('filename', self.name)
        
        d = {v:[getattr(self, v)] for v in starInputQuantities if v not in params}
        df = pd.DataFrame.from_dict(d)
        df.to_csv(os.path.join(folder, filename + '_constants.csv'))

        return



    def load(self, filename, folder=parent_dir):
        
        df = pd.read_csv(os.path.join(folder, filename + '_constants.csv'))
        d = df.to_dict(orient='list')
        for key, val in d.items(): setattr(self, key, val[0])

        return



class Planet():



    def __init__(self, filename=None, M_=1.3105, R_=1.097, a_=0.01580, name='TRAPPIST_1c', **kwargs):
        
        if filename is None:
            
            self.M_ = M_
            self.R_ = R_
            self.a_ = a_
            self.name = name

            self.init_C_man = kwargs.pop('init_C_man', 2.5e21)
            self.mu_ref = kwargs.pop('mu_ref', 1e21)
            self.init_T_man = kwargs.pop('init_T_man', 2000)  
            self.HPE_ = kwargs.pop('HPE_', 1)
            self.tidal_heating = kwargs.pop('tidal_heating', False) 

            dMdt_model = kwargs.pop('dMdt_model', 'Dong')
            dMdt_boost = kwargs.pop('dMdt_boost', 1)
            self.star = kwargs.pop('star', Star(dMdt_model=dMdt_model, dMdt_boost=dMdt_boost)) 

            self.init_P_surface = kwargs.pop('init_P_surface', 0)
            self.init_cumulative_V_crust = kwargs.pop('init_cumulative_V_crust', 0)
            self.init_V_crust = kwargs.pop('init_V_crust', 0)
            self.init_C_crust = kwargs.pop('init_C_crust', 0)
            self.init_Q_U238_crust = kwargs.pop('init_Q_U238_crust', 0)
            self.init_Q_U235_crust = kwargs.pop('init_Q_U235_crust', 0)
            self.init_Q_Th_crust = kwargs.pop('init_Q_Th_crust', 0)
            self.init_Q_K_crust = kwargs.pop('init_Q_K_crust', 0)

            self.Ev = kwargs.pop('Ev', 300000.0)
            self.alpha = kwargs.pop('alpha', 3e-5)
            self.kappa = kwargs.pop('kappa', 1e-6)
            self.df_melt_dP = kwargs.pop('df_melt_dP', 0.15e-9)
            self.Cp = kwargs.pop('Cp', 1250)
            self.k = kwargs.pop('k', 5)
            self.C1 = kwargs.pop('C1', 0.5)
            self.T_surface = kwargs.pop('T_surface', 273.0)
            self.L_m = kwargs.pop('L_m', 600*1e3) 
            self.f_C_trapped = kwargs.pop('f', 0)
            self.F_melt_plume = kwargs.pop('F_melt_plume', 0)
            self.rho_melt = kwargs.pop('rho_melt', 2800)
            self.DT_sol_dep = kwargs.pop('DT_sol_dep', 150)

            self.rho_man = kwargs.pop('rho_man', 5380)
            self.rho0_man = kwargs.pop('rho0_man', 3300.0)
            self.dT_dP_man = kwargs.pop('dT_dP_man', 2e-8)

            self.evolved = False
  
        else:
            self.load(filename, **kwargs)
            
        initializeTheRest(self)
        
        return
    
        
    
    def evolve(self, t_end=10*Gyr, N=500, plot=None, **kwargs):
        
        def func(t, Y):
            
            """
            Equivalent of 'thermal_evol_delam_melt_full_Qpartition_atm2_dep_LHS3844b' from MATLAB code.
            
            Solves 14 coupled ordinary differential equations for evolution of mantle
            temperature, crust thickness, heat budget, and CO2 in the atmosphere and
            mantle
            Y[0] is mantle temperature
            Y[1] is the cummulative volume of the crust
            Y[2] is the heat production rate in the crust from U238
            Y[3] is the mantle CO2 reservoir size (in moles)
            Y[4] is the volume of crust present at any one time
            Y[5] is the crust CO2 reservoir size (moles)
            Y[6] is the heat production rate in the mantle from U238
            Y[7] is the thickness of the lithosphere
            Y[8] is the heat production rate in the crust from U235
            Y[9] is the heat production rate in the mantle from U235
            Y[10] is the heat production rate in the crust from Th
            Y[11] is the heat production rate in the mantle from Th
            Y[12] is the heat production rate in the crust from K
            Y[13] is the heat production rate in the mantle from K
            Y[14] is the mass of the atmosphere (entirely CO2)
            """
            
            dYdt = np.zeros(15)
            
            T_man, cumulative_V_crust, Q_U238_crust, C_man, V_crust, C_crust, Q_U238, d_lid, Q_U235_crust, Q_U235, Q_Th_crust, Q_Th, Q_K_crust, Q_K, M_atm = Y
            
            hQ = computeHelperQuantities(self, *Y)
            V_man = hQ['V_man']
            d_crust = hQ['d_crust']
            SA_man_wo_lid = hQ['SA_man_wo_lid']
            V_man_wo_lid = hQ['V_man_wo_lid']
            F_man = hQ['F_man']
            T_lid = hQ['T_lid']
            Q_man = hQ['Q_man']
            x_crust = hQ['x_crust']
            x_man = hQ['x_man']
            T_crust = hQ['T_crust']
            P_melt = hQ['P_melt']
            f_melt = hQ['f_melt']
            F_melt = hQ['F_melt']
            f_2 = hQ['f_2']
            f_U = hQ['f_U']
            f_Th = hQ['f_Th']
            f_K = hQ['f_K']
            d_carb = hQ['d_carb']
            
            # How does the mantle temperature change?
            Q_tidal_heating = 0.62*self.SA if self.tidal_heating else 0
            dYdt[0] = (Q_man + Q_tidal_heating)/(V_man_wo_lid*self.rho_man*self.Cp) - (F_man * SA_man_wo_lid)/(V_man_wo_lid*self.rho_man*self.Cp) - (F_melt + self.F_melt_plume)*self.rho_melt*(self.L_m + (T_man - self.T_surface - P_melt*gamma_ad) * self.Cp)/(V_man_wo_lid*self.rho_man*self.Cp)

            # How does the thickness of the lithosphere change?
            if (T_lid > T_crust+1):
                dYdt[7] = (1/(self.rho_man*self.Cp*(T_man - T_lid))) * (-F_man - d_lid*x_man + (self.k*(T_lid-T_crust))/(d_lid-d_crust) + x_man*(d_lid**2 - d_crust**2)/(2*(d_lid - d_crust)))
            else:
                dYdt[7] = (1/(self.rho_man*self.Cp*(T_man - T_lid))) * (-F_man - d_crust*x_crust*(1/2) + (self.k*(T_lid-self.T_surface))/d_crust)
            
            # How does the cumulative volume of the crust change?
            dYdt[1] = F_melt + self.F_melt_plume
            
            # How does the volume of the crust change?
            dYdt[4] = (F_melt + self.F_melt_plume) - (F_melt + self.F_melt_plume - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1)
            
            # How does the heat production in the mantle and crust change?
            if (V_crust > 0) and (F_melt > 0):
                
                dYdt[2] = (Q_U238/V_man) * (F_melt/f_melt) * f_U - (Q_U238_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - Q_U238_crust/tau['U238']  
                dYdt[8] = (Q_U235/V_man) * (F_melt/f_melt) * f_U - (Q_U235_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - Q_U235_crust/tau['U235'] 
                dYdt[10] = (Q_Th/V_man) * (F_melt/f_melt) * f_Th - (Q_Th_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - Q_Th_crust/tau['Th'] 
                dYdt[12] = (Q_K/V_man) * (F_melt/f_melt) * f_K - (Q_K_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - Q_K_crust/tau['K']
            
                dYdt[6] = (Q_U238_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - (Q_U238/V_man) * (F_melt/f_melt) * f_U - Q_U238/tau['U238'] 
                dYdt[9] = (Q_U235_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - (Q_U235/V_man) * (F_melt/f_melt) * f_U - Q_U235/tau['U235'] 
                dYdt[11] = (Q_Th_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - (Q_Th/V_man) * (F_melt/f_melt) * f_Th - Q_Th/tau['Th']           
                dYdt[13] = (Q_K_crust/V_crust) * (F_melt - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) - (Q_K/V_man) * (F_melt/f_melt) * f_K - Q_K/tau['K']     
            
            else:
                dYdt[2] = -Y[2]/tau['U238']
                dYdt[8] = -Y[8]/tau['U235']
                dYdt[10] = -Y[10]/tau['Th']
                dYdt[12] = -Y[12]/tau['K']
                dYdt[6] = -Y[6]/tau['U238']
                dYdt[9] = -Y[9]/tau['U235']
                dYdt[11] = -Y[11]/tau['Th']
                dYdt[13] = -Y[13]/tau['K']
            
            # How does the carbon in the mantle change?
            if (d_carb < d_crust) and (V_crust > 0) and (F_melt > 0):   
                
                dYdt[3] =-(C_man/V_man) * (F_melt + self.F_melt_plume) * (f_2/f_melt) + (C_crust/V_crust) * (F_melt + self.F_melt_plume - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1) * self.f_C_trapped    
            
            elif (V_crust > 0) and (F_melt > 0):
                
                dYdt[3] =-(C_man/V_man) * (F_melt + self.F_melt_plume) * (f_2/f_melt) + (C_crust/V_crust) * (F_melt + self.F_melt_plume - SA_man_wo_lid*min(0,dYdt[7])) * (np.tanh((d_crust-d_lid) * 20) + 1)
                
            else:
                
                dYdt[3] = 0
            
            # How does the carbon in the crust change?
            dYdt[5] = 0  # (No weathering)

            # How does the mass of the atmosphere change?
            dM_lost_dt = _dMdt_atm(self.R_, self.a_, self.star.dMdt_model, t, dMdt_boost=self.star.dMdt_boost)
            
            if M_atm < dM_lost_dt*self.max_step:
                dM_lost_dt = Y[14]/self.max_step
            
            dM_added_dt = -dYdt[3]*m_bar_co2
            dYdt[14] = -dM_lost_dt + dM_added_dt
            
            return dYdt
        
        def event_F_degas_100(t, Y):
            
            """
            Time when degassing first falls below modern Earth rate
            """
            
            hQ = computeHelperQuantities(self, *Y)
            F_dcarb = hQ['F_dcarb']
            F_degas = hQ['F_degas']
            
            return F_degas + F_dcarb - EARTH['F_degas']*(self.SA/EARTH['SA'])

        def event_F_degas_10(t, Y):
            
            """
            Time when degassing first falls below 10 % modern Earth rate
            """
            
            hQ = computeHelperQuantities(self, *Y)
            F_dcarb = hQ['F_dcarb']
            F_degas = hQ['F_degas']
            
            return F_degas + F_dcarb - 0.1*EARTH['F_degas']*(self.SA/EARTH['SA'])

        def event_F_degas_0(t, Y):
            
            """
            Time when degassing ends
            """
            
            hQ = computeHelperQuantities(self, *Y)
            F_dcarb = hQ['F_dcarb']
            F_degas = hQ['F_degas']
            
            return F_degas + F_dcarb

        def event_weathering_SL(t, Y):
            
            """
            Time when weathering becomes supply limited.
            """
            
            hQ = computeHelperQuantities(self, *Y)
            F_dcarb = hQ['F_dcarb']
            F_degas = hQ['F_degas']
            F_weath_SL = hQ['F_weath_SL']
            
            return F_degas + F_dcarb - F_weath_SL - F_offset
        
        event_F_degas_100.terminal = False
        event_F_degas_10.terminal = False
        event_F_degas_0.terminal = False
        event_weathering_SL.terminal = False
        event_F_degas_100.direction = -1
        event_F_degas_10.direction = -1
        event_F_degas_0.direction = -1
        event_weathering_SL.direction = 1
        events = [event_F_degas_100, event_F_degas_10, event_F_degas_0, event_weathering_SL]

        self.t_span = [0,t_end]
        self.max_step = (10*Gyr)/10000
        t_eval = np.linspace(0,t_end,N)
        self.dt = t_end/N

        sol = solve_ivp(func, self.t_span, self.y0, method='RK45', events=events, rtol=1e90, atol=1e90, max_step=self.max_step, first_step=self.max_step, t_eval=t_eval, **kwargs)
        self.sol = sol
        assert self.sol.success, f"INTEGRATION FAILED: {self.sol.message}"
        
        self.t = sol.t
        self.t_gyr = self.t/Gyr
        
        for i, q in enumerate(evolvedQuantities): setattr(self, q, sol.y[i])
        self.y = {q:getattr(self, q) for q in evolvedQuantities}

        vchQ = np.vectorize(computeHelperQuantities)
        hQ = vchQ(self, *sol.y)
        hQ = {key:np.array([d[key] for d in hQ]) for key in hQ[0]}
        for key, val in hQ.items(): setattr(self, key, val)
        
        t_events = [(t[0]/Gyr if len(t)>0 else 999) for t in sol.t_events]
        self.F_degas_100, self.F_degas_10, self.F_degas_0, self.weathering_SL = t_events
        self.evolved = True
        
        if plot is not None:
            fig, ax = plt.subplots(dpi=200)
            if isinstance(plot, str): plot = [plot]
            for q in plot: self.plot(q, ax=ax, label=QU(q))
            if len(plot)>1: 
                ax.set_ylabel('Multiple Quantities')
                ax.legend()
        
        return
    


    def plot(self, q, ax=None, label=None, **kwargs):
        
        if ax is None: fig, ax = plt.subplots(dpi=200)
        ax.plot(self.t_gyr, getattr(self, q), label=label)
        ax.set(xlabel=QU('t_gyr'), ylabel=QU(q), **kwargs)
        ax.grid()

        return



    def save(self, star=True, constants=True, evolution=True, params=[], folder=parent_dir, **kwargs):
        
        filename = kwargs.pop('filename', self.name)

        if star:
            self.star.save(folder=folder, filename=filename+'_star', params=params)

        if constants:
            d = {v:[getattr(self, v)] for v in inputQuantities if v not in params}
            df = pd.DataFrame.from_dict(d)
            df.to_csv(os.path.join(folder, filename + '_constants.csv'))

        if evolution and self.evolved:
            d = {}
            d['t_gyr'] = self.t_gyr
            for key, val in self.y.items(): d[key] = val
            df = pd.DataFrame.from_dict(d)
            df.to_csv(os.path.join(folder, filename + '_evolution.csv'), index=False)

        return



    def load(self, filename, star=True, constants=True, evolution=True, folder=parent_dir, **kwargs):
        
        i = kwargs.pop('i', None) # Use 'i' to signal retrieval from suite and give index of simulation.
        
        if star: 
            if i is None:
                self.star = Star(filename + '_star', folder=folder)
            else:
                self.star = Star('_star', folder=folder)

        if constants:
            df = None
            if i is None:
                df = pd.read_csv(os.path.join(folder, filename + '_constants.csv'))
            else:
                df = pd.read_csv(os.path.join(folder, '_constants.csv'))
            d = df.to_dict(orient='list')
            for key, val in d.items(): setattr(self, key, val[0])

        if i is not None:
            df = pd.read_csv(os.path.join(folder, 'Summary.csv'))
            d = df.to_dict(orient='list')
            for key, val in d.items(): setattr(self, key, val[i])

        initializeTheRest(self)

        if evolution:
            if i is not None: filename = f'{i:07d}'

            df = pd.read_csv(os.path.join(folder, filename + '_evolution.csv'))
            d = df.to_dict(orient='list')
            for key, val in d.items(): setattr(self, key, np.array(val))

            self.t = self.t_gyr * Gyr
            self.y = {q:getattr(self, q) for q in evolvedQuantities}

            self.Q_man = self.Q_U238 + self.Q_U235 + self.Q_Th + self.Q_K
            self.Q_crust = self.Q_U238_crust + self.Q_U235_crust + self.Q_Th_crust + self.Q_K_crust
            self.P_surface = _P_surface(self.M_atm, self.SA, self.g)

            vchQ = np.vectorize(computeHelperQuantities)
            hQ = vchQ(self, *[getattr(self, q) for q in evolvedQuantities])
            hQ = {key:np.array([d[key] for d in hQ]) for key in hQ[0]}
            for key, val in hQ.items(): setattr(self, key, val)

            self.outgassing_rate = self.F_degas*m_bar_co2

        return



## GET DATA ##



def runOne(savekws={}, **inputs):

    p = Planet(**inputs)
    p.evolve()
    p.save(params=inputs.keys(), **savekws)

    P_surface = CS(p.t_gyr, p.P_surface)
    t_max = fmin(lambda t: -P_surface(t), p.t_gyr[np.argmax(p.P_surface)], disp=False)[0]
    
    return P_surface([t_max, 5.4, 7.6, 9.8])



def suite(name='Suite', folder=parent_dir, product=True, **inputs):

    folder = os.path.join(folder, name)
    os.mkdir(folder)

    for key, val in inputs.items():
        if not isinstance(val, (list,np.ndarray,tuple)): inputs[key] = [val]

    runOne(savekws={'filename':'', 'folder':folder, 'evolution':False}, **{key:val[0] for key, val in inputs.items()})

    combos = []
    if product: 
        combos = list(itertools.product(*inputs.values()))
    else:
        combos = np.hstack((np.array([val]).T for val in inputs.values()))
    df = pd.DataFrame(data=combos, columns=inputs.keys())
    df['P_surface Max'] = np.zeros(len(combos))
    df['P_surface @ 5.4 Gyr'] = np.zeros(len(combos))
    df['P_surface @ 7.6 Gyr'] = np.zeros(len(combos))
    df['P_surface @ 9.8 Gyr'] = np.zeros(len(combos))

    if __name__=='__main__':
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fs = []
            for i, combo in enumerate(combos):
                thisInput = {key:combo[i] for i, key in enumerate(inputs.keys())}
                fs.append(executor.submit(runOne, **thisInput, savekws={'filename':f'{i:07d}', 'evolution':True, 'star':False, 'constants':False, 'folder':folder}))
                
            for i, f in enumerate(fs):
                special_P_surface_vals = f.result()
                df.at[i, 'P_surface Max'] = special_P_surface_vals[0]
                df.at[i, 'P_surface @ 5.4 Gyr'] = special_P_surface_vals[1]
                df.at[i, 'P_surface @ 7.6 Gyr'] = special_P_surface_vals[2]
                df.at[i, 'P_surface @ 9.8 Gyr'] = special_P_surface_vals[3]
    
    df.to_csv(os.path.join(folder, 'Summary.csv'))

    return



def load(suite, q=None, folder=parent_dir, **which):

    df = pd.read_csv(os.path.join(folder, suite, 'Summary.csv'))

    for key, val in which.items():
        if not isinstance(val, (list,np.ndarray,tuple)): which[key] = [val]

    where = np.array([True]*len(df))
    for key, val in which.items():
        temp_where = np.array([False]*len(df))
        for v in val:
            temp_where = temp_where | (np.array(getattr(df, key)) == val)
        where = where & temp_where
    
    infodict = {s:np.array(getattr(df, s))[where] for s in ['index', *df.columns[1:-4]]}

    data = None

    if q in df.columns: 
        data = np.array(getattr(df, q))[where]
        
    elif q in evolvedQuantities:
        for j, i in enumerate(infodict['index']):
            evolution_df = pd.read_csv(os.path.join(folder, suite, f'{i:07d}_evolution.csv'))
            if j==0: data = np.zeros((np.sum(where), len(evolution_df)))
            data[j] = np.array(getattr(evolution_df, q))
            
    elif q in helperQuantities:
        for j, i in enumerate(infodict['index']):
            p = Planet('', i=i, folder=os.path.join(folder, suite))
            if j==0: data = np.zeros((np.sum(where), len(p.t)))
            data[j] = getattr(p, q)

    else:
        data = np.empty(np.sum(where), dtype=object)
        for j, i in enumerate(infodict['index']):
            data[j] = Planet('', i=i, folder=os.path.join(folder, suite))

    return data, infodict



## EXPERIMENTATION ##



def TRAPPIST_1_planets_pCO2(a):

    TRAPPIST_1c = Planet()
    TRAPPIST_1c_dMdt_atm = _dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Garraffo', TRAPPIST_1c.star.age_)
    TRAPPIST_1c_mCO2 = (0.1e5*TRAPPIST_1c.SA)/(TRAPPIST_1c.g)
    mCO2_0 = TRAPPIST_1c_mCO2+(TRAPPIST_1c_dMdt_atm*TRAPPIST_1c.star.age)

    TRAPPIST_1p_dMdt_atm = TRAPPIST_1c_dMdt_atm*(TRAPPIST_1c.a_/a)**2
    TRAPPIST_1p_mCO2 = mCO2_0 - TRAPPIST_1p_dMdt_atm*TRAPPIST_1c.star.age
    TRAPPIST_1p_pCO2 = (TRAPPIST_1p_mCO2*TRAPPIST_1c.g)/(TRAPPIST_1c.SA)

    return TRAPPIST_1p_pCO2/1e5



def plotHeatingComponents(P, **kwargs):

    fig, ax = plt.subplots(nrows=2, dpi=200, sharex=True, figsize=(4,6))

    ax[0].plot(P.t_gyr, P.T_man, 'ko-')

    denominator = P.V_man_wo_lid * P.rho_man * P.Cp
    #term1 = (P.Q_man + 0.62*P.SA)/denominator * P.dt
    term11 = P.Q_man/denominator * P.dt
    term12 = 0.62*P.SA/denominator * P.dt if P.tidal_heating else np.zeros_like(P.t_gyr)
    term2 = - (P.F_man * P.SA_man_wo_lid)/denominator * P.dt
    term3 = - (P.F_melt + P.F_melt_plume)*P.rho_melt*(P.L_m + (P.T_man-P.T_surface-P.P_melt*gamma_ad) * P.Cp)/denominator * P.dt
    
    ax[1].plot(P.t_gyr, term11, c='purple', marker='o', linestyle='-', label='Radiogenic Heating')
    ax[1].plot(P.t_gyr, term12, c='red', marker='o', linestyle='-', label='Tidal Heating')
    ax[1].plot(P.t_gyr, term2, c='C1', marker='o', linestyle='-', label='Conduction onto Lid')
    ax[1].plot(P.t_gyr, term3, c='C2', marker='o', linestyle='-', label='Erupting Magma')
    ax[1].plot(P.t_gyr, term11 + term12 + term2 + term3, c='k', marker='o', linestyle='-', label='Total')

    ax[0].set(ylabel='Mantle Temperature [K]', **kwargs)
    ax[1].set(xlabel='Time [Gyr]', ylabel=r'$\Delta T_{\rm{man}}$ in One Timestep [K]', **kwargs)
    ax[1].legend(fontsize='x-small', loc='upper right')
    
    return



def plotHPEbyLayer(P, **kwargs):

    fig, ax = plt.subplots(dpi=200)
    ax.plot(P.t_gyr, P.Q_crust, label='In Crust')
    ax.plot(P.t_gyr, P.Q_man, label='In Mantle')
    ax.set(xlabel='Time [Gyr]', ylabel='Radiogenic Heat Production [W]', **kwargs)
    ax.legend()

    return



def h_co2_drag_calculation(N_EO=1.0, T_upp_atm=300):

    M_H2O = N_EO*m_EO
    M_H = ((2*m_H)/m_H2O)*M_H2O
    N_H_i = M_H/m_H
    N_CO2_i = N_H_i*1e-2

    def b_H_CO2(T_upp_atm):
        return 8.4e19*T_upp_atm**0.6
    
    TRAPPIST_1c = Planet()

    beta = 1
    eta = 0.15
    Delta_phi = (G*TRAPPIST_1c.M)/TRAPPIST_1c.R

    def func(t, Y):

        F_XUV = _L_XUV(t/Gyr, TRAPPIST_1c.star.M_)/(4*np.pi*TRAPPIST_1c.a**2)
                
        dYdt = np.zeros(2)
                
        N_H, N_CO2 = Y

        f_CO2 = N_CO2/N_H

        F_H = ((beta**2*eta*F_XUV)/(4*Delta_phi) + (TRAPPIST_1c.g*m_molecule_co2*f_CO2*(m_molecule_co2-m_H)*b_H_CO2(T_upp_atm))/(k_b*T_upp_atm*(1+f_CO2))) / (m_H+m_molecule_co2*f_CO2)

        x_CO2 = 1 - (TRAPPIST_1c.g*(m_molecule_co2-m_H)*b_H_CO2(T_upp_atm))/(F_H*k_b*T_upp_atm*(1+f_CO2))

        F_CO2 = F_H*f_CO2*x_CO2

        dYdt[0] = -F_H*4*np.pi*TRAPPIST_1c.R**2
        dYdt[1] = -F_CO2*4*np.pi*TRAPPIST_1c.R**2

        return dYdt

    def event1(t, Y):

        return Y[0] 
    
    def event2(t, Y):

        return Y[1] 

    event1.terminal = True
    event1.direction = -1

    event2.terminal = True
    event2.direction = -1

    sol = solve_ivp(func, t_span=[0,7.6*Gyr], y0=[N_H_i, N_CO2_i], events=[event1, event2], rtol=1e45, atol=1e45, max_step = (7.6*Gyr)/10000000)

    co2_loss = (sol.y[1][0] - sol.y_events[0][0][1])*m_molecule_co2*(TRAPPIST_1c.g/TRAPPIST_1c.SA)*1e-5
    time_of_h_loss = sol.t_events[0][0]/Gyr

    return sol, co2_loss, time_of_h_loss

    

## FIGURES FOR PAPER ##



def figure1(save=False, **kwargs):

    '''
    Planet Context
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'01 - Planet Context')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    trappist_1_names = ['b', 'c', 'd', 'e', 'f', 'g', 'h']
    trappist_1_radii = np.array([1.116, 1.097, 0.778, 0.920, 1.045, 1.129, 0.775])
    trappist_1_mass = np.array([1.3771, 1.3105, 0.3885, 0.6932, 1.0411, 1.3238, 0.3261])
    trappist_1_semimajor =np.array([0.01154, 0.01580, 0.02227, 0.02925, 0.03849, 0.04683, 0.06189])
    trappist_1_incident_flux = (0.000553*SUN['L'])/(4*np.pi*(trappist_1_semimajor*AU)**2)
    trappist_1_esc_vel = np.sqrt((2*G*trappist_1_mass*EARTH['M'])/(trappist_1_radii*EARTH['R']))/1000

    df = pd.read_csv('JWST_Rocky_Planets.csv', sep=',')
    ep_names = df['Planet']
    ep_stellar_temp = df['Tstar']
    ep_radii = df['Rpl (Rearth)']
    ep_mass = df['Mpl (Mearth)']
    ep_incident_flux = df['Incident Flux']
    ep_esc_vel = np.sqrt((2*G*ep_mass*EARTH['M'])/(ep_radii*EARTH['R']))/1000

    earth_incident_flux = SUN['L']/(4*np.pi*AU**2)
    venus_incident_flux = SUN['L']/(4*np.pi*(0.723*AU)**2)
    mars_incident_flux = SUN['L']/(4*np.pi*(1.524*AU)**2)

    fig, ax = plt.subplots(ncols=2, figsize=(10,4), gridspec_kw={'width_ratios':[4,5]}, dpi=dpi)
    plt.subplots_adjust(wspace=0.25)
    cm = plt.cm.get_cmap('RdYlBu')

    sc = ax[0].scatter(trappist_1_esc_vel, trappist_1_incident_flux/earth_incident_flux, c=[2566]*7, vmin=2000, vmax=6000, cmap=cm, s=150, alpha=0.7, ec='gray')
    ax[0].scatter(np.sqrt((2*G*EARTH['M'])/(EARTH['R']))/1000, 1, c=5772, vmin=2000, vmax=6000, cmap=cm, s=150, alpha=0.7, ec='gray')
    ax[0].scatter(np.sqrt((2*G*0.815*EARTH['M'])/(0.949*EARTH['R']))/1000, venus_incident_flux/earth_incident_flux, c=5772, vmin=2000, vmax=6000, cmap=cm, s=150, alpha=0.7, ec='gray')
    ax[0].scatter(np.sqrt((2*G*0.107*EARTH['M'])/(0.532*EARTH['R']))/1000, mars_incident_flux/earth_incident_flux, c=5772, vmin=2000, vmax=6000, cmap=cm, s=150, alpha=0.7, ec='gray')
    for i in range(len(trappist_1_incident_flux)):
        ax[0].text(trappist_1_esc_vel[i], trappist_1_incident_flux[i]/earth_incident_flux, trappist_1_names[i], fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax[0].text(np.sqrt((2*G*EARTH['M'])/(EARTH['R']))/1000, 1, 'E', fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax[0].text(np.sqrt((2*G*0.815*EARTH['M'])/(0.949*EARTH['R']))/1000, venus_incident_flux/earth_incident_flux, 'V', fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax[0].text(np.sqrt((2*G*0.107*EARTH['M'])/(0.532*EARTH['R']))/1000, mars_incident_flux/earth_incident_flux, 'M', fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax[0].text(trappist_1_esc_vel[0], trappist_1_incident_flux[0]/earth_incident_flux, 'x', fontsize=30, color='black', alpha=0.6, verticalalignment='center', horizontalalignment='center')
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'Escape Velocity [$\rm{km}/\rm{s}$]')
    ax[0].set_ylabel(r'Incident Flux [$\rm{F}_\oplus$]')
    ax[0].set_xticks(np.arange(5,14))
    ax[0].set_xticklabels(np.arange(5,14))

    sc = ax[1].scatter(trappist_1_esc_vel, trappist_1_incident_flux/earth_incident_flux, c=[2566]*7, vmin=2000, vmax=6000, cmap=cm, s=150, alpha=0.7, ec='gray')
    ax[1].scatter(ep_esc_vel, ep_incident_flux/earth_incident_flux, c=ep_stellar_temp, vmin=2000, vmax=6000, cmap=cm, s=150, alpha=0.7, ec='gray')
    for i in range(len(trappist_1_incident_flux)):
        ax[1].text(trappist_1_esc_vel[i], trappist_1_incident_flux[i]/earth_incident_flux, trappist_1_names[i], fontweight='bold', horizontalalignment='center', verticalalignment='center')
    def cosmic_shoreline(x):
        return (0.17*x)**4
    x_array = np.linspace(6,30,100)
    ax[1].plot(x_array, cosmic_shoreline(x_array), c='gray')
    plt.annotate(ep_names[4], (ep_esc_vel[4]*1.05, ep_incident_flux[4]/earth_incident_flux), fontsize=8, fontweight='bold', verticalalignment='center', horizontalalignment='left')
    plt.annotate(ep_names[14], (ep_esc_vel[14]*1.05, ep_incident_flux[14]/earth_incident_flux), fontsize=8, fontweight='bold', verticalalignment='center', horizontalalignment='left')
    plt.annotate(ep_names[9], (ep_esc_vel[9]*1.05, ep_incident_flux[9]/earth_incident_flux), fontsize=8, fontweight='bold', verticalalignment='center', horizontalalignment='left')
    ax[1].text(trappist_1_esc_vel[0], trappist_1_incident_flux[0]/earth_incident_flux, 'x', fontsize=30, color='black', alpha=0.6, verticalalignment='center', horizontalalignment='center')
    ax[1].text(ep_esc_vel[4], ep_incident_flux[4]/earth_incident_flux, 'x', fontsize=30, color='black', alpha=0.6, verticalalignment='center', horizontalalignment='center')
    ax[1].text(ep_esc_vel[14], ep_incident_flux[14]/earth_incident_flux, 'x', fontsize=30, color='black', alpha=0.6, verticalalignment='center', horizontalalignment='center')
    ax[1].text(ep_esc_vel[9], ep_incident_flux[9]/earth_incident_flux, 'x', fontsize=30, color='black', alpha=0.6, verticalalignment='center', horizontalalignment='center')
    cbar = fig.colorbar(sc)
    cbar.set_label('Stellar Temperature [K]')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(6.7,26)
    ax[1].set_ylim(1e-1, 1.2e4)
    ax[1].set_xlabel(r'Escape Velocity [$\rm{km}/\rm{s}$]')
    ax[1].set_ylabel(r'Incident Flux [$\rm{F}_\oplus$]')
    ax[1].set_xticks(np.arange(10,26,5))
    ax[1].set_xticklabels(np.arange(10,26,5))

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')

    return



def figure2(): 
    
    '''Schematic'''
    
    return



def figure3(save=False, **kwargs): 
    
    '''Atmospheric Mass Loss Rate as Function of Age'''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'03 - Atmospheric dMdt Vs Age')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    TRAPPIST_1c = Planet()
    ages = np.logspace(-1.7,1,1000)*Gyr

    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)

    ax.plot(ages/Gyr, [_dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Dong', ages[i], TRAPPIST_1c.star.dMdt_boost) for i in range(len(ages))], label='Low (Dong et al. 2018)', color='k', linestyle='-')
    ax.plot(ages/Gyr, [_dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Garraffo', ages[i], TRAPPIST_1c.star.dMdt_boost) for i in range(len(ages))], label='High (Garraffo et al. 2017)', color='k', linestyle='--')
    ax.plot(ages/Gyr, [_dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Variable', ages[i], TRAPPIST_1c.star.dMdt_boost) for i in range(len(ages))], label='Variable', color='k', linestyle=':')
    ax.plot(TRAPPIST_1c.star.age_, _dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Variable', TRAPPIST_1c.star.age, TRAPPIST_1c.star.dMdt_boost), marker='*', color='red')
    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel('Atmospheric Mass-Loss Rate [kg/s]')
    ax.legend()

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')
    
    return



def figure4(save=False, **kwargs): 
    
    '''
    pCO2 Vs Time for Edges Cases of Parameter Space
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'04 - pCO2 Vs Time')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(figsize=(8,7), nrows=2, ncols=2, sharex=True, sharey=True, dpi=dpi)
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    colors = ['blue', 'orange', 'green']
    linestyles = {'Dong':'-', 'Garraffo':'--', 'Variable':':'}
    loc = 'upper left'

    for dMdt_model, linestyle in linestyles.items():
        
        planet, info_dict = load('Edge Case Simulations', mu_ref=1.0e21, init_T_man=2000, HPE_=1, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])
        t_gyr = planet[0].t_gyr

        for i in range(len(planet)):
            ax[0,0].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mathrm{C_{tot}}[\oplus]=$'f'{info_dict["init_C_man"][i]/1e22}')
        if dMdt_model == 'Dong':
            ax[0,0].legend(loc=loc)

        planet, info_dict = load('Edge Case Simulations', init_C_man=2.5e21, init_T_man=2000, HPE_=1, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])

        for i in range(len(planet)):
            text = '{:.1e}'.format(num2tex(info_dict['mu_ref'][i]))
            ax[0,1].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mu_\mathrm{ref}=$'+f'${text}$'+r'$ \ \mathrm{Pa \cdot s}$')
        if dMdt_model == 'Dong':
            ax[0,1].legend(loc=loc)

        planet, info_dict = load('Edge Case Simulations', init_C_man=2.5e21, mu_ref=1.0e21, HPE_=1, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])

        for i in range(len(planet)):
            ax[1,0].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mathrm{T_{init}}=$'f'{info_dict["init_T_man"][i]}'' K')
        if dMdt_model == 'Dong':
            ax[1,0].legend(loc=loc)

        planet, info_dict = load('Edge Case Simulations', init_C_man=2.5e21, mu_ref=1.0e21, init_T_man=2000, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])

        for i in range(len(planet)):
            ax[1,1].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mathrm{HPE}[\oplus]=$'f'{info_dict["HPE_"][i]}')
        if dMdt_model == 'Dong':
            ax[1,1].legend(loc=loc)

    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(1e-1,1e1)
    ax[0,0].set_ylim(1e-1,5e2)
    ax[1,0].set_xlabel('Time [Gyr]')
    ax[1,1].set_xlabel('Time [Gyr]')
    ax[0,0].set_ylabel(r'Surface Pressure ($\mathrm{pCO}_2$) [bar]')
    ax[1,0].set_ylabel(r'Surface Pressure ($\mathrm{pCO}_2$) [bar]')
    
    dong = mlines.Line2D([], [], color='black', linestyle='-',
                          markersize=15, label='Low')
    garraffo = mlines.Line2D([], [], color='black', linestyle='--',
                          markersize=15, label='High')
    variable = mlines.Line2D([], [], color='black', linestyle=':',
                          markersize=15, label='Variable')
    handles_dict = {'Dong':dong, 'Garraffo':garraffo, 'Variable':variable}
    handles = [handles_dict[i] for i in handles_dict]
    legend = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91,0.5))
    legend.set_title("Loss Rates")

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')
    
    return



def figure5(save=False, **kwargs): 
    
    '''
    Outgassing Rate Vs Time for 3 Different Carbon Budgets
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'05 - Outgassing Rate Vs Time')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)

    colors = ['blue', 'orange', 'green']
    
    planet, info_dict = load('Carbon Budget Simulations')
    outgassing_rate = np.array([p.outgassing_rate for p in planet])
    t_gyr = planet[0].t_gyr

    for i in range(len(planet)):
        ax.plot(t_gyr, outgassing_rate[i], c=colors[i], label=r'$\mathrm{C_{tot}}[\oplus]=$'f'{info_dict["init_C_man"][i]/1e22}')

    ax.set_yscale('log')
    ax.set_xlim(0,3)
    ax.set_ylim(8e-2,5e5)
    ax.set_xlabel('Age [Gyr]')
    ax.set_ylabel('Outgassing Rate [kg/s]')
    ax.legend()

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')
    
    return



def figure6(save=False, **kwargs):

    '''
    Highest Maximum Surface Pressure for Each Loss Rate
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'06 - High Max pCO2')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)

    linestyles = {'Dong':'-', 'Garraffo':'--', 'Variable':':'}
    labels = {'Dong':'Low', 'Garraffo':'High', 'Variable':'Variable'}

    for dMdt_model, linestyle in linestyles.items():

        P_surface_max, info_dict = load(f'All {dMdt_model} Simulations', 'P_surface Max')

        successful = (info_dict['P_surface @ 7.6 Gyr'] < 0.1e5)
        i_max = np.where(P_surface_max==np.max(P_surface_max[successful]))[0]

        p = Planet(init_C_man = info_dict['init_C_man'][i_max], mu_ref = info_dict['mu_ref'][i_max], init_T_man = info_dict['init_T_man'][i_max], HPE_ = info_dict['HPE_'][i_max], dMdt_model = dMdt_model)
        p.evolve()
        ax.plot(p.t_gyr, p.P_surface/1e5, c='k', linestyle=linestyle, label=labels[dMdt_model])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-1,1e1)
    ax.set_ylim(1e-1,3e1)
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel(r'Surface Pressure ($\mathrm{pCO}_2$) [bar]')
    ax.legend()

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')

    return



def figure7(save=False, **kwargs): 
    
    '''
    Histograms of Simulation Parameter Values that Result in pCO2 < 0.1 bar at 7.6 Gyr
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'07 - Successful Simulation Parameters Histogram')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)
    
    fig, ax = plt.subplots(figsize=(8.5,5.5), nrows=3, ncols=4, dpi=dpi, sharex=False, sharey=True)

    dMdt_models = ['Dong', 'Garraffo', 'Variable']

    for i in range(len(dMdt_models)):

        P_surface_at_age, info_dict = load(f'All {dMdt_models[i]} Simulations', 'P_surface @ 7.6 Gyr')

        successful = (P_surface_at_age < 0.1e5)
        successful_at_low_age = (info_dict['P_surface @ 5.4 Gyr'] < 0.1e5)
        successful_at_high_age = (info_dict['P_surface @ 9.8 Gyr'] < 0.1e5)

        N = len(np.unique(info_dict['init_C_man']))
        dx = (np.log10(2.5e22/1e22)-np.log10(2.5e20/1e22))/N
        ax[i,0].hist(np.log10(info_dict['init_C_man'][successful_at_high_age]/1e22), bins=np.linspace(np.log10(2.5e20/1e22)-0.5*dx, np.log10(2.5e22/1e22)+0.5*dx, N+1), color='r', edgecolor='black', linewidth=1.2)
        ax[i,0].hist(np.log10(info_dict['init_C_man'][successful]/1e22), bins=np.linspace(np.log10(2.5e20/1e22)-0.5*dx, np.log10(2.5e22/1e22)+0.5*dx, N+1), color='b', edgecolor='black', linewidth=1.2)
        ax[i,0].hist(np.log10(info_dict['init_C_man'][successful_at_low_age]/1e22), bins=np.linspace(np.log10(2.5e20/1e22)-0.5*dx, np.log10(2.5e22/1e22)+0.5*dx, N+1), color='y', edgecolor='black', linewidth=1.2)
        ax[i,0].vlines(np.median(np.log10(info_dict['init_C_man'][successful]/1e22)), 0, 30000, color='black', linestyle='dashed')
        ax[i,0].set_ylabel('Count')
        ax[i,0].set_ylim(0,3e4)

        N = len(np.unique(info_dict['mu_ref']))
        dx = (22-20)/N
        ax[i,1].hist(np.log10(info_dict['mu_ref'][successful_at_high_age]), bins=np.linspace(20-0.5*dx, 22+0.5*dx, N+1), color='y', edgecolor='black', linewidth=1.2)
        ax[i,1].hist(np.log10(info_dict['mu_ref'][successful]), bins=np.linspace(20-0.5*dx, 22+0.5*dx, N+1), color='b', edgecolor='black', linewidth=1.2)
        ax[i,1].hist(np.log10(info_dict['mu_ref'][successful_at_low_age]), bins=np.linspace(20-0.5*dx, 22+0.5*dx, N+1), color='r', edgecolor='black', linewidth=1.2)
        ax[i,1].vlines(np.median(np.log10(info_dict['mu_ref'][successful])), 0, 30000, color='black', linestyle='dashed')

        N = len(np.unique(info_dict['init_T_man']))
        dx = (2000-1700)/N
        ax[i,2].hist(info_dict['init_T_man'][successful_at_high_age], bins=np.linspace(1700-0.5*dx, 2000+0.5*dx, N+1), color='y', edgecolor='black', linewidth=1.2)
        ax[i,2].hist(info_dict['init_T_man'][successful], bins=np.linspace(1700-0.5*dx, 2000+0.5*dx, N+1), color='b', edgecolor='black', linewidth=1.2)
        ax[i,2].hist(info_dict['init_T_man'][successful_at_low_age], bins=np.linspace(1700-0.5*dx, 2000+0.5*dx, N+1), color='r', edgecolor='black', linewidth=1.2)
        ax[i,2].vlines(np.median(info_dict['init_T_man'][successful]), 0, 30000, color='black', linestyle='dashed')

        N = len(np.unique(info_dict['HPE_']))
        dx = (2-0.5)/N
        ax[i,3].hist(info_dict['HPE_'][successful_at_high_age], bins=np.linspace(0.5-0.5*dx, 2+0.5*dx, N+1), color='y', edgecolor='black', linewidth=1.2, label='9.8 Gyr')
        ax[i,3].hist(info_dict['HPE_'][successful], bins=np.linspace(0.5-0.5*dx, 2+0.5*dx, N+1), color='b', edgecolor='black', linewidth=1.2, label='7.6 Gyr')
        ax[i,3].hist(info_dict['HPE_'][successful_at_low_age], bins=np.linspace(0.5-0.5*dx, 2+0.5*dx, N+1), color='r', edgecolor='black', linewidth=1.2, label='5.4 Gyr')
        ax[i,3].vlines(np.median(info_dict['HPE_'][successful]), 0, 30000, color='black', linestyle='dashed')

    ax[0,0].text(-3, 15000, 'Low', fontweight='bold', horizontalalignment='right')
    ax[1,0].text(-3, 15000, 'High', fontweight='bold', horizontalalignment='right')
    ax[2,0].text(-3, 15000, 'Variable', fontweight='bold', horizontalalignment='right')
    ax[2,0].set_xlabel(r'$\mathrm{log}_{10}\mathrm{(C_{tot}}[\oplus])$')
    ax[2,1].set_xlabel(r'$\mathrm{log}_{10}\mathrm{(\mu_{ref}[Pa \cdot s])}$')
    ax[2,2].set_xlabel(r'$\mathrm{T_{init}}[\mathrm{K}]$')
    ax[2,3].set_xlabel(r'$\mathrm{HPE}[\oplus]$')
    ax[0,3].legend()

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')
    
    return



def figure8(save=False, **kwargs): 
    
    '''
    pCO2 @ 7.6 Gyr For 10 Different Carbon Budgets and 10 Different Constant Loss Rates
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'08 - pCO2 Vs Carbon Budget Vs Loss Rate Constant')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(dpi=dpi)

    P_surface_at_age, info_dict = load(f'Carbon Budget and Loss Rate (Constant) Simulations', 'P_surface @ 7.6 Gyr')
    P_surface_at_age = np.reshape(P_surface_at_age, (10,10))

    im = ax.imshow(P_surface_at_age/1e5, extent=[-2.22222222, 2.22222222, -1.7131711, 0.50905112], 
                origin='lower', aspect=2, norm=mpl.colors.LogNorm(vmin=4e-2, vmax=2e2))
    ax.set_xticks(np.linspace(-2, 2, 10))
    ax.set_yticks(np.log10(np.logspace(np.log10(2.5e20), np.log10(2.5e22), 10)/1e22))
    ax.tick_params(axis='x', labelrotation = 90)
    fig.colorbar(im)
    ax.set_xlabel(r'$\mathrm{log}_{10}(\mathrm{Loss \ Rate \ [Low]})$')
    ax.set_ylabel(r'$\mathrm{log}_{10}\mathrm{(C_{tot}}[\oplus])$')
    ax.set_title(r'$\mathrm{pCO}_2$ @ 7.6 Gyr [bar]')
    ax.set_facecolor('gray')
    ax.text(1.45,-0.8,'NO ATMOSPHERE',va='center',ha='center',rotation=70,fontsize=20)

    if save: plt.savefig(os.path.join(folder, filename+ext), bbox_inches='tight')
    
    return



def figure9(save=False, **kwargs): 
    
    '''
    pCO2 @ 7.6 Gyr For 10 Different Carbon Budgets and 10 Different Variable Loss Rates
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'09 - pCO2 Vs Carbon Budget Vs Loss Rate Variable')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(figsize=(11,4), nrows=1, ncols=2, dpi=dpi)

    P_surface_at_age, info_dict = load(f'Carbon Budget and Loss Rate (Variable) Simulations', 'P_surface @ 7.6 Gyr')
    P_surface_at_age = np.reshape(P_surface_at_age, (10,10))
    P_surface_at_age[P_surface_at_age/1e5<4e-2] = np.nan

    im = ax[0].imshow(P_surface_at_age/1e5, extent=[-2.22222222, 2.22222222, -1.7131711, 0.50905112], 
                    origin='lower', aspect=2, norm=mpl.colors.LogNorm(vmin=4e-2, vmax=2e2))
    ax[0].set_xticks(np.linspace(-2, 2, 10))
    ax[0].set_yticks(np.log10(np.logspace(np.log10(2.5e20), np.log10(2.5e22), 10)/1e22))
    ax[0].tick_params(axis='x', labelrotation = 90)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_xlabel(r'$\mathrm{log}_{10}(\mathrm{Loss \ Rate \ at \ 7.6  \ Gyr[Low]})$')
    ax[0].set_ylabel(r'$\mathrm{log}_{10}\mathrm{(C_{tot}}[\oplus])$')
    ax[0].set_title(r'$\mathrm{pCO}_2$ @ 7.6 Gyr [bar]')
    ax[0].set_facecolor('gray')
    ax[0].text(1.00,-0.7,'NO ATMOSPHERE',va='center',ha='center',rotation=63,fontsize=20)

    TRAPPIST_1c = Planet()
    ages = np.logspace(-1.7,1,1000)*Gyr

    dMdt_boosts = np.unique(info_dict['dMdt_boost'])
    for j in range(len(dMdt_boosts)):
        ax[1].semilogy(ages/Gyr, [_dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Variable', ages[i], dMdt_boosts[j]) for i in range(len(ages))], color='k', linestyle=':')
        ax[1].plot(TRAPPIST_1c.star.age_, _dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Variable', TRAPPIST_1c.star.age, dMdt_boosts[j]), marker='*', color='red')
    ax[1].plot(ages/Gyr, [_dMdt_atm(TRAPPIST_1c.R_, TRAPPIST_1c.a_, 'Dong', ages[i], TRAPPIST_1c.star.dMdt_boost) for i in range(len(ages))], label='Low (Dong et al. 2018)', color='k', linestyle='-')
    ax[1].set_ylim(1e-1,1e6)
    ax[1].set_xlabel('Age [Gyr]')
    ax[1].set_ylabel('Atmospheric Mass-Loss Rate [kg/s]')
    ax[1].legend()

    if save: plt.savefig(os.path.join(folder, filename+ext), bbox_inches='tight')
    
    return



def figure10(save=False, **kwargs):

    '''
    pCO2 Vs Time for Edges Cases of Parameter Space including Tidal Heating
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'10 - pCO2 Vs Time w Tidal Heating')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(figsize=(8,7), nrows=2, ncols=2, sharex=True, sharey=True, dpi=dpi)
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    colors = ['blue', 'orange', 'green']
    linestyles = {'Dong':'-', 'Garraffo':'--', 'Variable':':'}
    loc = 'upper left'

    for dMdt_model, linestyle in linestyles.items():
        
        planet, info_dict = load('Edge Case Tidal Heating Simulations', mu_ref=1.0e21, init_T_man=2000, HPE_=1, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])
        t_gyr = planet[0].t_gyr

        for i in range(len(planet)):
            ax[0,0].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mathrm{C_{tot}}[\oplus]=$'f'{info_dict["init_C_man"][i]/1e22}')
        if dMdt_model == 'Dong':
            ax[0,0].legend(loc=loc)

        planet, info_dict = load('Edge Case Tidal Heating Simulations', init_C_man=2.5e21, init_T_man=2000, HPE_=1, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])

        for i in range(len(planet)):
            text = '{:.1e}'.format(num2tex(info_dict['mu_ref'][i]))
            ax[0,1].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mu_\mathrm{ref}=$'+f'${text}$'+r'$ \ \mathrm{Pa \cdot s}$')
        if dMdt_model == 'Dong':
            ax[0,1].legend(loc=loc)

        planet, info_dict = load('Edge Case Tidal Heating Simulations', init_C_man=2.5e21, mu_ref=1.0e21, HPE_=1, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])

        for i in range(len(planet)):
            ax[1,0].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mathrm{T_{init}}=$'f'{info_dict["init_T_man"][i]}'' K')
        if dMdt_model == 'Dong':
            ax[1,0].legend(loc=loc)

        planet, info_dict = load('Edge Case Tidal Heating Simulations', init_C_man=2.5e21, mu_ref=1.0e21, init_T_man=2000, dMdt_model=dMdt_model)
        P_surface = np.array([p.P_surface for p in planet])

        for i in range(len(planet)):
            ax[1,1].plot(t_gyr, P_surface[i]/1e5, c=colors[i], linestyle=linestyle, label=r'$\mathrm{HPE}[\oplus]=$'f'{info_dict["HPE_"][i]}')
        if dMdt_model == 'Dong':
            ax[1,1].legend(loc=loc)

    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlim(1e-1,1e1)
    ax[0,0].set_ylim(1e-1,5e2)
    ax[1,0].set_xlabel('Time [Gyr]')
    ax[1,1].set_xlabel('Time [Gyr]')
    ax[0,0].set_ylabel(r'Surface Pressure ($\mathrm{pCO}_2$) [bar]')
    ax[1,0].set_ylabel(r'Surface Pressure ($\mathrm{pCO}_2$) [bar]')
    
    dong = mlines.Line2D([], [], color='black', linestyle='-',
                          markersize=15, label='Low')
    garraffo = mlines.Line2D([], [], color='black', linestyle='--',
                          markersize=15, label='High')
    variable = mlines.Line2D([], [], color='black', linestyle=':',
                          markersize=15, label='Variable')
    handles_dict = {'Dong':dong, 'Garraffo':garraffo, 'Variable':variable}
    handles = [handles_dict[i] for i in handles_dict]
    legend = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.91,0.5))
    legend.set_title("Loss Rates")

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')

    return



def figure11(save=False, **kwargs):

    '''
    pCO2 Vs Time for Edges Cases of Initial Carbon Budget Varying Fraction in Atmosphere
    '''

    folder = kwargs.pop('folder', figures_dir)
    filename = kwargs.pop('filename', f'11 - pCO2 Vs Time w Varied C Location')
    ext = kwargs.pop('ext', ".pdf")
    dpi = kwargs.pop('dpi', 200)

    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    colors = ['blue', 'orange', 'green']
    loc = 'lower center'

    planet, info_dict = load('Vary C Location')

    P_surface = np.array([p.P_surface for p in planet])
    t_gyr = planet[0].t_gyr

    for i in range(len(planet)):
        init_C_man = info_dict["init_C_man"][i]
        init_C_atm = info_dict["init_P_surface"][i]/(m_bar_co2*(planet[i].g/planet[i].SA))
        init_C_tot = init_C_man + init_C_atm
        f_atm = init_C_atm/init_C_tot
        if init_C_tot>2.4e20 and init_C_tot<2.6e20:
            ax.plot(t_gyr, P_surface[i]/1e5, c=colors[0], alpha=f_atm*0.9+0.1)
            if f_atm==1:
                ax.plot(t_gyr, P_surface[i]/1e5, c=colors[0], alpha=f_atm*0.9+0.1, label=f'{np.round(init_C_tot/1e22,3)}')
        if init_C_tot>2.4e21 and init_C_tot<2.6e21:
            ax.plot(t_gyr, P_surface[i]/1e5, c=colors[1], alpha=f_atm*0.9+0.1)
            if f_atm==1:
                ax.plot(t_gyr, P_surface[i]/1e5, c=colors[1], alpha=f_atm*0.9+0.1, label=f'{np.round(init_C_tot/1e22,2)}')
        if init_C_tot>2.4e22 and init_C_tot<2.6e22:
            ax.plot(t_gyr, P_surface[i]/1e5, c=colors[2], alpha=f_atm*0.9+0.1)
            if f_atm==1:
                ax.plot(t_gyr, P_surface[i]/1e5, c=colors[2], alpha=f_atm*0.9+0.1, label=f'{np.round(init_C_tot/1e22,1)}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-1,1e1)
    ax.set_ylim(1e-1,5e2)
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel(r'Surface Pressure ($\mathrm{pCO}_2$) [bar]')

    legend_1 = fig.legend(loc='lower left', bbox_to_anchor=(0.91,0.5))
    legend_1.set_title(r'$\mathrm{C_{tot}}[\oplus]$')
    
    f_atm_0 = mlines.Line2D([], [], color='black', alpha=0.1,
                        markersize=15, label='0')
    f_atm_1 = mlines.Line2D([], [], color='black', alpha=0.5,
                        markersize=15, label='0.5')
    f_atm_2 = mlines.Line2D([], [], color='black', alpha=1,
                        markersize=15, label='1')
    handles_dict = {'f_atm_0':f_atm_0, 'f_atm_1':f_atm_1, 'f_atm_2':f_atm_2}
    handles = [handles_dict[i] for i in handles_dict]
    legend_2 = fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.91,0.5))
    legend_2.set_title(r'$f_\mathrm{atm}$')

    if save: plt.savefig(os.path.join(folder, filename+ext), transparent=True, bbox_inches='tight')

    return

# %%
