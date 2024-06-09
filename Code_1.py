import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dm4bem

h = 3               # m height of the building
l = 5               # m length of the short sides of the building
La = 4               # m length of room a
Lb = 6                  # m length of room b
Sw = 3               # m² surface area of the glass window; w=3m, h=1m
# Sgb = 9              # total window area in second room
Sd = 2              # Surface area of the door between room a and b; w=1m, h=2m
# Sci = 78             # m² surface area of concrete & insulation of the 4 walls
# Sc = 13              # m² surface area of concrete of the 1 wall
w = 0.28     # m, wall thickness
# wi = 0.2     #inside wall thickness in m
# S = 20.0    # m², wall surface area
Sa = l*h - Sw + La*h*2
Sb = l*h + Lb*h*2
Sintw = l*h - Sd
# Sina1 = l*h - Sw
# Sina2 = l*h - Sd
T0 = -5.0   # °C, outside air temperature
# Ti = 24.0   # °C, inside air temperature
# λ = 1.0     # W/(m·K), thermal conductivity
ho = 25.0   # W/(m²·K), outside convection coefficient
hi = 8.0    # W/(m²·K), outside convection coefficient
α = 0.70    # -, absorbtance of outdoor surface
E = 200.0   # W/m², solar irradiance on the outdoor surface

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

np.set_printoptions(precision=1)


concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2,                   # m
            # 'Surface': 91}                  # m²; 4 outside walls + wall in the middle

# insulation = {'Conductivity': 0.027,        # W/(m·K)
#               'Density': 55.0,              # kg/m³
#               'Specific heat': 1210,        # J/(kg⋅K)
#               'Width': 0.08,                # m
#               'Surface': 78}                # m²

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         # 'Surface': 6}                      # m²; 2 windows

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')
# wall

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant
# print(f'σ = {σ} W/(m²⋅K⁴)')

######### IMPORTANT 1 ##############
# h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
# h

# conduction
# G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
# pd.DataFrame(G_cd, columns=['Conductance'])

# # convection
# Gw = h * wall['Surface'].iloc[0]     # wall
# Gg = h * wall['Surface'].iloc[2]     # glass

# # view factor wall-glass
# Fwg = glass['Surface'] / concrete['Surface']

# # Long wave radiation
# T_int = 273.15 + np.array([0, 40])
# coeff = np.round((4 * σ * T_int**3), 1)
# # print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

# T_int = 273.15 + np.array([10, 30])
# coeff = np.round((4 * σ * T_int**3), 1)
# # print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

# T_int = 273.15 + 20
# coeff = np.round((4 * σ * T_int**3), 1)
# # print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')

# # long wave radiation
# Tm = 20 + 273   # K, mean temp for radiative exchange

# GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Layer_in']
# GLW12 = 4 * σ * Tm**3 * Fwg * wall['Surface']['Layer_in']
# GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']

# GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

# Ventilation flow rate
Va = l*La*h + l*Lb*h                     # m³, volume of air for room a
Va = l*La*h + l*Lb*h                     # m³, volume of air
Vt = Va + Vb                             # m³, total volume of air
ACH = 1                                  # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va                 # m³/s, air infiltration
m_dot = air['Density'] * Va_dot          # mass air flow rate

# Ventilation & advection
Gv = m_dot * air['Specific heat']

# P-controler gain
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
Kp = 0

# glass: convection outdoor & conduction
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 1 / (2 * G_cd['Glass'])))

C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])

C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns=['Capacity'])

Φoa = α * Sexta * E
Φob = α * Sextb * E
Φia1 =  E * Sga * 0.2
Φia2 =  E * Sga * 0.8
Φib1 =  0                   # we assume that the sun is not facing the windows of room b
Φib2 =  0
Φa = 0

nq, nθ = 25, 19  # number of flow-rates branches and of temperaure nodes
nfinal = 6       # node for which we want the final value

# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7', 'θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14', 'θ15', 'θ16', 'θ17', 'θ18']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24']

A = np.zeros([nq, nθ])          # n° of branches X n° of nodes
A[0, 0] = 1                     # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1        # branch 1: node 0, node 1
A[2, 1], A[2, 2] = -1, 1        # branch 2: node 1, node 2
A[3, 2], A[3, 3] = -1, 1        # branch 3: node 2, node 3
A[4, 3], A[4, 4] = -1, 1        # branch 4: node 3, node 4
A[5, 4], A[5, 5] = -1, 1        # branch 5: node 4, node 5
A[6, 5], A[6, 6] = -1, 1        # branch 6: node 5, node 6
A[7, 6], A[7, 7] = -1, 1        # branch 7: node 6, node 7
A[8, 7], A[8, 8] = -1, 1        # branch 8: node 7, node 8
A[9, 8], A[9, 9] = -1, 1        # branch 9: node 8, node 9
A[10, 9], A[10, 10] = -1, 1     # branch 10: node 9, node 10
A[11, 10], A[11, 11] = -1, 1    # branch 11: node 10, node 11
A[12, 11], A[12, 12] = -1, 1    # branch 12: node 11, node 12
A[13, 12], A[13, 13] = -1, 1    # branch 13: node 12, node 13
A[14, 13], A[14, 14] = -1, 1    # branch 14: node 13, node 14
A[15, 14] = 1                   # branch 15: node 14
A[16, 4], A[16, 16] = 1, -1     # branch 16: node 4, node 16
A[17, 15] = 1                   # branch 17: node 15
A[18, 15], A[18, 16] = -1, 1    # branch 18: node 15, node 16
A[19, 5], A[19, 16] = 1, -1     # branch 19: node 5, node 16
A[20, 5], A[20, 9] = -1, 1      # branch 20: node 5, node 9
A[21, 9], A[21, 17] = 1, -1     # branch 21: node 9, node 17
A[22, 10], A[22, 17] = 1, -1     # branch 22: node 10, node 17
A[23, 17], A[23, 18] = 1, -1     # branch 23: node 17, node 18
A[24, 18] = 1                   # branch 24: node 18
pd.DataFrame(A, index=q, columns=θ)

# Calculate the conductances
G0 = ho*Sexta
G1 = 1/2*concrete['Conductivity']*Sexta/w
G2 = 1/2*concrete['Conductivity']*Sexta/w
G3 = 1/2*insulation['Conductivity']*Sexta/w
G4 = 1/2*insulation['Conductivity']*Sexta/w
G5 = hi*Sexta
G6 = hi*Sint
G7 = 1/2*concrete['Conductivity']*Sint/wi
G8 = 1/2*concrete['Conductivity']*Sint/wi
G9 = hi*Sint
G10 = hi*Sextb
G11 = 1/2*insulation['Conductivity']*Sextb/w
G12 = 1/2*insulation['Conductivity']*Sextb/w
G13 = 1/2*concrete['Conductivity']*Sextb/w
G14 = 1/2*concrete['Conductivity']*Sextb/w
G15 = ho*Sextb
G16 = ε_gLW / (1-ε_gLW) * Sga # long wave radiation in room a
G17 = ho*Sga
G18 = glass['Conductivity']*Sga/glass['Width']
G19 = hi*Sga
G20 = air['Specific heat'] * m_dot # transport through door
G21 = hi*Sgb
G22 = ε_gLW / (1-ε_gLW) * Sgb # long wave radiation in room b
G23 = glass['Conductivity']*Sgb/glass['Width']
G24 = ho*Sgb
#G25 = Ka                # Controller constant for room a
#G26 = Kb                # Controller constant for room b

# G = pd.Series([G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, 0, T0, 0, 0, 0, 0, 0, 0, T0],
#               index=q)

G = pd.Series([G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20, G21, G22, G23, G24],
               index=q)


# # Capacities
neglect_air_glass = False

if neglect_air_glass:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0, 0, C['Air'], 0, 0, 0, C['Layer_in'], 0, C['Layer_out'], 0, 0, 0, 0, 0])
else:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0, 0, C['Air'], 0, 0, 0, C['Layer_in'], 0, C['Layer_out'], 0, C['Glass'], 0, 0, C['Glass']])

pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)

# C['Layer_out'] = concrete['Density'] * concrete['Specific heat'] * concrete['Surface'] * concrete['Width']
# C['Layer_in'] = insulation['Density'] * insulation['Specific heat'] * insulation['Surface'] * insulation['Width']
# C['Air'] = air['Density'] * air['Specific heat'] * Va
# C['Glass'] = glass['Density'] * glass['Specific heat'] * glass['Surface'] * glass['Width']

#temperature source network
b = pd.Series([T0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, T0, 0, T0, 0, 0, 0, 0, 0, 0, T0],
              index=q)

#heat flow source vector
f = pd.Series([Φoa, 0, 0, 0, Φia1, 0, Φia2, 0, Φib2, 0, Φib1, 0, 0, 0, Φob, Φa, 0, 0, Φa],
              index=θ)

#output matrix
y = np.zeros(nθ)         # nodes
y[[nfinal]] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)

# [As, Bs, Cs, Ds, us] = fTC2SS(A,G,b,C,f,y)

###############CSV######################

# TC = dm4bem.file2TC('Thermal Calculations - Matrices.csv', name='', auto_number=False)

# print(b)


print('Matrices and vectors for thermal circuit from Figure 1') 
df = pd.read_csv('Thermal Calculations - Matrices.csv')
df.style.apply(lambda x: ['background-color: yellow'
                          if x.name in df.index[-3:] or c in df.columns[-2:]
                          else '' for c in df.columns], axis=1)

# State-space
# [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
# print(Bs)
#print("Dimensions are: ", Bs.ndim)


###############Steady state######################33
bss = np.zeros(24)        # temperature sources b for steady state
bss[[0, 15, 17, 23]] = -5      # outdoor temperature
#bss[[24]] = -5            # indoor set-point temperature

fss = np.zeros(19)         # flow-rate sources f for steady state

#


#
A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A.to_numpy()) @ (A.T @ diag_G @ bss + fss)
print(f'θss = {np.around(θss, 2)} °C')
