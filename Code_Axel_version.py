import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem

h = 3               # m height of the building
l = 5               # m length of the short sides of the building
La = 4               # m length of room a
Lb = 6                  # m length of room b
Sga = 3               # m² surface area of the glass window; w=3m, h=1m
Sgb = 9              # total window area in second room
Sd = 2              # Surface area of the door between room a and b
wc = 0.2     # m, outer wall thickness, concrete part
wi = 0.08       #m, outer wall thickness, insulation part
win = 0.2     #inside wall thickness in m
wg = 0.04    # m, window thickness
Sexta = l*h - Sga + La*h*2
Sextb = l*h - Sgb + Lb*h*2
Sint = l*h - Sd
T0 = -5.0   # °C, outside air temperature
Ti = 20.0   # °C, inside air temperature setpoint
λ = 1.0     # W/(m·K), thermal conductivity
ho = 25.0   # W/(m²·K), outside convection coefficient
hi = 8.0    # W/(m²·K), outside convection coefficient
α = 0.70    # -, absorbtance of outdoor surface
E = 200.0   # W/m², solar irradiance on the outdoor surface
Ka = 1e6
Kb = 1e6

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

np.set_printoptions(precision=1)

concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880}           

insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210}        # J/(kg⋅K)

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210}             # J/(kg⋅K)

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant
# print(f'σ = {σ} W/(m²⋅K⁴)')

# ventilation flow rate
Va = La*l*h                   # m³, volume of air
ACH = 1                     # 1/h, air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration
m_dot = air['Density'] * Va_dot               # mass air flow rate

nq, nθ = 25, 19  # number of flow-rates branches and of temperaure nodes
nfinal = [5, 9]       # node for which we want the final value

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
G1 = 1/2*concrete['Conductivity']*Sexta/wc
G2 = 1/2*concrete['Conductivity']*Sexta/wc
G3 = 1/2*insulation['Conductivity']*Sexta/wi
G4 = 1/2*insulation['Conductivity']*Sexta/wi
G5 = hi*Sexta
G6 = hi*Sint
G7 = 1/2*concrete['Conductivity']*Sint/win
G8 = 1/2*concrete['Conductivity']*Sint/win
G9 = hi*Sint
G10 = hi*Sextb
G11 = 1/2*insulation['Conductivity']*Sextb/wi
G12 = 1/2*insulation['Conductivity']*Sextb/wi
G13 = 1/2*concrete['Conductivity']*Sextb/wc
G14 = 1/2*concrete['Conductivity']*Sextb/wc
G15 = ho*Sextb
G16 = ε_gLW / (1-ε_gLW) * Sga # long wave radiation in room a
G17 = ho*Sga
G18 = glass['Conductivity']*Sga/wg
G19 = hi*Sga
G20 = air['Specific heat'] * m_dot # transport through door
G21 = hi*Sgb
G22 = ε_gLW / (1-ε_gLW) * Sgb # long wave radiation in room b
G23 = glass['Conductivity']*Sgb/wg
G24 = ho*Sgb
G25 = Ka                # Controller constant for room a
G26 = Kb                # Controller constant for room b

G = pd.Series([G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20, G21, G22, G23, G24],
              index=q)

# Capacity (only walls not neglected)
Cwac = concrete['Density']*concrete['Specific heat']*Sexta*wc
Cwai = insulation['Density']*insulation['Specific heat']*Sexta*wi
Cwbc = concrete['Density']*concrete['Specific heat']*Sextb*wc
Cwbi = insulation['Density']*insulation['Specific heat']*Sextb*wi
Cwin = concrete['Density']*concrete['Specific heat']*Sint*win
C = pd.Series([0, Cwac, 0, Cwai, 0, 0, 0, Cwin, 0, 0, 0, Cwbi, 0, Cwbc, 0, 0, 0, 0, 0], index=θ)

#temperature source network
b = pd.Series([T0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, T0, 0, T0, 0, 0, 0, 0, 0, 0, T0],
              index=q)

#heat flow source vector
Φoa = α * Sexta * E
Φob = α * Sextb * E
Φia1 =  E * Sga * 0.2
Φia2 =  E * Sga * 0.8
Φib1 =  E * Sgb * 0.8
Φib2 =  E * Sgb * 0.2
Φa = 0
f = pd.Series([Φoa, 0, 0, 0, Φia1, 0, Φia2, 0, Φib2, 0, Φib1, 0, 0, 0, Φob, Φa, 0, 0, Φa],
              index=θ)

#output matrix
y = np.zeros(nθ)         # nodes
y[nfinal] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
