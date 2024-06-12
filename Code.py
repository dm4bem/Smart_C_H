import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Building dimensions and properties
h = 3                     # m height of the building
l = 5                     # m length of the short sides of the building
La = 4                    # m length of room a
Lb = 6                    # m length of room b
Sg = 3                    # m² surface area of the glass window; w=3m, h=1m
Sd = 2                    # Surface area of the door between room a and b; w=1m, h=2m
Sa = l*h - Sg + La*h*2
Sb = l*h + Lb*h*2
Sintw = l*h - Sd
To = -5.0                 # °C, outside air temperature
Tsp = 25                  # Controller temperature
# Ti = 24.0               # °C, inside air temperature
ho = 25.0                 # W/(m²·K), outside convection coefficient
hi = 8.0                  # W/(m²·K), outside convection coefficient
αo = 0.70                 # short wave absorptivity: outdoor surface
αi = 0.25                 # short wave absorptivity: white smooth surface
αiw = 0.38                # short wave absorptivity: reflective blue glass
E = 200.0                 # W/m², solar irradiance on the outdoor surface

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

np.set_printoptions(precision=1)


concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2}                   # m

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04}                     # m

wall = pd.DataFrame.from_dict({'Concrete': concrete,
                               'Glass': glass},
                              orient='index')

# Ventilation flow rate
Va = l*La*h                              # m³, volume of air for room a
Vb = l*Lb*h                              # m³, volume of air for room b
Vt = Va + Vb                             # m³, total volume of air
ACH = 0.5                                # air changes per hour
Va_d = (ACH / 3600)                      # m³/s, air infiltration


########## A matrix ##########
nq, nθ = 18, 14                 # number of flow-rates branches and temperature nodes

A = np.zeros([nq, nθ])          # n° of branches X n° of nodes

# Outside Convection
A[0, 0] = 1                     # branch 0: node 0
A[1, 10] = 1                    # branch 1: node 10

# Wall Conduction
A[2, 0], A[2, 1] = -1, 1        # branch 2: node 0, node 1
A[3, 1], A[3, 2] = -1, 1        # branch 3: node 1, node 2
A[4, 9], A[4, 10] = -1, 1       # branch 4: node 9, node 10
A[5, 8], A[5, 9] = -1, 1        # branch 5: node 8, node 9

# Wall Indoor Convection
A[6, 2], A[6, 3] = -1, 1        # branch 6: node 2, node 3
A[7, 7], A[7, 8] = -1, 1        # branch 7: node 7, node 8

# Inner-wall Conduction
A[8, 4], A[8, 5] = -1, 1        # branch 8: node 7, node 8
A[9, 5], A[9, 6] = -1, 1        # branch 9: node 5, node 6

# Inner-wall Indoor Convection
A[10, 3], A[10, 4] = -1, 1      # branch 10: node 3, node 4
A[11, 6], A[11, 7] = -1, 1      # branch 11: node 6, node 7

# Advection
A[12, 3], A[12, 7] = -1, 1      # branch 12: node 3, node 7

# Controller
A[13, 7] = 1                    # branch 13: node 7

# Window Outside Convection
A[14, 11] = 1                   # branch 14: node 11

# Window Conduction
A[15, 11], A[15, 12] = -1, 1    # branch 15: node 11, node 12
A[16, 12], A[16, 13] = -1, 1    # branch 16: node 12, node 13

# Window Indoor Convection
A[17, 3], A[17, 13] = 1, -1     # branch 17: node 3, node 13


########## G matrix ##########
G = np.zeros(A.shape[0])

# Outside Convection
G[0] = ho * Sa
G[1] = ho * Sb

# Wall Conduction
G[2:4] = (concrete['Conductivity'] / concrete['Width']) * Sa
G[4:6] = (concrete['Conductivity'] / concrete['Width']) * Sb

# Wall Indoor Convection
G[6] = hi * Sa
G[7] = hi * Sb

# Inner-wall Conduction
G[8:10] = (concrete['Conductivity'] / concrete['Width']) * Sintw

# Inner-wall Indoor Convection
G[10:12] = hi * Sintw

# Advection
G[12] = air['Density'] * Va_d * Vt * air['Specific heat']

# Controller
G[13] = 0

# Window Outside Convection
G[14] = ho * Sg

# Window Conduction
G[15:17] = (glass['Conductivity'] / glass['Width']) * Sg

# Window Indoor Convection
G[17] = hi * Sg

########## b matrix ##########
b = np.zeros(A.shape[0])
b[[0,1,14]] = To               # outdoor temperature for walls
b[13] = Tsp                    # Controller temperature for room b

########## C matrix ##########
C = np.zeros(A.shape[1])
C[1] = concrete['Density'] * Va_d * Va * concrete['Specific heat']         # Capacitances in Concrete Wall for room a
C[9] = concrete['Density'] * Va_d * Vb * concrete['Specific heat']         # Capacitances in Concrete Wall for room b
C[5] = concrete['Density'] * Va_d * Vt * concrete['Specific heat']         # Capacitances in Concrete Wall for inner wall
C[12] = glass['Density'] * Va_d * Va * glass['Specific heat']              # Capacitances in Glass window

########## f matrix ##########
f = np.zeros(A.shape[1])
f[0] = αo * Sa * E                     # Outdoor Radiation absorbed for room a
f[10] = αo * Sb * E                    # Outdoor Radiation absorbed for room b
f[2] = αi * Sa * E                     # Indoor Radiation absorbed for room a
f[8] = αi * Sb * E                     # Indoor Radiation absorbed for room b 
f[[4,6]] = αi * Sintw * E              # Indoor Radiation absorbed for inner-wall 
f[11] = αo * Sg * E                    # Outdoor Radiation absorbed for Window
f[13] = αiw * Sg * E                   # Indoor Radiation absorbed for Window

########## Outputs ##########
indoor_air = [3, 7]           # indoor air temperature nodes
controller = [13]             # controller node

b[controller] = Tsp           # °C setpoint temperature for room b
G[controller] = 1e4           # P-controller gain

θ = np.linalg.inv(np.diag(C) + A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)

########## State-space Equations ##########

epsilon = 1e-10  # Small regularization term
C_diag_inv = np.linalg.inv(np.diag(C) + epsilon * np.eye(nθ))

# State matrix As
As = -C_diag_inv @ (A.T @ np.diag(G) @ A)

# Input matrix Bs
Bs = C_diag_inv @ (A.T @ np.diag(G) @ b + f)

# Output matrix Cs (assuming the outputs are the states themselves)
Cs = np.eye(nθ)

# Direct transmission matrix Ds (assuming no direct transmission)
Ds = np.zeros((nθ, 1))

########## Step response ##########

# Define the time parameters
dt = 1.0  # Time step in hours
total_time = 24  # Total simulation time in hours (24 hours)
time_steps = np.arange(0, total_time + dt, dt)

# Initialize temperature array
theta = np.zeros((len(time_steps), nθ))
theta[0, :] = θ  # Initial condition

# Apply a step change in outdoor temperature at t = 0
To_step = 0.0  # New outdoor temperature after the step change
b[[0, 1, 14]] = To_step

# Regularization term to stabilize As
regularization_term = 1e-5
As_reg = As - regularization_term * np.eye(nθ)

# Function to perform backward Euler integration step
def backward_euler_step(theta_prev, As_reg, Bs, dt):
    I = np.eye(len(As_reg))
    theta_dot = np.linalg.inv(I - dt * As_reg) @ (theta_prev + dt * Bs.flatten())
    return theta_dot

# Calculate the step response
for i in range(1, len(time_steps)):
    try:
        theta[i, :] = backward_euler_step(theta[i-1, :], As_reg, Bs, dt)
        if np.any(np.abs(theta[i, :]) > 1e10):
            raise OverflowError("Potential overflow detected in theta.")
    except OverflowError as e:
        print(f"Overflow detected at step {i}. Clamping values.")
        theta[i, :] = np.clip(theta[i, :], -1e10, 1e10)  # Clamp values to prevent overflow
