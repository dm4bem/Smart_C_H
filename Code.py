########## A matrix ##########
nq, nθ = 18, 14  # number of flow-rates branches and temperature nodes

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
b[[0,1,14]] = To         # outdoor temperature for walls
b[13] = Tsp              # Controller temperature for room b

########## C matrix ##########
C = np.zeros(A.shape[1])
C[1] = concrete['Density'] * Va_d * Va * concrete['Specific heat']         # Capacitances in Concrete Wall for room a
C[9] = concrete['Density'] * Va_d * Vb * concrete['Specific heat']         # Capacitances in Concrete Wall for room b
C[5] = concrete['Density'] * Va_d * Vt * concrete['Specific heat']         # Capacitances in Concrete Wall for inner wall
C[12] = glass['Density'] * Va_d * Va * glass['Specific heat']         # Capacitances in Glass window

########## f matrix ##########
f = np.zeros(A.shape[1])
f[0] = αo * Sa * E                     # Outdoor Radiation absorbed for room a
f[10] = αo * Sb * E                    # Outdoor Radiation absorbed for room b
f[2] = αi * Sa * E                       # Indoor Radiation absorbed for room a
f[8] = αi * Sb * E                       # Indoor Radiation absorbed for room b 
f[[4,6]] = αi * Sintw * E                       # Indoor Radiation absorbed for inner-wall 
f[11] = αo * Sg * E                       # Outdoor Radiation absorbed for Window
f[13] = αiw * Sg * E                       # Indoor Radiation absorbed for Window

########## Outputs ##########
indoor_air = [3, 7]         # indoor air temperature nodes
controller = [13]            # controller node

b[controller] = Tsp          # °C setpoint temperature for room b
G[controller] = 1e4         # P-controller gain

θ = np.linalg.inv(np.diag(C) + A.T @ np.diag(G) @ A) @ (A.T @ np.diag(G) @ b + f)
q = np.diag(G) @ (-A @ θ + b)
print("Only Room b is controlled")
print("θ:", θ[indoor_air], "°C")
print("q:", q[controller], "W")      # The thermal loads are the heat flow rates of the controllers

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

print("State matrix As:\n", As)
print("Input matrix Bs:\n", Bs)
print("Output matrix Cs:\n", Cs)
print("Direct transmission matrix Ds:\n", Ds)
