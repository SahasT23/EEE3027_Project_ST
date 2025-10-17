"""
EEE3027: Integrated Circuit Design and Embedded Systems
Task 4 – SPICE Under the Hood

Newton–Raphson Iteration for Diode Circuit Simulation (Non Linear Equation Solving)
We consider a simple diode–resistor circuit:

     Vs ─────>|────── R ──
          │  Diode       │
          │              │
          └──────────────┘
                  |
                  GND

Kirchhoff’s Voltage Law (KVL):
    Vs - I*R - Vd = 0

Diode current (Shockley equation):
    I = Is * (exp(Vd / (n * Vt)) - 1)

Substituting gives:
    f(Vd) = Vs - R*Is*(exp(Vd / (n*Vt)) - 1) - Vd = 0

We solve f(Vd) = 0 for the diode voltage Vd using Newton–Raphson iteration.
"""

import math
import pandas as pd

# --- Given parameters ---
Vs = 5.0        # Supply voltage (V)
R = 1000.0      # Resistance (Ω)
Is = 1e-12      # Diode saturation current (A)
n = 1.0         # Ideality factor
Vt = 0.0259     # Thermal voltage (V)

# --- Define nonlinear function and derivative ---
def f(Vd):
    return Vs - R * Is * (math.exp(Vd / (n * Vt)) - 1) - Vd

def df(Vd):
    return -R * Is * (math.exp(Vd / (n * Vt)) / (n * Vt)) - 1

# --- Newton–Raphson iteration parameters ---
Vd = 0.7           # Initial guess (V)
tolerance = 1e-6
max_iter = 20

# --- Store iteration results for display ---
results = []

for i in range(max_iter):
    f_val = f(Vd)
    df_val = df(Vd)
    Vd_new = Vd - f_val / df_val
    results.append({
        "Iteration": i + 1,
        "Vd (V)": Vd_new,
        "Change (V)": abs(Vd_new - Vd)
    })
    if abs(Vd_new - Vd) < tolerance:
        Vd = Vd_new
        break
    Vd = Vd_new

# --- Compute final diode current ---
I = Is * (math.exp(Vd / (n * Vt)) - 1)

# --- Display iteration table ---
df_results = pd.DataFrame(results)
print("\nNewton–Raphson Iteration Results:\n")
print(df_results.to_string(index=False, float_format="%.6e"))

# --- Display final results ---
print("\nFinal Results:")
print(f"Diode voltage (Vd): {Vd:.6f} V")
print(f"Diode current (I): {I:.6e} A")
