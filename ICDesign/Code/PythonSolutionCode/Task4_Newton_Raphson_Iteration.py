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

We solve f(Vd) = 0 for the diode voltage Vd using Newton–Raphson iteration,
and generate a cobweb diagram showing the iterative process.
"""

"""
EEE3027: Integrated Circuit Design and Embedded Systems
Task 4 – SPICE Under the Hood

Newton–Raphson Iteration for Diode Circuit Simulation
Improved Cobweb Diagram (y = x vs. Newton Update Function)
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# --- Store iteration results for table and plotting ---
results = []
Vd_values = [Vd]

for i in range(max_iter):
    f_val = f(Vd)
    df_val = df(Vd)
    Vd_new = Vd - f_val / df_val
    results.append({
        "Iteration": i + 1,
        "Vd (V)": Vd_new,
        "Change (V)": abs(Vd_new - Vd)
    })
    Vd_values.append(Vd_new)
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
print("\nFinal Results:")
print(f"Diode voltage (Vd): {Vd:.6f} V")
print(f"Diode current (I): {I:.6e} A")

# --- Cobweb diagram setup ---
x = np.linspace(0.5, 0.8, 400)
g = [val - f(val)/df(val) for val in x]  # Newton update function
y_line = x                               # y = x line

# Plot setup
plt.figure(figsize=(8, 6))
plt.plot(x, g, 'b', label=r'$g(V_d) = V_d - \frac{f(V_d)}{f\'(V_d)}$')
plt.plot(x, y_line, 'k--', label=r'$y = V_d$')

# Plot cobweb path
for i in range(len(Vd_values) - 1):
    x0, x1 = Vd_values[i], Vd_values[i+1]
    # vertical line to g(x)
    plt.plot([x0, x0], [x0, x1], color='red', lw=1.2)
    # horizontal line to y = x
    plt.plot([x0, x1], [x1, x1], color='red', lw=1.2)
    plt.text(x0, x1, f"$V_{{{i}}}$", fontsize=8, ha='right', va='bottom')

# Final converged point
plt.scatter(Vd_values[-1], Vd_values[-1], color='green', s=60, zorder=5, label='Converged Point')

plt.title("Newton–Raphson Cobweb Diagram for Diode Equation")
plt.xlabel(r"$V_d$ (V)")
plt.ylabel(r"$V_{d+1}$ (V)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("newton_cobweb_diagram.png", dpi=300)
plt.show()

print("\nCobweb diagram saved as 'newton_cobweb_diagram.png'.")
