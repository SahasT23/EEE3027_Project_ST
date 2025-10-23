import numpy as np

# Given parameters
Vs = 5.0        # Supply voltage (V)
R = 1000.0      # Resistance (Ω)
Is = 1e-12      # Saturation current (A)
n = 1.0         # Ideality factor
Vt = 0.026      # Thermal voltage (V)

def diode_current(Vd):
    """Calculate diode current given voltage"""
    return Is * (np.exp(Vd / (n * Vt)) - 1)

def f(Vd):
    """Function we want to zero: KVL equation"""
    I = diode_current(Vd)
    return Vs - R * I - Vd

def f_derivative(Vd):
    """Derivative of f for Newton-Raphson"""
    return -R * Is * (1 / (n * Vt)) * np.exp(Vd / (n * Vt)) - 1

def newton_raphson(initial_guess, tolerance=1e-6, max_iterations=50):
    """
    Newton-Raphson iterative solver
    """
    Vd = initial_guess
    iterations = []
    
    for i in range(max_iterations):
        # Calculate function value and derivative
        f_val = f(Vd)
        f_prime = f_derivative(Vd)
        
        # Calculate current for this Vd
        I = diode_current(Vd)
        VR = I * R
        
        # Store iteration data
        iterations.append({
            'iteration': i,
            'Vd': Vd,
            'I': I,
            'VR': VR,
            'f_val': f_val,
            'error': abs(f_val)
        })
        
        # Check convergence
        if abs(f_val) < tolerance:
            print(f"Converged in {i+1} iterations")
            break
        
        # Newton-Raphson update
        Vd_new = Vd - f_val / f_prime
        
        # Ensure Vd stays in reasonable range
        if Vd_new < 0:
            Vd_new = Vd / 2  # Bisection fallback
        elif Vd_new > Vs:
            Vd_new = (Vd + Vs) / 2
        
        Vd = Vd_new
    
    return Vd, I, iterations

def simple_bisection(Vd_low=0.5, Vd_high=1.0, tolerance=1e-6, max_iterations=50):
    """
    Simple bisection method for comparison
    """
    iterations = []
    
    for i in range(max_iterations):
        # Try midpoint
        Vd = (Vd_low + Vd_high) / 2
        
        # Calculate function value
        f_val = f(Vd)
        I = diode_current(Vd)
        VR = I * R
        
        # Store data
        iterations.append({
            'iteration': i,
            'Vd': Vd,
            'I': I,
            'VR': VR,
            'f_val': f_val,
            'error': abs(f_val)
        })
        
        # Check convergence
        if abs(f_val) < tolerance:
            print(f"Bisection converged in {i+1} iterations")
            break
        
        # Update bounds
        if f_val > 0:
            Vd_low = Vd  # Need higher Vd
        else:
            Vd_high = Vd  # Need lower Vd
    
    return Vd, I, iterations

# Solve using Newton-Raphson
print("=" * 60)
print("NEWTON-RAPHSON METHOD")
print("=" * 60)
Vd_nr, I_nr, iter_nr = newton_raphson(initial_guess=0.7)

print(f"\nFinal Solution:")
print(f"  Vd = {Vd_nr:.6f} V")
print(f"  I  = {I_nr*1000:.6f} mA")
print(f"  VR = {I_nr*R:.6f} V")
print(f"\nVerification (should equal Vs = {Vs}V):")
print(f"  VR + Vd = {I_nr*R + Vd_nr:.6f} V")
print(f"  Error = {abs(Vs - (I_nr*R + Vd_nr)):.6f} V")

print("\n\nIteration History:")
print(f"{'Iter':<6} {'Vd (V)':<12} {'I (mA)':<12} {'VR (V)':<12} {'f(Vd)':<12} {'Error':<12}")
print("-" * 72)
for it in iter_nr:
    print(f"{it['iteration']:<6} {it['Vd']:<12.6f} {it['I']*1000:<12.6f} "
          f"{it['VR']:<12.6f} {it['f_val']:<12.6f} {it['error']:<12.6f}")

# Solve using Bisection
print("\n\n" + "=" * 60)
print("BISECTION METHOD")
print("=" * 60)
Vd_bis, I_bis, iter_bis = simple_bisection()

print(f"\nFinal Solution:")
print(f"  Vd = {Vd_bis:.6f} V")
print(f"  I  = {I_bis*1000:.6f} mA")
print(f"  VR = {I_bis*R:.6f} V")

print("\n\nIteration History:")
print(f"{'Iter':<6} {'Vd (V)':<12} {'I (mA)':<12} {'VR (V)':<12} {'f(Vd)':<12} {'Error':<12}")
print("-" * 72)
for it in iter_bis:
    print(f"{it['iteration']:<6} {it['Vd']:<12.6f} {it['I']*1000:<12.6f} "
          f"{it['VR']:<12.6f} {it['f_val']:<12.6f} {it['error']:<12.6f}")

print("\n" + "=" * 60)
print(f"Newton-Raphson: {len(iter_nr)} iterations")
print(f"Bisection: {len(iter_bis)} iterations")
print("=" * 60)
