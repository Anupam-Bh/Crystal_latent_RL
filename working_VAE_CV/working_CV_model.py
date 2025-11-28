import numpy as np
import matplotlib.pyplot as plt

# ---- Constants ----
F = 96485.33212
R = 8.314462618
T_DEFAULT = 298.15

# =============================================================================
# Water-splitting helpers (HER/OER)
# =============================================================================
def eq_potential_HER(pH: float, reference: str = "RHE", T: float = T_DEFAULT) -> float:
    """
    Equilibrium potential for HER at temperature T.
    reference: "RHE" (0 V by definition) or "SHE" (shifts with pH).
    """
    if reference.upper() == "RHE":
        return 0.0
    # SHE reference (Nernst: E = -0.05916 * pH at 298 K)
    return -(R * T / F) * np.log(10.0) * pH

def eq_potential_OER(pH: float, reference: str = "RHE", T: float = T_DEFAULT, pO2_bar: float = 1.0) -> float:
    """
    Equilibrium potential for OER at temperature T and oxygen partial pressure.
    At 298 K and pO2=1 bar:
        - vs RHE: ~1.229 V (independent of pH)
        - vs SHE: 1.229 - 0.05916 * pH
    """
    E0_298 = 1.229  # V at 25 C, standard states
    if reference.upper() == "RHE":
        # Include pO2 dependence (small unless pO2 != 1 bar)
        return E0_298 + (R * T / (4.0 * F)) * np.log(pO2_bar)
    # SHE reference
    return (E0_298 - (R * T / F) * np.log(10.0) * pH) + (R * T / (4.0 * F)) * np.log(pO2_bar)

def her_current_density_oneway(i0, n, alpha, E, Eeq, T):
    """Cathodic-only HER: negative current for E < Eeq, ~0 otherwise."""
    if i0 <= 0:
        return 0.0
    f = n * F / (R * T)
    eta = E - Eeq
    if eta >= 0:
        return 0.0
    # cathodic Tafel branch from BV: i_c = -i0 * exp(-alpha f eta)
    return -i0 * np.exp(-alpha * f * eta)

def oer_current_density_oneway(i0, n, alpha, E, Eeq, T):
    """Anodic-only OER: positive current for E > Eeq, ~0 otherwise."""
    if i0 <= 0:
        return 0.0
    f = n * F / (R * T)
    eta = E - Eeq
    if eta <= 0:
        return 0.0
    # anodic Tafel branch from BV: i_a = +i0 * exp((1-alpha) f eta)
    return i0 * np.exp((1.0 - alpha) * f * eta)

# def butler_volmer_current_density(i0: float, n: int, alpha: float, E: float, Eeq: float, T: float) -> float:
#     """
#     Symmetric (single alpha) Butler-Volmer current density (A/m^2).
#     Sign convention: Positive current is anodic.
#     """
#     if i0 <= 0.0:
#         return 0.0
#     f = n * F / (R * T)
#     eta = E - Eeq
#     # Full BV: i = i0 * [exp((1-alpha) f eta) - exp(-alpha f eta)]
#     return i0 * (np.exp((1.0 - alpha) * f * eta) - np.exp(-alpha * f * eta))

# =============================================================================
# Original multi-couple CV simulator (extended with HER/OER channels)
# + New waveform: 0 -> E_max -> E_vertex -> 0
# =============================================================================

# Example redox couples (you can modify these)
couples = [
    dict(label="A/A+",  n=1,  E0=0.00, DO=1e-3,  DR=1e-3,  CObulk=1.0, CRbulk=0.0, k0=None,  alpha=0.5),
    dict(label="B/B-",  n=2,  E0=-0.25,DO=7e-4, DR=9e-4,  CObulk=0.5, CRbulk=0.0, k0=1e-5, alpha=0.5),
]

# Global parameters
p = dict(
    # Electrode / waveform
    A=7e-6,                 # m^2 (~0.07 cm^2)
    E_max=1,              # V (from 0 -> E_max)
    E_vertex=-0.2,          # V (then E_max -> E_vertex -> 0)
    scan_rate=0.010,        # V/s
    n_cycles=3,

    # Double-layer
    Cdl=1.0e-3,             # F/m^2 (set 0 to disable)

    # Numerics
    Nx=200,
    x_pad_factor=6.0,
    dt_safety=0.8,

    # Temperature
    T=298.15,               # K

    # Reference scale & pH (matter only for HER/OER E_eq calculation)
    reference="RHE",        # "RHE" or "SHE"
    pH=7.0,

    # HER kinetic channel (surface-only, no transport)
    HER=dict(
        enabled=True,
        n=2,
        i0=7e-1,            # A/m^2  (tune for your system)
        alpha=0.45,
        pH=2.4,            # None -> use p['pH']
        p_gas_bar=1.0,      # H2 partial pressure (not used in HER Eeq here)
        label="HER",
    ),

    # OER kinetic channel (surface-only, no transport)
    OER=dict(
        enabled=True,
        n=4,
        i0=1e-10,            # A/m^2  (typically much smaller than HER)
        alpha=0.12,
        pH=2.5,            # None -> use p['pH']
        pO2_bar=1.0,        # O2 partial pressure for Eeq
        label="OER",
    ),
)

# ---- Three-segment waveform: 0 -> E_max -> E_vertex -> 0 ----
def three_segment_waveform(t, p):
    """
    Vectorized periodic waveform:
        segment 1: 0          -> E_max     at +/−v
        segment 2: E_max      -> E_vertex  at +/−v
        segment 3: E_vertex   -> 0         at +/−v
    """
    v = p["scan_rate"]
    E0 = 0.0
    Emax = p["E_max"]
    Ever = p["E_vertex"]

    t1 = abs(Emax - E0) / v
    t2 = abs(Ever - Emax) / v
    t3 = abs(E0   - Ever) / v
    Tcyc = t1 + t2 + t3

    s1 = v * np.sign(Emax - E0)
    s2 = v * np.sign(Ever - Emax)
    s3 = v * np.sign(E0 - Ever)

    tc = np.mod(t, Tcyc)
    E = np.empty_like(t)

    m1 = tc < t1
    m2 = (tc >= t1) & (tc < t1 + t2)
    m3 = ~ (m1 | m2)

    E[m1] = E0    + s1 * (tc[m1])
    E[m2] = Emax  + s2 * (tc[m2] - t1)
    E[m3] = Ever  + s3 * (tc[m3] - (t1 + t2))
    return E, (t1, t2, t3)

# ------ Surface boundary functions (when DO /= DR) ----
def surface_nernst_unequal(E, CO1, CR1, n, E0, DO, DR, T=298.15):
    """
    Reversible (Nernstian) boundary with DO != DR.
    Enforces: (i) Nernst ratio CO0/CR0 = exp(nF(E-E0)/RT),
              (ii) flux equality DO(CO1-CO0) = DR(CR0-CR1) (no net accumulation at the interface).
    Returns CO0, CR0.
    """
    f = n * F / (R * T)
    r = np.exp(f * (E - E0))             # CO0/CR0
    num = DO * CO1 + DR * CR1
    den = DO * r + DR
    CR0 = num / den
    CO0 = r * CR0
    return max(CO0, 0.0), max(CR0, 0.0)

def surface_BV_unequal(E, CO1, CR1, n, E0, DO, DR, k0, alpha, dx, T=298.15):
    """
    Quasi-reversible (BV) boundary with DO != DR.
    Enforces: (i) O-side flux = BV rate: DO(CO1-CO0)/dx = k0[CO0*exp(-alpha f eta) - CR0*exp((1-alpha) f eta)]
              (ii) Flux equality: DO*CO0 + DR*CR0 = DO*CO1 + DR*CR1
    Solves a 2x2 linear system for CO0, CR0 without iteration.
    """
    f = n * F / (R * T)
    eta = E - E0
    ef = np.exp(-alpha * f * eta)
    eb = np.exp((1.0 - alpha) * f * eta)

    # Linear system A @ [CO0, CR0]^T = b
    a11 = -DO / dx - k0 * ef
    a12 =  k0 * eb
    b1  = -DO * CO1 / dx

    a21 = DO
    a22 = DR
    b2  = DO * CO1 + DR * CR1

    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-30:
        det = 1e-30  # regularize
    CO0 = ( b1 * a22 - a12 * b2) / det
    CR0 = ( a11 * b2 - b1  * a21) / det

    return max(CO0, 0.0), max(CR0, 0.0)

# ---- Simulator ----
def simulate_cv_multi_with_watersplitting(p, couples):
    """
    Extends the multi-couple diffusion CV with two additional *kinetic-only* channels:
    - HER (hydrogen evolution reaction)
    - OER (oxygen evolution reaction)
    These channels contribute Faradaic current but do not track transport of H2/O2.

    Waveform per cycle:
        0 V  -> E_max    -> E_vertex -> 0 V
    """
    # --- timing from waveform ---
    n_cycles = p.get("n_cycles", 1)
    v = p["scan_rate"]
    E0 = 0.0
    Emax = p["E_max"]
    Ever = p["E_vertex"]

    t1 = abs(Emax - E0) / v
    t2 = abs(Ever - Emax) / v
    t3 = abs(E0   - Ever) / v
    t_cycle = t1 + t2 + t3
    t_total = t_cycle * n_cycles

    # --- grids and numerics ---
    Nx = p.get("Nx", 300)
    x_pad = p.get("x_pad_factor", 6.0)
    dt_safety = p.get("dt_safety", 0.8)
    A = p.get("A", 1e-6)
    Cdl = p.get("Cdl", 0.0)
    T = p.get("T", T_DEFAULT)
    reference = p.get("reference", "RHE")
    pH_global = p.get("pH", 7.0)

    # stability based on the largest D across all couples
    Dmax = max(max(c["DO"], c["DR"]) for c in couples) if couples else 1e-9
    x_max = x_pad * np.sqrt(Dmax * t_total)
    x = np.linspace(0.0, x_max, Nx)
    dx = x[1] - x[0]
    dt_max = dx**2 / (2 * Dmax)
    dt = dt_safety * dt_max
    Nt = int(np.ceil(t_total / dt))
    dt = t_total / Nt
    t = np.linspace(0.0, t_total, Nt + 1)

    # Waveform + dE/dt
    E, _seg_times = three_segment_waveform(t, p)
    dEdt = np.zeros_like(E)
    dEdt[1:] = np.diff(E) / dt
    dEdt[0] = dEdt[1]

    # --- allocate concentrations: shape (Nc, Nx) for CO and CR ---
    Nc = len(couples)
    if Nc > 0:
        CO = np.stack([np.full(Nx, c["CObulk"]) for c in couples])
        CR = np.stack([np.full(Nx, c["CRbulk"]) for c in couples])
        lamO = np.array([c["DO"] for c in couples]) * dt / (dx * dx)
        lamR = np.array([c["DR"] for c in couples]) * dt / (dx * dx)
    else:
        CO = np.zeros((0, Nx))
        CR = np.zeros((0, Nx))
        lamO = np.zeros((0,))
        lamR = np.zeros((0,))

    # Currents
    IF_components = np.zeros((Nc, len(E)))
    IF_total = np.zeros_like(E)
    IC = Cdl * A * dEdt

    # Water splitting channels
    HER_params = p.get("HER", {})
    OER_params = p.get("OER", {})
    her_enabled = bool(HER_params.get("enabled", False))
    oer_enabled = bool(OER_params.get("enabled", False))

    I_HER = np.zeros_like(E)
    I_OER = np.zeros_like(E)

    # Precompute Eeq for HER/OER (constant if reference/pH don't change)
    pH_HER = HER_params.get("pH", pH_global)
    pH_OER = OER_params.get("pH", pH_global)
#    Eeq_HER = eq_potential_HER(pH_HER, reference=reference, T=T)
#    Eeq_OER = eq_potential_OER(pH_OER, reference=reference, T=T, pO2_bar=OER_params.get("pO2_bar", 1.0))
    Eeq_HER = 0.0      ### modified to ease randomization 
    Eeq_OER = 1.229

    for k, Ek in enumerate(E):
        # Far boundary (bulk) & surface boundary for each redox couple
        for i, c in enumerate(couples):
            # bulk
            CO[i, -1] = c["CObulk"]
            CR[i, -1] = c["CRbulk"]
            # surface
            if c.get("k0", None) is None:
                CO0, CR0 = surface_nernst_unequal(Ek, CO[i, 1], CR[i, 1],
                                                  c["n"], c["E0"], c["DO"], c["DR"], T=T)
            else:
                CO0, CR0 = surface_BV_unequal(Ek, CO[i, 1], CR[i, 1],
                                              c["n"], c["E0"], c["DO"], c["DR"],
                                              c["k0"], c.get("alpha", 0.5), dx, T=T)
            CO[i, 0] = CO0
            CR[i, 0] = CR0
 
            ## Couple Faradaic current
            #j_i = c["DO"] * (CO[i, 1] - CO[i, 0]) / dx   # mol m^-2 s^-1
            #IF_components[i, k] = c["n"] * F * A * j_i
            
            # --- Consistent anodic-positive current for O + ne- <-> R ---
            # Fick fluxes at the interface (into solution is negative gradient)
            jO = -c["DO"] * (CO[i, 1] - CO[i, 0]) / dx  # mol m^-2 s^-1, O-flux (into solution)
            jR = -c["DR"] * (CR[i, 1] - CR[i, 0]) / dx  # mol m^-2 s^-1, R-flux (into solution)

            # Net current density with ANODIC positive:
            # i = nF [ -D_O dC_O/dx + D_R dC_R/dx ]  => i = nF ( jO - jR )
            i_density = c["n"] * F * (jO - jR)          # A m^-2
            IF_components[i, k] = A * i_density         # A

        # Water splitting kinetic currents (surface-only)
        if her_enabled:
            i_density_her = her_current_density_oneway(p['HER']['i0'], p['HER']['n'], p['HER']['alpha'], Ek, Eeq_HER, T)
            I_HER[k] = A * i_density_her

        if oer_enabled:
            i_density_oer = oer_current_density_oneway(p['OER']['i0'], p['OER']['n'], p['OER']['alpha'], Ek, Eeq_OER, T)
            I_OER[k] = A * i_density_oer

        # if her_enabled:
        #     i0 = HER_params.get("i0", 0.0)
        #     n_her = int(HER_params.get("n", 2))
        #     alpha_her = HER_params.get("alpha", 0.5)
        #     #print(f'HER parameters: i0:{i0}, n_her:2, alpha: {alpha_her},E_eq:{Eeq_HER}, T:298 \n')
        #     i_density_her = butler_volmer_current_density(i0, n_her, alpha_her, Ek, Eeq_HER, T)
        #     I_HER[k] = A * i_density_her * (-1)

        # if oer_enabled:
        #     i0 = OER_params.get("i0", 0.0)
        #     n_oer = int(OER_params.get("n", 4))
        #     alpha_oer = OER_params.get("alpha", 0.5)
        #     #print(f'OER parameters: i0:{i0}, n_her:4, alpha: {alpha_oer},E_eq:{Eeq_OER}, T:298 \n\n')
        #     i_density_oer = butler_volmer_current_density(i0, n_oer, alpha_oer, Ek, Eeq_OER, T)
        #     I_OER[k] = A * i_density_oer * (-1)

        IF_total[k] = IF_components[:, k].sum() + I_HER[k] + I_OER[k]

        # advance diffusion (explicit FTCS) for each couple
        if k < len(E) - 1 and Nc > 0:
            CO[:, 1:-1] = CO[:, 1:-1] + lamO[:, None] * (CO[:, 2:] - 2 * CO[:, 1:-1] + CO[:, :-2])
            CR[:, 1:-1] = CR[:, 1:-1] + lamR[:, None] * (CR[:, 2:] - 2 * CR[:, 1:-1] + CR[:, :-2])

    I = IF_total + IC
#    ##Return only lat cycle
#    Ln = round(k /n_cycles)
#    return E[-Ln:], I[-Ln:], IF_total[-Ln:], IF_components, IC, I_HER, I_OER, x, t, CO, CR
#    #return E, I, IF_total, IF_components, IC, I_HER, I_OER, x, t, CO, CR
    # --- Return only the last cycle cleanly (matching array lengths) ---
    samples_per_cycle = max(1, len(E) // n_cycles)
    #print(len(E),samples_per_cycle, n_cycles, E[-samples_per_cycle-1:-samples_per_cycle+1])
    E_last   = E[-samples_per_cycle:]
    I_last   = I[-samples_per_cycle:]
    IF_last  = IF_total[-samples_per_cycle:]
    IFc_last = IF_components[:, -samples_per_cycle:]
    IC_last  = IC[-samples_per_cycle:]
    IHER_last = I_HER[-samples_per_cycle:]
    IOER_last = I_OER[-samples_per_cycle:]
    t_last = t[-samples_per_cycle:]

    return E_last, I_last, IF_last, IFc_last, IC_last, IHER_last, IOER_last, x, t_last, CO, CR


if __name__ == "__main__":
    # Run a demo
    E, I, IF, IF_comp, IC, I_HER, I_OER, x, t, CO, CR = simulate_cv_multi_with_watersplitting(p, couples)
    #print(E)
    #print(I)
    #print(len(E))

    # Plot: total + components (including HER/OER)
    plt.figure()
    plt.plot(E, I, label="Total")

    # Diffusion-coupled redox components
#    for i, c in enumerate(couples):
#        plt.plot(E, IF_comp[i], ls="--", label=c.get("label", f"couple_{i}"))
#
#    # Water-splitting components
#    if p["HER"]["enabled"]:
#        plt.plot(E, I_HER, ls=":", label=p["HER"].get("label", "HER"))
#    if p["OER"]["enabled"]:
#        plt.plot(E, I_OER, ls="-.", label=p["OER"].get("label", "OER"))

    plt.xlabel("Potential / V")
    plt.ylabel("Current / A")
    plt.legend()
#    plt.grid(True)
    plt.tight_layout()
    plt.savefig('a.png',dpi=400)
    # plt.show()
