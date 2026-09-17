r"""
NAME
    irf_klein.py - Log-linear system and Klein (2000) IRF solve for the
    TANK model with traditional/robotic capital, adjustment costs, and
    the four-cell (skill x participation) household structure.

SYNOPSIS
    Execute at the Python prompt as >>> import irf_klein,
    or run as `python3 irf_klein.py` at the terminal.

AUTHOR
    Alvaro Mezquita Martinez

DESCRIPTION
    A E_t[x_{t+1}] = B x_t + C eps_t,  x = [predetermined | jump]

    Predetermined (6): a, pz, k, z, iKlag, iZlag
    Jump (26):         cr, nSr, nUr, nSh, CSh, nUh, CUh, qk, qz, rk, rz,
                       wS, wU, pX, pL, Lc, X, nbarS, nbarU, y, mc, pi,
                       Rn, Cagg, iK, iZ

    Notes: NKPC is written with zeta_p = 0 baked in (matches the current
    calibration), so no pi(-1) term / no extra state for it. Revisit
    if zeta_p is ever calibrated away from zero.
"""

import numpy as np
import pandas as pd
from scipy.linalg import ordqz
import matplotlib.pyplot as plt

from steady_state import solve as solve_ss, seed0

# 1. VARIABLE ORDERING ===============================

PRE = ['a', 'pz', 'k', 'z', 'iKlag', 'iZlag', 'RnLag']
JUMP = ['cr', 'nSr', 'nUr', 'nSh', 'CSh', 'nUh', 'CUh', 'qk', 'qz', 'rk', 'rz',
        'wS', 'wU', 'pX', 'pL', 'Lc', 'X', 'nbarS', 'nbarU', 'y', 'mc', 'pi',
        'Rn', 'Cagg', 'iK', 'iZ']
VARS = PRE + JUMP
SHOCKS = ['epsA', 'epsR', 'epsP']

N_PRE = len(PRE)
N = len(VARS)
IDX = {name: i for i, name in enumerate(VARS)} # Variable name + ID number
SIDX = {name: i for i, name in enumerate(SHOCKS)} # Shock name + ID number
SHOCK_SD_PARAM = {'epsA': 'sd_eps_a', 'epsR': 'sd_eps_r', 'epsP': 'sigma_p'}  # shock -> its s.d. parameter

# 2.SHARES FROM THE STEADY STATE ===============================

def derive_shares(ss, p):
    """
    Cost/consumption/expenditure shares and aggregation weights
    needed as coefficients in the log-linear system.
    """
    P_L, P_X = ss['P_L'], ss['P_X']

    s_S = ss['w_S'] * ss['Nbar_S'] / (ss['w_S'] * ss['Nbar_S'] + P_L * ss['L'])
    s_Z = ss['r_z'] * ss['Zbar']  / (ss['r_z'] * ss['Zbar']  + ss['w_U'] * ss['Nbar_U'])

    I_K, I_Z = p['delta_k'] * ss['K'], p['delta_z'] * ss['Z']
    s_C  = ss['C'] / ss['Y']
    s_IK = (1 - p['lam']) * I_K / ss['Y']
    s_IZ = (1 - p['lam']) * p['p_Z_bar'] * I_Z / ss['Y']

    chi_r  = (1 - p['lam']) * ss['C_r'] / ss['C']
    chi_hS = (1 - p['mu']) * p['lam'] * ss['C_S_h'] / ss['C']
    chi_hU = p['mu']       * p['lam'] * ss['C_U_h'] / ss['C']

    denom_S = (1 - p['lam']) * ss['N_S_r'] + p['lam'] * ss['N_S_h']
    denom_U = (1 - p['lam']) * ss['N_U_r'] + p['lam'] * ss['N_U_h']
    omega_Sr, omega_Sh = (1 - p['lam']) * ss['N_S_r'] / denom_S, p['lam'] * ss['N_S_h'] / denom_S
    omega_Ur, omega_Uh = (1 - p['lam']) * ss['N_U_r'] / denom_U, p['lam'] * ss['N_U_h'] / denom_U

    kappa_f = p['beta'] / (1 + p['beta'] * p['zeta_p'])
    kappa_p = (1 - p['theta_p']) * (1 - p['beta'] * p['theta_p']) / (p['theta_p'] * (1 + p['beta'] * p['zeta_p']))

    return dict(s_S=s_S, s_Z=s_Z, s_C=s_C, s_IK=s_IK, s_IZ=s_IZ,
                chi_r=chi_r, chi_hS=chi_hS, chi_hU=chi_hU,
                omega_Sr=omega_Sr, omega_Sh=omega_Sh,
                omega_Ur=omega_Ur, omega_Uh=omega_Uh,
                kappa_f=kappa_f, kappa_p=kappa_p, P_L=P_L, P_X=P_X)

# 3.BUILDING SYSTEM MATRICES ===============================


def build_system(p, s):
    A, B = np.zeros((N, N)), np.zeros((N, N))
    C = np.zeros((N, len(SHOCKS)))
    row = [0]  # mutable counter, closed over by add_eq

    def add_eq(lead=None, curr=None, shock=None):
        r = row[0]
        for name, coef in (lead or {}).items():
            A[r, IDX[name]] = coef
        for name, coef in (curr or {}).items():
            B[r, IDX[name]] = coef
        for name, coef in (shock or {}).items():
            C[r, SIDX[name]] = coef
        row[0] += 1

    beta, gamma, phi = p['beta'], p['gamma'], p['phi']
    delta_k, delta_z = p['delta_k'], p['delta_z']
    tau_K, tau_Z, rho_p, rho_A = p['tau_K'], p['tau_Z'], p['rho_p'], p['rho_A']
    alpha, sigma_X, sigma_L = p['alpha'], p['sigma_X'], p['sigma_L']
    phi_pi, phi_x, rho_R = p['phi_pi'], p['phi_x'], p['rho_R']
    sS, sZ = s['s_S'], s['s_Z']

    # --- Ricardian household ---
    add_eq(lead={'cr': 1, 'pi': 1/gamma}, curr={'cr': 1, 'Rn': 1/gamma})               # bond Euler
    add_eq(curr={'nSr': phi, 'cr': gamma, 'wS': -1})                                   # labour supply, skilled
    add_eq(curr={'nUr': phi, 'cr': gamma, 'wU': -1})                                   # labour supply, unskilled

    # --- Hand-to-mouth ---
    add_eq(curr={'CSh': 1, 'wS': -1, 'nSh': -1})                                       # HtM budget, skilled
    add_eq(curr={'nSh': phi, 'CSh': gamma, 'wS': -1})                                  # HtM labour supply, skilled
    add_eq(curr={'CUh': 1, 'wU': -1, 'nUh': -1})                                       # HtM budget, unskilled
    add_eq(curr={'nUh': phi, 'CUh': gamma, 'wU': -1})                                  # HtM labour supply, unskilled

    # --- Investment FOCs (CEE adjustment cost) ---
    add_eq(lead={'iK': beta*tau_K},
           curr={'iK': tau_K + beta*tau_K, 'iKlag': -tau_K, 'qk': -1})                 # FOC, I_K
    add_eq(lead={'iZ': beta*tau_Z},
           curr={'iZ': tau_Z + beta*tau_Z, 'iZlag': -tau_Z, 'qz': -1, 'pz': 1})        # FOC, I_Z
    add_eq(lead={'RnLag':1}, curr={'Rn':1})

    # --- Valuation (Tobin's Q) conditions ---
    add_eq(lead={'cr': 1, 'rk': -(1/gamma)*(1-beta*(1-delta_k)), 'qk': -(1/gamma)*beta*(1-delta_k)},
           curr={'cr': 1, 'qk': -(1/gamma)})                                           # valuation, K
    add_eq(lead={'cr': 1, 'rz': -(1/gamma)*(1-beta*(1-delta_z)), 'qz': -(1/gamma)*beta*(1-delta_z)},
           curr={'cr': 1, 'qz': -(1/gamma)})                                           # valuation, Z

    # --- Accumulation / exogenous processes ---
    add_eq(lead={'k': 1}, curr={'k': 1-delta_k, 'iK': delta_k})                        # K accumulation
    add_eq(lead={'z': 1}, curr={'z': 1-delta_z, 'iZ': delta_z})                        # Z accumulation
    add_eq(lead={'pz': 1}, curr={'pz': rho_p}, shock={'epsP': 1})                      # p^Z process
    add_eq(lead={'a': 1}, curr={'a': rho_A}, shock={'epsA': 1})                        # TFP process
    add_eq(lead={'iKlag': 1}, curr={'iK': 1})                                          # lag carrier, I_K
    add_eq(lead={'iZlag': 1}, curr={'iZ': 1})                                          # lag carrier, I_Z

    # --- Firm: nested CES/Cobb-Douglas cost minimisation ---
    add_eq(curr={'pL': 1, 'rz': -sZ, 'wU': -(1-sZ)})                                   # unit cost, L (lower nest)
    add_eq(curr={'z': 1, 'Lc': -1, 'rz': sigma_L, 'pL': -sigma_L})                     # demand, Z (pins r_z)
    add_eq(curr={'nbarU': 1, 'Lc': -1, 'wU': sigma_L, 'pL': -sigma_L})                 # demand, N_U bar
    add_eq(curr={'pX': 1, 'wS': -sS, 'pL': -(1-sS)})                                   # unit cost, X (upper nest)
    add_eq(curr={'nbarS': 1, 'X': -1, 'wS': sigma_X, 'pX': -sigma_X})                  # demand, N_S bar
    add_eq(curr={'Lc': 1, 'X': -1, 'pL': sigma_X, 'pX': -sigma_X})                     # demand, L (upper nest)
    add_eq(curr={'k': 1, 'y': -1, 'mc': -1, 'rk': 1})                                  # demand, K bar (pins r_K)
    add_eq(curr={'X': 1, 'y': -1, 'mc': -1, 'pX': 1})                                  # demand, X (outer)
    add_eq(curr={'mc': 1, 'a': 1, 'rk': -alpha, 'pX': -(1-alpha)})                     # marginal cost

    # --- Aggregation and market clearing ---
    add_eq(curr={'nbarS': 1, 'nSr': -s['omega_Sr'], 'nSh': -s['omega_Sh']})             # labour agg., skilled
    add_eq(curr={'nbarU': 1, 'nUr': -s['omega_Ur'], 'nUh': -s['omega_Uh']})             # labour agg., unskilled
    add_eq(curr={'Cagg': 1, 'cr': -s['chi_r'], 'CSh': -s['chi_hS'], 'CUh': -s['chi_hU']})  # aggregate consumption
    add_eq(curr={'y': 1, 'Cagg': -s['s_C'], 'iK': -s['s_IK'], 'iZ': -s['s_IZ'], 'pz': -s['s_IZ']})  # resource constraint

    # --- Phillips curve and policy (placeholder Taylor rule) ---
    add_eq(lead={'pi': s['kappa_f']}, curr={'pi': 1, 'mc': -s['kappa_p']})              # NKPC (zeta_p = 0)
    add_eq(curr={'Rn': 1, 'RnLag': -rho_R,
                 'pi': -(1-rho_R)*phi_pi, 'y': -(1-rho_R)*phi_x},
           shock={'epsR': -1})                                                        # Taylor rule, SW-2003 smoothing

    assert row[0] == N, f"expected {N} equations, wrote {row[0]}"
    return A, B, C

# 4.KLEIN SOLVE & IRF ===============================

def klein_solve(A, B, n_pre):
    """Model: A E_t[x_{t+1}] = B x_t  (Klein's Gamma0 = A, Gamma1 = B).
    Klein's eigenvalues are diag(T)/diag(S) where B = Q T Z^H, A = Q S Z^H.
    scipy's ordqz(X, Y) reports alpha/beta = diag(X-transform)/diag(Y-transform),
    so calling ordqz(B, A) — not ordqz(A, B) — makes scipy's own alpha/beta
    equal Klein's eigenvalue directly, with no extra inversion needed."""
    T, S, alpha_, beta_, Q, Z = ordqz(B, A, sort='iuc')
    safe = np.abs(beta_) > 1e-12
    eigvals = np.full_like(alpha_, np.inf, dtype=complex)
    eigvals[safe] = alpha_[safe] / beta_[safe]
    n_stable = np.sum(np.abs(eigvals) < 1)
    assert n_stable == n_pre, (
        f"Blanchard-Kahn fails: {n_stable} stable roots, {n_pre} predetermined vars"
    )

    Z11, Z21 = Z[:n_pre, :n_pre], Z[n_pre:, :n_pre]
    S11, T11 = S[:n_pre, :n_pre], T[:n_pre, :n_pre]
    Z11_inv = np.linalg.inv(Z11)
    F = np.real(Z21 @ Z11_inv)
    P = np.real(Z11 @ np.linalg.inv(S11) @ T11 @ Z11_inv)
    return P, F, eigvals

def shock_impact(A, B, F, n_pre):
    """Solve jointly for:
       G — how each shock moves NEXT period's predetermined state
       M — how each shock moves THIS period's jump variables (impact)
    from  A @ [g; F@g] = B @ [0; m] + C@eps, for each shock column of C.
    """
    n_jump = A.shape[0] - n_pre
    I_pre, I_jump = np.eye(n_pre), np.eye(n_jump)

    Ag = A @ np.vstack([I_pre, F])                              # N x n_pre
    Bm = B @ np.vstack([np.zeros((n_pre, n_jump)), I_jump])     # N x n_jump
    lhs = np.hstack([Ag, -Bm])                                   # N x N

    def solve_for(Cmat):
        sol = np.linalg.solve(lhs, Cmat)          # (n_pre+n_jump) x n_shocks
        return sol[:n_pre], sol[n_pre:]            # G, M
    return solve_for

def compute_irf(P, F, A, B, C, shock_index, n_pre, T=40, size=1.0):
    solve_for = shock_impact(A, B, F, n_pre)
    G, M = solve_for(C)
    eps = np.zeros(C.shape[1]); eps[shock_index] = size

    s = np.zeros((T, n_pre))
    j = np.zeros((T, F.shape[0]))

    j[0] = M @ eps                 # jump variables respond immediately
    s[1] = G @ eps                 # state only starts moving from t=1
    j[1] = F @ s[1]
    for t in range(2, T):
        s[t] = P @ s[t-1]
        j[t] = F @ s[t]
    return s, j

# 5. SOLVING DYNAMIC SYSTEM ====================

def run_regime(base_params, overrides=None, x0=None, shock_index=None, T=40):
    """Solve the steady state, build and solve the log-linear system, and
    compute the IRF for one calibration. `overrides` patches ss_params
    (e.g. {'rshare_target': 0.10}) without mutating the module-level dict.
    `x0` is an optional steady-state seed (e.g. the previous regime's
    solution, for warm-starting a sweep). Returns enough to chain into the
    next call and to reconstruct every panel/table downstream."""
    p = dict(base_params)
    if overrides:
        p.update(overrides)
    ss = solve_ss(p=p, x0=x0 if x0 is not None else seed0)
    shares = derive_shares(ss, p)
    A, B, Cmat = build_system(p, shares)
    P, F, eig = klein_solve(A, B, N_PRE)

    if shock_index is None:
        shock_index = SIDX['epsR']
    size = p[SHOCK_SD_PARAM[SHOCKS[shock_index]]]
    s, jumps = compute_irf(P, F, A, B, Cmat, shock_index=shock_index, n_pre=N_PRE, T=T, size=size)

    paths = pd.DataFrame(np.hstack([s, jumps]), columns=VARS)

    # converged steady-state vector, in seed0's order, for warm-starting the next regime
    x0_next = np.array([ss['Z'], ss['K'], ss['N_S_r'], ss['N_U_r'],
                         ss['theta_Z'], ss['theta_S'], ss['w_S'], ss['w_U']])

    return dict(params=p, ss=ss, shares=shares, A=A, B=B, C=Cmat,
                P=P, F=F, eig=eig, s=s, jumps=jumps, paths=paths, x0=x0_next)


if __name__ == "__main__":

    params = dict(
        beta=0.99, gamma=0.75, phi=1/0.621, # intertemporal discount, relative risk aversion, and labor elasticity 
        alpha=0.30,      # capital income share, outer nest 
        sigma_X=0.90,     # upper-nest elasticity  
        sigma_L=10,     # lower-nest elasticity
        delta_k=0.025, delta_z=0.04, # traditional & robotic capital depreciation
        mu=0.24, lam=0.24,           # lambda_S = lambda_U = lam  =>  n_S = 1-mu, n_U = mu 0.24
        omega=9.0,       # labor disutility weight  
        eps_p=6.0, theta_p=0.75, zeta_p=0.0, 
        p_Z_bar=1.0,      # robots relative price 
        A=1.0,           # total factor productivity
        prem_target=1.7, # fixed skill premium
        tau_K=6.962,     # Smets-Wouters (2003) euro-area posterior mean, adjustment cost curvature
        tau_Z=6.962,     # same curvature applied to robotic investment  
        rho_p=0.90,      # persistence, p^Z process     
        rho_A=0.90,      # persistence, TFP process
        sigma_p=0.01,    # s.d., p^Z innovation — placeholder, confirm value
        sd_eps_a=0.01,   # s.d., TFP innovation
        sd_eps_r=0.01,   # s.d., monetary innovation
        phi_pi=1.5,      # inflation weight in monetary policy
        phi_x=0.125,     # output gap weight in monetary policy
        rho_R=0.8    # Monetary policy lag (Smets-Wouters 2003 = 0.956)
    )
    out = run_regime(base_params=params)
    paths = out['paths']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, var, title in zip(axes, ['y', 'pi', 'Rn'], ['Output', 'Inflation', 'Policy rate']):
        ax.plot(paths[var].iloc[1:].values)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'{title} (t≥1)')
    plt.tight_layout(); plt.show()
