r"""
NAME
    steady_state.py - Deterministic steady state of the TANK model with
    traditional and robotic capital, and a skill x participation
    (Ricardian/hand-to-mouth) household partition.

SYNOPSIS
    Execute at the Python prompt as >>> import steady_state,
    or run as `python3 steady_state.py` at the terminal.

AUTHOR
    Alvaro Mezquita Martinez

DESCRIPTION
    Solves eight equations in eight unknowns:
        (Z, K, N_S^r, N_U^r, theta_Z, theta_S, w_S, w_U)

    R1  robot factor-demand / no-arbitrage      (pins theta_Z, Z)
    R2  capital factor-demand / no-arbitrage    (pins K)
    R3  Ricardian skilled labour supply
    R4  Ricardian unskilled labour supply
    R5  skill premium target       Upsilon = w_S / w_U
    R6  robot income share target  Omega_Z = r_z * Zbar / Y
    R7  wage consistency, skilled    (closes the HtM <-> wage loop)
    R8  wage consistency, unskilled  (closes the HtM <-> wage loop)

    Modelling choices baked in below:
      - A = 1 (TFP normalisation)
      - D^ss = 0 (zero-profit fixed cost, per the model's own identity)
      - lambda_S = lambda_U = lambda  =>  n_S = 1-mu, n_U = mu
      - omega_S = omega_U = omega
"""

import numpy as np
from scipy.optimize import root


# Seed: [Z, K, N_S^r, N_U^r, theta_Z, theta_S, w_S, w_U]
seed0 = np.array([0.3712, 4.4998, 0.3896, 0.2802, 0.1583, 0.8206, 0.9001, 0.5295])   

def residuals(x, p):    
    Z, K, N_S_r, N_U_r, theta_Z, theta_S, w_S, w_U = x

    n_S, n_U = 1 - p['mu'], p['mu']        # from lambda_S = lambda_U = lam

    # --- hand-to-mouth hours, closed form, given the trial wages -------
    N_S_h = p['omega']**(-1/(p['phi']+p['gamma'])) * w_S**((1-p['gamma'])/(p['phi']+p['gamma']))
    N_U_h = p['omega']**(-1/(p['phi']+p['gamma'])) * w_U**((1-p['gamma'])/(p['phi']+p['gamma']))

    # --- population-weighted aggregates ---------------------------------
    Nbar_S = (1 - p['mu']) * ((1 - p['lam']) * N_S_r + p['lam'] * N_S_h)
    Nbar_U = p['mu']       * ((1 - p['lam']) * N_U_r + p['lam'] * N_U_h)
    Zbar   = (1 - p['lam']) * Z
    Kbar   = (1 - p['lam']) * K

    # --- nested CES / Cobb-Douglas technology, quantities-first --------
    rho_L = (p['sigma_L'] - 1) / p['sigma_L']
    rho_X = (p['sigma_X'] - 1) / p['sigma_X']

    L = (theta_Z * Zbar**rho_L + (1 - theta_Z) * Nbar_U**rho_L) ** (1/rho_L)
    X = (theta_S * Nbar_S**rho_X + (1 - theta_S) * L**rho_X) ** (1/rho_X)
    Y = p['A'] * Kbar**p['alpha'] * X**(1 - p['alpha'])

    MC = (p['eps_p'] - 1) / p['eps_p']

    # --- prices implied by the technology, forward from quantities -----
    P_X = (1 - p['alpha']) * MC * Y / X
    r_K_implied = p['alpha'] * MC * Y / Kbar

    P_L        = P_X * (1 - theta_S) * (X / L) ** (1 / p['sigma_X'])
    w_S_implied = P_X * theta_S       * (X / Nbar_S) ** (1 / p['sigma_X'])

    r_z_implied = P_L * theta_Z       * (L / Zbar)   ** (1 / p['sigma_L'])
    w_U_implied = P_L * (1 - theta_Z) * (L / Nbar_U) ** (1 / p['sigma_L'])

    # --- closed-form rentals from the household Euler equations --------
    r_K_closed = 1/p['beta'] - (1 - p['delta_k'])
    r_z_closed = p['p_Z_bar'] * (1/p['beta'] - (1 - p['delta_z']))

    # --- Ricardian consumption (per member), D^ss = 0 ------------------
    C_r = (n_S * w_S * N_S_r + n_U * w_U * N_U_r
           + (r_K_closed - p['delta_k']) * K
           + (r_z_closed - p['p_Z_bar'] * p['delta_z']) * Z)

    # ==================== RESIDUALS ====================
    r1 = r_z_implied - r_z_closed                              # robot no-arbitrage
    r2 = r_K_implied - r_K_closed                              # capital no-arbitrage
    r3 = p['omega'] * N_S_r**p['phi'] * C_r**p['gamma'] - w_S  # Ricardian skilled labour supply
    r4 = p['omega'] * N_U_r**p['phi'] * C_r**p['gamma'] - w_U  # Ricardian unskilled labour supply
    r5 = w_S / w_U - p['prem_target']                          # skill premium target
    r6 = r_z_closed * Zbar / Y - p['rshare_target']            # robot income share target
    r7 = w_S_implied - w_S                                     # wage consistency, skilled
    r8 = w_U_implied - w_U                                     # wage consistency, unskilled

    return np.array([r1, r2, r3, r4, r5, r6, r7, r8])


def solve(p, x0=seed0):
    """ 
    We search for the root of residuals defined in function "residuals(x,p)" as rX equations
    """
    sol = root(residuals, x0, args=(p,), tol=1e-10)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
        raise RuntimeError(f"steady state failed: {sol.message}")

    Z, K, N_S_r, N_U_r, theta_Z, theta_S, w_S, w_U = sol.x
    n_S, n_U = 1 - p['mu'], p['mu']

    N_S_h = p['omega']**(-1/(p['phi']+p['gamma'])) * w_S**((1-p['gamma'])/(p['phi']+p['gamma']))
    N_U_h = p['omega']**(-1/(p['phi']+p['gamma'])) * w_U**((1-p['gamma'])/(p['phi']+p['gamma']))

    C_S_h = p['omega']**(-1/(p['phi']+p['gamma'])) * w_S**((1+p['phi'])/(p['phi']+p['gamma']))
    C_U_h = p['omega']**(-1/(p['phi']+p['gamma'])) * w_U**((1+p['phi'])/(p['phi']+p['gamma']))

    Nbar_S = (1-p['mu']) * ((1-p['lam'])*N_S_r + p['lam']*N_S_h)
    Nbar_U = p['mu']     * ((1-p['lam'])*N_U_r + p['lam']*N_U_h)
    Zbar, Kbar = (1-p['lam'])*Z, (1-p['lam'])*K

    rho_L = (p['sigma_L']-1)/p['sigma_L']
    rho_X = (p['sigma_X']-1)/p['sigma_X']
    L = (theta_Z*Zbar**rho_L + (1-theta_Z)*Nbar_U**rho_L)**(1/rho_L)
    X = (theta_S*Nbar_S**rho_X + (1-theta_S)*L**rho_X)**(1/rho_X)
    Y = p['A'] * Kbar**p['alpha'] * X**(1-p['alpha'])
    MC = (p['eps_p']-1)/p['eps_p']

    r_K = 1/p['beta'] - (1-p['delta_k'])
    r_z = p['p_Z_bar'] * (1/p['beta'] - (1-p['delta_z']))

    C_r = (n_S*w_S*N_S_r + n_U*w_U*N_U_r
           + (r_K - p['delta_k'])*K + (r_z - p['p_Z_bar']*p['delta_z'])*Z)

    C = (1-p['lam'])*C_r + (1-p['mu'])*p['lam']*C_S_h + p['mu']*p['lam']*C_U_h
    Phi = (1 - MC) * Y
    I_K, I_Z = p['delta_k']*K, p['delta_z']*Z

    walras = Y - C - (1-p['lam'])*(I_K + p['p_Z_bar']*I_Z) - Phi

    # (inside solve(), just before building the ss dict)
    P_X = (1 - p['alpha']) * MC * Y / X               # outer Cobb-Douglas: P_X * X = (1-alpha)*MC*Y
    P_L = P_X * (1 - theta_S) * (X / L) ** (1 / p['sigma_X'])   # upper-nest conditional demand for L

    ss = dict(Z=Z, K=K, N_S_r=N_S_r, N_U_r=N_U_r, theta_Z=theta_Z, theta_S=theta_S,
              w_S=w_S, w_U=w_U, N_S_h=N_S_h, N_U_h=N_U_h, C_S_h=C_S_h, C_U_h=C_U_h,
              C_r=C_r, C=C, Y=Y, MC=MC, r_K=r_K, r_z=r_z, Phi=Phi,
              L=L, X=X, P_L=P_L, P_X=P_X, Zbar=Zbar, Kbar=Kbar, Nbar_S=Nbar_S, Nbar_U=Nbar_U,
              capital_share=p['alpha']*MC, robot_share=r_z*Zbar/Y,
              skill_premium=w_S/w_U, walras_residual=walras)
    return ss


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
        rho_R=0.8,    # Monetary policy lag (Smets-Wouters 2003 = 0.956)
        rshare_target = 0.1 # baseline robotic income regime
    )
    ss = solve(p=params)
    for k, v in ss.items():
        print(f"{k:16s} = {v: .6f}")
