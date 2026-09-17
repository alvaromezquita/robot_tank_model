r"""
NAME
    main_tank_robots.py - Automation-intensity regime sweep, replicating
    the thesis experimental design of Section on regime comparison.

DESCRIPTION
    Regimes are indexed by the robot income share of output, Omega_Z.

    Each regime is solved with the previous regime's converged steady 
    state as the warm start, sweeping from the least- to the most-automated 
    economy, since this is the path already validated in the thesis write-up.

    Shock: a one-standard-deviation contractionary monetary policy shock
    (epsR).

    Output: a 2x3 panel figure of t>=1 impulse responses (aggregates row:
    output, inflation, policy rate; distributional row: skilled and
    unskilled hand-to-mouth consumption), colour-coded from light to dark
    red as Omega_Z rises, plus two t=0 impact tables (aggregates,
    distributional) with one column per regime.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from irf_klein import run_regime 

# 1.PRELIMINARIES ========================

# FILE LOCATION 

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTDIR, exist_ok=True)

# PLOT CONFIGURATION [OPTIONAL] (Requires TeX installed locally)
r"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "text.latex.preamble": r"\usepackage{mathpazo}",
    "font.size": 11
})
"""

# SETTING REGIME GRID 

REGIMES = [0.05, 0.15, 0.25, 0.35]   # Omega_Z


# PARAMETERS 

PARAMS = dict(
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

# 2.IRFs COMPUTATION ========================

# SWEEP (warm-started, out from the seed's own basin) 

results, x0 = {}, None 

for omega_z in REGIMES:
    results[omega_z] = run_regime(PARAMS, overrides={'rshare_target': omega_z}, x0=x0) #Computing and storing dynamic paths
    x0 = results[omega_z]['x0'] #Warm start 

# Normalisation 
RN_TARGET = 0.0025 # 25bp target

for oz in REGIMES:
    scale = RN_TARGET / results[oz]['paths']['Rn'].iloc[0]
    results[oz]['scale'] = scale
    results[oz]['paths'] = results[oz]['paths'] * scale
    results[oz].pop('s', None)
    results[oz].pop('jumps', None) 

# 3.PLOTS ========================

# COLOUR SCALE
cmap = matplotlib.colormaps['Reds']
n = len(REGIMES)
colors = {oz: cmap(0.35 + 0.60 * i / (n - 1)) for i, oz in enumerate(REGIMES)} # The darker the red, the more automated the economy is.


def impact(var):
    """Impact (t=0) response of `var`, one entry per regime, in REGIMES order."""
    return np.array([results[oz]['paths'][var].iloc[0] for oz in REGIMES])


# FIGURE 1: WAGE DIVERGENCE (Section: mechanism) 

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(REGIMES, impact('wS'), marker='o', ls='--', color=cmap(0.45),
        label='Skilled wage, $\\hat{w}_S$')
ax.plot(REGIMES, impact('wU'), marker='s', ls='-', color=cmap(0.85),
        label='Unskilled wage, $\\hat{w}_U$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Robot income share, $\\Omega_Z$')
ax.set_ylabel('Impact response')
ax.set_xticks(REGIMES)
ax.legend(fontsize=9)
ax.set_title('Impact wage responses by automation regime', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'wage_divergence.pdf'), dpi=150)
plt.close(fig)

# FIGURE 2: ROBOTIC INVESTMENT (Section: savers) 

fig, ax = plt.subplots(figsize=(6.0, 4.2))
for oz in REGIMES:
    ax.plot(results[oz]['paths']['iZ'].values, color=colors[oz],
            label=f'$\\Omega_Z$ = {oz:.2f}')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('Quarters since shock')
ax.set_ylabel('Response')
ax.legend(fontsize=9)
ax.set_title('Robotic investment, $\\hat{\\imath}_Z$', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'robot_investment.pdf'), dpi=150)
plt.close(fig)

# FIGURE 3: AGGREGATES (Section: dynamics) 

HORIZON = 12       # beyond ~t=6 all four regimes sit on top of each other
AGG_VARS = ['Rn', 'y', 'pi']
AGG_LABELS = ['Policy rate', 'Output', 'Inflation']

fig, axes = plt.subplots(1, len(AGG_VARS), figsize=(4.3*len(AGG_VARS), 3.8), sharex=True)
for ax, var, label in zip(axes, AGG_VARS, AGG_LABELS):
    for oz in REGIMES:
        ax.plot(results[oz]['paths'][var].values[:HORIZON], color=colors[oz],
                label=f'$\\Omega_Z$ = {oz:.2f}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Quarters since shock')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.99, 0.14), fontsize=9)
fig.suptitle('IRFs to a contractionary monetary shock, by automation regime')
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUTDIR, 'regime_irfs.pdf'), dpi=150)
plt.close(fig)

# 4.PRINTING AND EXPORTING TABLES ========================

# IMPACT TABLES (t=0)

AGG_TAB_VARS = ['Rn', 'y', 'pi', 'mc']
AGG_TAB_LABELS = ['Policy rate', 'Output', 'Inflation', 'Real marginal cost']
DIST_VARS = ['cr', 'CSh', 'CUh', 'wS', 'wU', 'nSh', 'nUh']
DIST_LABELS = ['Ricardian consumption', 'Skilled HtM consumption', 'Unskilled HtM consumption',
               'Skilled wage', 'Unskilled wage', 'Skilled HtM hours', 'Unskilled HtM hours']


def impact_table(varlist, labels):
    data = {f'{oz:.2f}': [results[oz]['paths'][v].iloc[0] for v in varlist] for oz in REGIMES}
    return pd.DataFrame(data, index=labels)


agg_table = impact_table(AGG_TAB_VARS, AGG_TAB_LABELS)
dist_table = impact_table(DIST_VARS, DIST_LABELS)

agg_table.index.name = 'Aggregates (impact, t=0)'
dist_table.index.name = 'Distributional (impact, t=0)'

print(agg_table.round(4).to_string())
print()
print(dist_table.round(4).to_string())

all_paths = []
for oz in REGIMES:
    df = results[oz]['paths'].copy()
    df.insert(0, 't', np.arange(len(df)))
    df.insert(0, 'Omega_Z', oz)
    all_paths.append(df)
all_paths = pd.concat(all_paths, ignore_index=True)
all_paths.to_csv(os.path.join(OUTDIR, 'regime_paths.csv'), index=False) # Exporting  dynamics results as .csv

# STEADY STATES

ss_table = pd.DataFrame({oz: results[oz]['ss'] for oz in REGIMES}).T
ss_table.index.name = 'Omega_Z'
ss_table.to_csv(os.path.join(OUTDIR, 'regime_steady_states.csv')) # Exporting steady state results

