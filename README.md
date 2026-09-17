# TANK model with robotic capital

Python implementation of a Two-Agent New Keynesian (TANK) model with traditional and robotic capital, investment adjustment costs, and a skill × asset-market-participation household partition (skilled/unskilled, Ricardian/hand-to-mouth). The model studies how automation shapes monetary policy transmission, with a central result that unskilled hand-to-mouth households bear disproportionately larger consumption losses following a contractionary monetary policy shock.

Companion code to the working paper "TANK model with robotic capital" (Álvaro Mezquita Martínez).

## Repository structure

- `code/` — the Python replication package described below.
- `mezquita_2026.pdf` — the working paper.

## Files

All paths below are relative to `code/`.

- `steady_state.py` — deterministic steady state. Solves eight equations (robot and capital no-arbitrage, Ricardian labour supply ×2, skill-premium target, robot-income-share target, wage consistency ×2) in eight unknowns via `scipy.optimize.root`. Exposes `solve(p, x0=seed0)`.
- `irf_klein.py` — builds the log-linear system (`A E_t[x_{t+1}] = B x_t + C eps_t`) and solves it with the Klein (2000) QZ method. Exposes `run_regime(base_params, overrides=None, x0=None, shock_index=None, T=40)`, which chains the steady state, the linear system, and the impulse responses for one calibration.
- `main_tank_robots.py` — the paper's main experiment: a sweep across automation regimes (robot income share $\Omega_Z$), warm-started regime to regime, comparing impulse responses to a contractionary monetary policy shock. Produces the paper's figures and impact tables.

Each file also runs standalone (`python3 <file>.py`) with its own minimal calibration dictionary, for isolated testing — `main_tank_robots.py`'s `PARAMS` dictionary is the authoritative calibration for the paper's results; the dictionaries in the other two files' `if __name__ == "__main__":` blocks are for standalone checks only and are not the source of truth.

## Requirements

```
pip install -r code/requirements.txt
```

Python ≥3.9, with `numpy`, `pandas`, `scipy`, `matplotlib`.

By default, plots use matplotlib's own text rendering (no external dependencies). `main_tank_robots.py` has an optional LaTeX-rendered style block (Palatino via `mathpazo`, `text.usetex=True`), commented out near the top of the script — uncomment it for camera-ready figures if a local TeX installation is available (e.g. `texlive-latex-extra` and `texlive-fonts-extra` on Debian/Ubuntu, or a full MacTeX/MiKTeX install).

## Usage

```
cd code
python3 main_tank_robots.py
```

Runs the full four-regime sweep ($\Omega_Z \in \{0.05, 0.15, 0.25, 0.35\}$), writing figures (`wage_divergence.pdf`, `robot_investment.pdf`, `regime_irfs.pdf`) and tables (`regime_paths.csv`, `regime_steady_states.csv`) to `./output/`.

To run a single calibration directly:

```python
from irf_klein import run_regime
from main_tank_robots import PARAMS

out = run_regime(PARAMS, overrides={'rshare_target': 0.10})
paths = out['paths']       # IRF panel, one column per model variable
ss = out['ss']             # steady state
```

## Citation

If you use this code, please cite the working paper (full citation TBD pending venue).

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.
