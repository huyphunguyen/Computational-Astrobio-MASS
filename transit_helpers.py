"""
Provides:
  - load_systems_from_yaml()  : build SYSTEMS dict by reading sim_systems/*.yaml
  - load_psls_dat()           : read & downsample a PSLS .dat file
  - detrend_flux()            : running-median detrend
  - transit_depth_ppm()       : (Rp/Rs)^2 × 1e6
  - verdict()                 : Promising / Marginal / Non-promising label
"""

from __future__ import annotations

import glob
import os

import numpy as np
import yaml
from scipy.ndimage import median_filter

# ── Constants ──────────────────────────────────────────────────────────────────
RJ_TO_RE   = 11.209          # Jupiter radii → Earth radii
BLS_STRIDE = 72              # 25 s × 72 = 30 min cadence


def _rstar_from_teff(teff: float) -> float:
    """Approximate main-sequence R_star [R_sun] from T_eff using simple lookup."""
    if teff >= 6000:
        return 1.30   # F-dwarf
    if teff >= 5200:
        return 1.00   # G-dwarf
    if teff >= 4500:
        return 0.72   # K-dwarf
    return 0.60       # late K / early M


def load_systems_from_yaml(sim_dir: str) -> dict:
    """
    Build the SYSTEMS ground-truth dict by parsing every *.yaml in sim_dir.

    Each entry contains:
      p_inj, a_au, rp_rj, rp_re, teff, rstar, mag,
      star_type, science_case

    Fields are read directly from YAML keys:
      Transit.OrbitalPeriod      → p_inj
      Transit.PlanetSemiMajorAxis → a_au
      Transit.PlanetRadius        → rp_rj   (PSLS units = R_Jupiter)
      Star.Teff                   → teff
      Star.Mag                    → mag
      (rstar inferred from Teff)
    """
    yaml_paths = sorted(glob.glob(os.path.join(sim_dir, '*.yaml')))
    if not yaml_paths:
        raise FileNotFoundError(f'No YAML files found in {sim_dir}')

    systems = {}
    for path in yaml_paths:
        cfg_name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as fh:
            cfg = yaml.safe_load(fh)

        star    = cfg.get('Star', {})
        transit = cfg.get('Transit', {})

        teff   = float(star.get('Teff', 5772.0))
        mag    = float(star.get('Mag', 10.0))
        rstar  = _rstar_from_teff(teff)

        p_inj  = float(transit.get('OrbitalPeriod', 0.0))
        a_au   = float(transit.get('PlanetSemiMajorAxis', 0.0))
        rp_rj  = float(transit.get('PlanetRadius', 0.0))

        meta = cfg.get('Metadata', {})
        star_type    = meta.get('star_type', 'Unknown')
        science_case = meta.get('science_case', 'Unknown')

        systems[cfg_name] = {
            'p_inj':        p_inj,
            'a_au':         a_au,
            'rp_rj':        rp_rj,
            'rp_re':        rp_rj * RJ_TO_RE,
            'teff':         teff,
            'rstar':        rstar,
            'mag':          mag,
            'star_type':    star_type,
            'science_case': science_case,
        }

    return systems


# ── Light-curve helpers ────────────────────────────────────────────────────────

def load_psls_dat(path: str, stride: int = BLS_STRIDE):
    """
    Read a PSLS .dat file, keep flag==0 rows, downsample by stride.
    Returns (time_days, flux_relative).
    """
    data = np.genfromtxt(path, comments='#')
    mask = data[:, 2] == 0
    data_down = data[mask][::stride]
    time_days = data_down[:, 0] / 86400.0
    flux      = 1.0 + data_down[:, 1] * 1e-6
    return time_days, flux


def detrend_flux(
    time_days: np.ndarray,
    flux: np.ndarray,
    window_days: float = 1.0,
) -> np.ndarray:
    """
    Remove stellar variability with a running median filter.
    window_days should be > transit duration (~0.2 d) but << orbital period.
    Divides flux by the smoothed baseline to preserve transit dips.
    """
    dt = float(np.median(np.diff(time_days)))
    kernel = max(int(round(window_days / dt)) | 1, 3)
    if kernel % 2 == 0:
        kernel += 1
    baseline = median_filter(flux, size=kernel, mode='reflect')
    return flux / baseline


def transit_depth_ppm(rp_re: float, rstar_rsun: float) -> float:
    """(Rp/Rs)^2 × 1e6  [ppm]."""
    R_EARTH_M = 6.371e6
    R_SUN_M   = 6.957e8
    return float(((rp_re * R_EARTH_M) / (rstar_rsun * R_SUN_M)) ** 2 * 1e6)


def verdict(detected: bool, in_hz: bool, near_hz: bool = False) -> str:
    """Return 'Promising', 'Marginal', or 'Non-promising'."""
    if detected and (in_hz or near_hz):
        return 'Promising'
    if detected:
        return 'Marginal'
    return 'Non-promising'
