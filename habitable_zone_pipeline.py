"""
Habitable-zone target selection pipeline aligned with Project 10 (PLATO / PlatoSim).

PlatoSim/PLATOnium is the intended simulator (picsim → varsim → payload → platonium).
When those products are unavailable, this project uses **PSLS** (PLATO Solar-like
Light-curve Simulator, `pip install psls`): `sls.gen_up` produces solar-like
background variability (granulation, oscillations, photometry noise), onto which we
inject a Mandel–Agol-style geometric transit.

This module provides:
  - Physics: L*/L☉, incident flux S (Earth units), equilibrium temperature T_eq
  - Habitable-zone screening and transparent ranking score
  - Optional pure-synthetic box transits (`generate_synthetic_plato_like_lightcurve`)

Replace light-curve generation with real PlatoSim/PSLS exports when available
(time, flux columns) via `load_time_flux_csv`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import R_sun, au as AU

# ---------------------------------------------------------------------------
# Physical constants (SI where needed)
# ---------------------------------------------------------------------------

T_SUN_K = 5772.0
SIGMA_S = 1.0  # width for H_HZ(S) Gaussian around Earth-like irradiation (tunable)
SIGMA_R = 0.5  # Earth radii scale for H_Rp
SIGMA_DET = 0.15  # maps SNR to [0,1]-like detection term


@dataclass
class Star:
    """Host star (solar units for radius; Kelvin for Teff)."""

    name: str
    radius_solar: float  # R*/R☉
    teff_k: float  # T*

    def luminosity_solar(self) -> float:
        """L*/L☉ = (R*/R☉)² (T*/T☉)⁴."""
        return (self.radius_solar**2) * (self.teff_k / T_SUN_K) ** 4


@dataclass
class Planet:
    """Planet in injected truth table."""

    name: str
    radius_earth: float  # R_p / R⊕
    period_days: float
    semi_major_axis_au: float
    impact_parameter: float = 0.0  # 0–1, for optional duration tweak
    albedo: float = 0.3


@dataclass
class PlanetarySystem:
    """One system: star + planets (design sample for Tasks 2–3)."""

    star: Star
    planets: list[Planet]
    label: str = ""
    noise_multiplier: float = 1.0  # >1 = harder case (Task scope)


# ---------------------------------------------------------------------------
# Task 5 — Habitability-related quantities
# ---------------------------------------------------------------------------


def stellar_luminosity_solar(radius_solar: float, teff_k: float) -> float:
    """L*/L☉ = (R*/R☉)² (T*/T☉)⁴."""
    return (radius_solar**2) * (teff_k / T_SUN_K) ** 4


def incident_flux_earth_units(luminosity_solar: float, a_au: float) -> float:
    """S = (L*/L☉) / (a/AU)² — incident flux relative to Earth at 1 AU for the Sun."""
    if a_au <= 0:
        raise ValueError("semi-major axis must be positive")
    return luminosity_solar / (a_au**2)


def equilibrium_temperature_k(
    teff_k: float,
    r_star_m: float,
    a_m: float,
    albedo: float = 0.3,
) -> float:
    """
    T_eq = T* sqrt(R*/(2a)) (1-A)^(1/4) with R* and a in the same length unit.
    """
    if a_m <= 0 or r_star_m <= 0:
        raise ValueError("radius and semi-major axis must be positive")
    return float(teff_k * np.sqrt(r_star_m / (2.0 * a_m)) * ((1.0 - albedo) ** 0.25))


def equilibrium_temperature_from_au(
    teff_k: float,
    radius_solar: float,
    a_au: float,
    albedo: float = 0.3,
) -> float:
    """Convenience: R* in R☉, a in AU."""
    r_m = radius_solar * R_sun.to_value(u.m)
    a_m = a_au * AU.to_value(u.m)
    return equilibrium_temperature_k(teff_k, r_m, a_m, albedo=albedo)


# ---------------------------------------------------------------------------
# Task 6 — Habitable zone (working criterion: flux + T_eq diagnostics)
# ---------------------------------------------------------------------------


def habitable_zone_flags(
    s_earth: float,
    teq_k: float,
    s_inner: float = 0.95,
    s_outer: float = 1.37,
    teq_inner_k: float = 180.0,
    teq_outer_k: float = 270.0,
) -> dict[str, Any]:
    """
    Approximate conservative HZ screen: stellar flux and T_eq bands.
    These are screening tools, not evidence of habitability.
    """
    in_flux_hz = s_inner <= s_earth <= s_outer
    in_teq_band = teq_inner_k <= teq_k <= teq_outer_k
    near_hz = (0.7 <= s_earth <= 1.6) or (160 <= teq_k <= 300)
    return {
        "in_habitable_zone_flux": bool(in_flux_hz),
        "in_habitable_zone_teq_band": bool(in_teq_band),
        "near_habitable_zone_loose": bool(near_hz),
        "s_earth": s_earth,
        "teq_k": teq_k,
    }


def hz_flux_annulus_au(
    luminosity_solar: float,
    s_inner: float = 0.95,
    s_outer: float = 1.37,
) -> tuple[float, float]:
    """
    Habitable-zone radii (AU) from the same flux band as `habitable_zone_flags`.
    S = (L*/L☉)/(a/AU)² ⇒ a = √(L/S). Inner HZ edge (hotter) at S = s_outer;
    outer edge (cooler) at S = s_inner.
    """
    if luminosity_solar <= 0:
        raise ValueError("luminosity must be positive")
    r_in = float(np.sqrt(luminosity_solar / s_outer))
    r_out = float(np.sqrt(luminosity_solar / s_inner))
    return r_in, r_out


def plot_star_planet_system(
    sys: PlanetarySystem,
    planet_index: int = 0,
    *,
    s_inner: float = 0.95,
    s_outer: float = 1.37,
    ax: Any | None = None,
    figsize: tuple[float, float] = (6.5, 6.5),
    planet_phase_deg: float = 40.0,
):
    """
    Top-down schematic: star, green flux-based HZ annulus, planet on circular orbit.

    Stellar/planet disks are *schematic* (not to scale with orbit).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    star = sys.star
    if planet_index < 0 or planet_index >= len(sys.planets):
        raise IndexError("planet_index out of range")
    planet = sys.planets[planet_index]
    lum = star.luminosity_solar()
    r_hz_in, r_hz_out = hz_flux_annulus_au(lum, s_inner=s_inner, s_outer=s_outer)
    a = planet.semi_major_axis_au

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # HZ annulus (polygon: outer circle, inner hole)
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    xo, yo = r_hz_out * np.cos(theta), r_hz_out * np.sin(theta)
    xi, yi = r_hz_in * np.cos(theta[::-1]), r_hz_in * np.sin(theta[::-1])
    ax.fill(
        np.r_[xo, xi],
        np.r_[yo, yi],
        color="#2ca02c",
        alpha=0.35,
        zorder=1,
        label="Habitable zone (flux band)",
    )

    # Reference circles
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            r_hz_in,
            fill=False,
            ls=":",
            lw=1.0,
            color="#1f5c1f",
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            r_hz_out,
            fill=False,
            ls=":",
            lw=1.0,
            color="#1f5c1f",
            zorder=2,
        )
    )

    # Planet orbit
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            a,
            fill=False,
            ls="--",
            lw=1.2,
            color="0.35",
            zorder=2,
            label="Planet orbit",
        )
    )

    rad = np.radians(planet_phase_deg)
    px, py = a * np.cos(rad), a * np.sin(rad)

    # Star (symbolic size, not physical)
    r_star_draw = max(0.02 * max(r_hz_out, a), 0.015)
    ax.add_patch(Circle((0.0, 0.0), r_star_draw, color="#f4d03f", ec="#b7950b", lw=1.2, zorder=5))
    ax.plot(0.0, 0.0, marker="", color="none")  # anchor
    ax.text(
        0.0,
        0.0,
        star.name,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        zorder=6,
        color="#5d4e37",
    )

    # Planet marker (symbolic size)
    ax.scatter(
        [px],
        [py],
        s=120,
        c="#3498db",
        ec="#1a5276",
        lw=1.2,
        zorder=6,
        label=f'Planet {planet.name} (a = {a:.3f} AU)',
    )

    lim = max(r_hz_out, a) * 1.18
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    s_here = incident_flux_earth_units(lum, a)
    teq = equilibrium_temperature_from_au(
        star.teff_k, star.radius_solar, a, albedo=planet.albedo
    )
    title = f"{sys.label} — $L_*={lum:.3f}\\,L_\\odot$,  $S={s_here:.3f}$ $S_\\oplus$,  $T_{{eq}}={teq:.0f}$ K"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    if created_fig:
        plt.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Task 7 — Transparent ranking model (recommended framework)
# ---------------------------------------------------------------------------


def h_hz_score(s_earth: float, sigma_s: float = SIGMA_S) -> float:
    """H_HZ = exp(-(S-1)² / (2 σ_S²))."""
    return float(np.exp(-((s_earth - 1.0) ** 2) / (2.0 * sigma_s**2)))


def h_rp_score(radius_earth: float, sigma_r: float = SIGMA_R) -> float:
    """H_Rp = exp(-(R_p - 1.2)² / (2 σ_R²)) — favors Earth/super-Earth sizes."""
    return float(np.exp(-((radius_earth - 1.2) ** 2) / (2.0 * sigma_r**2)))


def h_host_score(teff_k: float) -> float:
    """Reward Sun-like hosts: Gaussian around 5800 K (simple interpretable form)."""
    t0, w = 5800.0, 800.0
    return float(np.exp(-((teff_k - t0) ** 2) / (2.0 * w**2)))


def h_det_score(snr_proxy: float, sigma_det: float = SIGMA_DET) -> float:
    """Map detection strength to [0,1]; use BLS power or SNR-like proxy."""
    x = float(np.clip(snr_proxy, 0.0, 20.0))
    return float(1.0 - np.exp(-x / max(sigma_det * 10.0, 1e-6)))


def h_stab_score(eccentricity: float | None) -> float:
    """Optional: prefer low e."""
    if eccentricity is None:
        return 0.5
    return float(np.exp(-((eccentricity) ** 2) / (2.0 * 0.1**2)))


def habitability_rank_score(
    s_earth: float,
    radius_earth: float,
    teff_k: float,
    snr_proxy: float,
    eccentricity: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Score = H_HZ + 0.2 H_Rp + 0.15 H_host + 0.15 H_det + 0.1 H_stab
    (document typo tH_host / tH_det corrected to H_host / H_det).
    """
    w = weights or {
        "H_HZ": 1.0,
        "H_Rp": 0.2,
        "H_host": 0.15,
        "H_det": 0.15,
        "H_stab": 0.1,
    }
    hhz = h_hz_score(s_earth)
    hrp = h_rp_score(radius_earth)
    hh = h_host_score(teff_k)
    hd = h_det_score(snr_proxy)
    hs = h_stab_score(eccentricity)
    total = (
        w["H_HZ"] * hhz
        + w["H_Rp"] * hrp
        + w["H_host"] * hh
        + w["H_det"] * hd
        + w["H_stab"] * hs
    )
    return {
        "H_HZ": hhz,
        "H_Rp": hrp,
        "H_host": hh,
        "H_det": hd,
        "H_stab": hs,
        "rank_score": float(total),
    }


def load_time_flux_csv(
    path: str,
    time_col: str = "time",
    flux_col: str = "flux",
    time_unit_days: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PlatoSim / export CSV with columns for time and flux (normalized or raw).
    Adjust `time_unit_days` if times are in seconds (e.g. time_unit_days = 1/86400).
    """
    df = pd.read_csv(path)
    t = np.asarray(df[time_col], dtype=float) * time_unit_days
    f = np.asarray(df[flux_col], dtype=float)
    return t, f


# ---------------------------------------------------------------------------
# Light curves: PSLS (preferred for Project 10) + simple synthetic fallback
# ---------------------------------------------------------------------------

LightCurveBackend = Literal["psls", "synthetic"]


def _stable_star_id(label: str, planet: str, lc_seed: int, run_seed: int) -> int:
    """Deterministic PSLS star id (avoid salted `hash()` across interpreter runs)."""
    payload = f"{label}|{planet}|{lc_seed}|{run_seed}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) % (10**9 - 1) + 1


def _transit_depth_from_radii(rp_re: float, r_star_solar: float) -> float:
    """Approximate (R_p/R_*)^2 with R_p in R_earth, R_star in R_sun."""
    r_earth_m = 6.371e6
    r_sun_m = R_sun.to_value(u.m)
    rp_m = rp_re * r_earth_m
    rs_m = r_star_solar * r_sun_m
    return float((rp_m / rs_m) ** 2)


def _transit_duration_days(period_days: float) -> float:
    """Rough ingress+egress duration scale for BLS-friendly grids."""
    duration_days = 0.05 * (period_days / 10.0) ** (1.0 / 3.0)
    return float(np.clip(duration_days, 0.08, 0.45))


def apply_box_transit_to_flux(
    flux: np.ndarray,
    time_days: np.ndarray,
    period_days: float,
    t0_days: float,
    rp_earth: float,
    r_star_solar: float,
) -> np.ndarray:
    """Multiply flux by (1 - depth) in-transit bins (relative flux units)."""
    depth = _transit_depth_from_radii(rp_earth, r_star_solar)
    dur = _transit_duration_days(period_days)
    phase = ((time_days - t0_days) / period_days) % 1.0
    half = 0.5 * dur / period_days
    in_transit = (phase < half) | (phase > 1.0 - half)
    out = np.asarray(flux, dtype=float, copy=True)
    out[in_transit] *= 1.0 - depth
    return out


def estimate_numax_mu_hz_solar_like(teff_k: float) -> float:
    """
    Order-of-magnitude ν_max for FGK dwarfs (for `sls.gen_up`).
    Piecewise linear between rough literature anchors.
    """
    t = float(np.clip(teff_k, 4800.0, 6300.0))
    t_lo, nu_lo = 5200.0, 2200.0
    t_mid, nu_mid = 5777.0, 3090.0
    t_hi, nu_hi = 6000.0, 3500.0
    if t <= t_mid:
        return float(nu_lo + (t - t_lo) * (nu_mid - nu_lo) / (t_mid - t_lo))
    return float(nu_mid + (t - t_mid) * (nu_hi - nu_mid) / (t_hi - t_mid))


def generate_psls_lightcurve(
    time_days: np.ndarray,
    period_days: float,
    t0_days: float,
    rp_earth: float,
    r_star_solar: float,
    teff_k: float,
    *,
    magnitude: float = 12.0,
    seed: int | None = None,
    star_id: int = 1,
    workdir: str = "",
    oscillation: bool = True,
    granulation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PSLS `gen_up`: stochastic solar-like background (ppm) on a uniform grid,
    converted to relative flux, interpolated to `time_days`, then box transit injected.

    Writes auxiliary files under `workdir` (modes list); use a unique directory per run.
    """
    try:
        import sls  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "PSLS is not installed. Run: pip install psls"
        ) from e

    td = np.asarray(time_days, dtype=float)
    if td.size < 2:
        raise ValueError("time_days must have at least two points")
    dt_s = float((td[-1] - td[0]) / max(td.size - 1, 1)) * 86400.0
    if dt_s <= 0:
        raise ValueError("Non-positive timestep in time_days")
    t_span_days = float(td[-1] - td[0] + dt_s / 86400.0)
    numax = estimate_numax_mu_hz_solar_like(teff_k)
    path_prefix = workdir if (workdir.endswith(os.sep) or workdir == "") else workdir + os.sep

    # time [s], ts [ppm]; writes modes file under path_prefix
    time_s, ts, *_rest = sls.gen_up(
        int(star_id),
        float(numax),
        dt_s,
        t_span_days,
        float(magnitude),
        seed=seed,
        teff=float(teff_k),
        path=path_prefix,
        verbose=False,
        oscillation=oscillation,
        granulation=granulation,
    )

    time_days_psls = (np.asarray(time_s, dtype=float) - float(time_s[0])) / 86400.0
    flux_rel = 1.0 + np.asarray(ts, dtype=float) * 1e-6
    flux_on_grid = np.interp(td, time_days_psls, flux_rel, left=flux_rel[0], right=flux_rel[-1])
    flux_final = apply_box_transit_to_flux(
        flux_on_grid, td, period_days, t0_days, rp_earth, r_star_solar
    )
    return td, flux_final


def generate_synthetic_plato_like_lightcurve(
    time_days: np.ndarray,
    period_days: float,
    t0_days: float,
    rp_earth: float,
    r_star_solar: float,
    noise_ppm: float,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple box transit + Gaussian noise (development / comparison only).
    """
    rng = np.random.default_rng(seed)
    flux = np.ones_like(time_days, dtype=float)
    flux = apply_box_transit_to_flux(
        flux, time_days, period_days, t0_days, rp_earth, r_star_solar
    )
    noise = noise_ppm * 1e-6
    flux += rng.normal(0.0, noise, size=flux.shape)
    return time_days, flux


def run_bls_recovery(
    time_days: np.ndarray,
    flux: np.ndarray,
    period_min: float,
    period_max: float,
    n_periods: int = 2000,
    n_durations: int = 20,
) -> dict[str, Any]:
    """
    Task 4 — BLS period search (Astropy BoxLeastSquares).
    TLS (e.g. transitleastsquares) can be plugged in similarly.
    """
    from astropy.timeseries import BoxLeastSquares

    y = np.asarray(flux, dtype=float)
    y = y - np.nanmean(y)
    t = np.asarray(time_days, dtype=float)
    bls = BoxLeastSquares(t, y)
    periods = np.linspace(period_min, period_max, n_periods)
    span = max(time_days[-1] - time_days[0], period_max * 2)
    # Keep lower bound small: long-period systems still have ~hours-long transits
    d_lo = 0.02
    d_hi = min(0.45 * period_max, 0.4 * span / max(len(t) / 200.0, 1.0))
    d_hi = float(np.clip(d_hi, d_lo + 0.01, 0.5))
    durations = np.linspace(d_lo, d_hi, n_durations)

    best_power = -np.inf
    best_p = float(period_min)
    best_dur = float(durations[len(durations) // 2])
    best_depth_snr = 0.0
    for d in durations:
        res = bls.power(periods, d)
        pw = np.asarray(res.power)
        j = int(np.argmax(pw))
        if pw[j] > best_power:
            best_power = float(pw[j])
            best_p = float(periods[j])
            best_dur = float(d)
            ds = np.asarray(res.depth_snr)
            best_depth_snr = float(ds[j]) if ds.size else 0.0

    return {
        "period_recovered_days": best_p,
        "bls_max_power": max(best_power, 0.0),
        "depth_snr": best_depth_snr,
        "duration_recovered_days": best_dur,
        "period_grid_days": periods,
    }


def refine_bls_harmonics(
    time_days: np.ndarray,
    flux: np.ndarray,
    coarse: dict[str, Any],
    period_min: float,
    period_max: float,
    max_num: int = 3,
) -> dict[str, Any]:
    """
    Re-evaluate BLS at integer-period ratio aliases (P, 2P, P/2, 2P/3, ...).
    Reduces common false positives where the global grid peaks on a harmonic.
    """
    from astropy.timeseries import BoxLeastSquares

    y = np.asarray(flux, dtype=float) - np.nanmean(flux)
    t = np.asarray(time_days, dtype=float)
    bls = BoxLeastSquares(t, y)
    p0 = float(coarse["period_recovered_days"])
    d0 = float(coarse["duration_recovered_days"])

    candidates: set[float] = {p0}
    for num in range(1, max_num + 1):
        for den in range(1, max_num + 1):
            pc = p0 * num / den
            if period_min <= pc <= period_max:
                candidates.add(float(pc))

    # Neighbors of rational aliases (true P rarely equals exactly n/m × wrong peak)
    polished: set[float] = set()
    for pc in candidates:
        for fac in np.linspace(0.98, 1.02, 11):
            p2 = pc * fac
            if period_min <= p2 <= period_max:
                polished.add(float(p2))
    candidates |= polished

    best_power = -np.inf
    best_p = p0
    best_dur = d0
    best_depth_snr = -np.inf

    span = max(time_days[-1] - time_days[0], period_max * 2)
    d_lo = 0.02
    d_hi = min(0.45 * period_max, 0.4 * span / max(len(t) / 200.0, 1.0))
    d_hi = float(np.clip(d_hi, d_lo + 0.01, 0.5))

    d_scan = np.linspace(d_lo, d_hi, 14)
    for pc in candidates:
        for dur_try in d_scan:
            dur_try = float(dur_try)
            res = bls.power(np.asarray([pc]), dur_try)
            pw = float(np.asarray(res.power)[0])
            ds = float(np.asarray(res.depth_snr)[0])
            # Among harmonics, depth_snr tracks in-transit coherence better than raw power.
            if ds > best_depth_snr or (np.isclose(ds, best_depth_snr) and pw > best_power):
                best_power = pw
                best_p = pc
                best_dur = dur_try
                best_depth_snr = ds

    out = dict(coarse)
    out["period_recovered_days"] = best_p
    out["duration_recovered_days"] = best_dur
    out["bls_max_power"] = max(best_power, 0.0)
    out["depth_snr"] = best_depth_snr
    return out


def run_bls_recovery_with_refinement(
    time_days: np.ndarray,
    flux: np.ndarray,
    period_min: float,
    period_max: float,
    n_periods: int = 2000,
    n_durations: int = 20,
    refine_harmonics: bool = True,
) -> dict[str, Any]:
    """BLS grid search + optional harmonic alias pick."""
    coarse = run_bls_recovery(
        time_days, flux, period_min, period_max, n_periods=n_periods, n_durations=n_durations
    )
    if not refine_harmonics:
        return coarse
    return refine_bls_harmonics(time_days, flux, coarse, period_min, period_max)


def phased_angles(time_days: np.ndarray, period_days: float, t0_days: float) -> np.ndarray:
    """Return phase in [0, 1) with mid-transit at phase 0."""
    return ((time_days - t0_days) / period_days + 0.5) % 1.0 - 0.5


def export_tables(
    injected: pd.DataFrame,
    results: pd.DataFrame,
    prefix: str = "plato_hz_project",
) -> tuple[str, str]:
    """Write Task deliverable CSV tables; returns (injected_path, results_path)."""
    pi = f"{prefix}_injected.csv"
    pr = f"{prefix}_results.csv"
    injected.to_csv(pi, index=False)
    results.to_csv(pr, index=False)
    return pi, pr


def synthetic_observation_for_system(
    sys: PlanetarySystem,
    baseline_days: float = 200.0,
    cadence_days: float = 0.05,
    rng: np.random.Generator | None = None,
    *,
    light_curve_backend: LightCurveBackend = "psls",
    psls_magnitude_bright: float = 11.0,
    psls_workdir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    One light curve + mid-exposure reference epoch (days) for folding.
    Uses PSLS by default; set `light_curve_backend='synthetic'` for box+noise only.
    """
    if rng is None:
        rng = np.random.default_rng()
    star = sys.star
    p = sys.planets[0]
    t0 = float(rng.uniform(0.0, 5.0))
    time = np.arange(0.0, baseline_days, cadence_days)
    seed = int(rng.integers(1 << 30))
    if light_curve_backend == "synthetic":
        noise = 80.0 * sys.noise_multiplier
        t, f = generate_synthetic_plato_like_lightcurve(
            time,
            p.period_days,
            t0,
            p.radius_earth,
            star.radius_solar,
            noise_ppm=noise,
            seed=seed,
        )
        return t, f, t0

    wd = psls_workdir
    if not wd:
        wd = tempfile.mkdtemp(prefix="psls_obs_") + os.sep
    mag = float(psls_magnitude_bright + 2.5 * np.log10(max(sys.noise_multiplier, 1e-3)))
    sid = _stable_star_id(sys.label, p.name, seed, seed)
    t, f = generate_psls_lightcurve(
        time,
        p.period_days,
        t0,
        p.radius_earth,
        star.radius_solar,
        star.teff_k,
        magnitude=mag,
        seed=seed,
        star_id=sid,
        workdir=wd,
    )
    return t, f, t0


# ---------------------------------------------------------------------------
# Sample of 5–10 systems (Task 2) — includes HZ, hot, cold, harder-noise case
# ---------------------------------------------------------------------------


def default_sample_systems() -> list[PlanetarySystem]:
    """Structured small sample as in project scope."""
    systems: list[PlanetarySystem] = []

    # Near HZ — promising
    systems.append(
        PlanetarySystem(
            label="HZ_earth_like",
            star=Star("HZ-A", radius_solar=1.0, teff_k=5780.0),
            planets=[
                Planet("b", radius_earth=1.1, period_days=289.0, semi_major_axis_au=1.02)
            ],
            noise_multiplier=1.0,
        )
    )

    # Hot inner control
    systems.append(
        PlanetarySystem(
            label="hot_jupiter_control",
            star=Star("Hot-A", radius_solar=1.05, teff_k=6000.0),
            planets=[
                Planet("b", radius_earth=11.0, period_days=3.5, semi_major_axis_au=0.045)
            ],
            noise_multiplier=1.0,
        )
    )

    # Cold outer control
    systems.append(
        PlanetarySystem(
            label="cold_subneptune",
            star=Star("Cold-A", radius_solar=0.85, teff_k=5200.0),
            planets=[
                Planet("b", radius_earth=3.2, period_days=420.0, semi_major_axis_au=1.15)
            ],
            noise_multiplier=1.0,
        )
    )

    # HZ super-Earth
    systems.append(
        PlanetarySystem(
            label="HZ_super_earth",
            star=Star("HZ-B", radius_solar=0.92, teff_k=5600.0),
            planets=[
                Planet("b", radius_earth=1.6, period_days=198.0, semi_major_axis_au=0.68)
            ],
            noise_multiplier=1.0,
        )
    )

    # Harder case: stronger noise
    systems.append(
        PlanetarySystem(
            label="HZ_noisy",
            star=Star("HZ-C", radius_solar=1.0, teff_k=5750.0),
            planets=[
                Planet("b", radius_earth=1.3, period_days=240.0, semi_major_axis_au=0.78)
            ],
            noise_multiplier=3.5,
        )
    )

    return systems


def analyze_systems(
    systems: list[PlanetarySystem] | None = None,
    cadence_days: float = 0.05,  # ~1.2 h; use ~25–30 min cadence to match PLATO
    baseline_days: float = 1000.0,
    base_noise_ppm: float = 80.0,
    seed: int = 42,
    *,
    light_curve_backend: LightCurveBackend = "psls",
    psls_magnitude_bright: float = 11.0,
    psls_workdir_root: str | None = None,
    cleanup_psls_root: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full loop: simulated LC (PSLS or simple synthetic) → BLS → physics → HZ flags → scores.
    Returns (injected_table, results_table).

    PSLS uses photon noise vs. magnitude; `noise_multiplier` maps to fainter magnitude.
    Optional PLATO systematic `.npy` tables are not bundled with pip; this path uses `gen_up` only.
    """
    if systems is None:
        systems = default_sample_systems()

    rng = np.random.default_rng(seed)
    injected_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    psls_root_created: str | None = None
    if light_curve_backend == "psls" and psls_workdir_root is None:
        psls_root_created = tempfile.mkdtemp(prefix="psls_hz_pipeline_")
        psls_workdir_root = psls_root_created

    try:
        for sys in systems:
            star = sys.star
            lum = star.luminosity_solar()
            for p in sys.planets:
                noise = base_noise_ppm * sys.noise_multiplier
                t0 = float(rng.uniform(0.0, 5.0))
                time = np.arange(0.0, baseline_days, cadence_days)
                lc_seed = int(rng.integers(1 << 30))

                if light_curve_backend == "synthetic":
                    time_d, flux = generate_synthetic_plato_like_lightcurve(
                        time,
                        p.period_days,
                        t0,
                        p.radius_earth,
                        star.radius_solar,
                        noise_ppm=noise,
                        seed=lc_seed,
                    )
                else:
                    assert psls_workdir_root is not None
                    sub = os.path.join(
                        psls_workdir_root,
                        f"{sys.label}_{p.name}_{lc_seed}".replace(" ", "_"),
                    )
                    os.makedirs(sub, exist_ok=True)
                    mag = float(
                        psls_magnitude_bright + 2.5 * np.log10(max(sys.noise_multiplier, 1e-3))
                    )
                    sid = _stable_star_id(sys.label, p.name, lc_seed, seed)
                    time_d, flux = generate_psls_lightcurve(
                        time,
                        p.period_days,
                        t0,
                        p.radius_earth,
                        star.radius_solar,
                        star.teff_k,
                        magnitude=mag,
                        seed=lc_seed,
                        star_id=sid,
                        workdir=sub + os.sep,
                    )

                pmin = max(0.4 * p.period_days, 0.2)
                # Need baseline ≳ 2P for BLS; cap search below ~0.48×baseline
                pmax = min(1.55 * p.period_days, 0.48 * baseline_days)
                rec = run_bls_recovery_with_refinement(time_d, flux, pmin, pmax)

                s_earth = incident_flux_earth_units(lum, p.semi_major_axis_au)
                teq = equilibrium_temperature_from_au(
                    star.teff_k, star.radius_solar, p.semi_major_axis_au, albedo=p.albedo
                )
                flags = habitable_zone_flags(s_earth, teq)
                snr_proxy = max(rec.get("depth_snr", 0.0), rec["bls_max_power"])
                scores = habitability_rank_score(
                    s_earth, p.radius_earth, star.teff_k, snr_proxy, eccentricity=None
                )

                injected_rows.append(
                    {
                        "system": sys.label,
                        "star": star.name,
                        "planet": p.name,
                        "P_injected_d": p.period_days,
                        "a_AU": p.semi_major_axis_au,
                        "Rp_Rearth": p.radius_earth,
                        "Teff_K": star.teff_k,
                        "Rstar_Rsun": star.radius_solar,
                        "noise_mult": sys.noise_multiplier,
                        "lc_backend": light_curve_backend,
                    }
                )

                result_rows.append(
                    {
                        "system": sys.label,
                        "star": star.name,
                        "planet": p.name,
                        "P_injected_d": p.period_days,
                        "P_recovered_d": rec["period_recovered_days"],
                        "L_Lsun": lum,
                        "S_earth": s_earth,
                        "Teq_K": teq,
                        **{k: v for k, v in flags.items() if k.startswith("in_") or k.startswith("near_")},
                        **scores,
                        "lc_backend": light_curve_backend,
                    }
                )

    finally:
        if cleanup_psls_root and psls_root_created and os.path.isdir(psls_root_created):
            shutil.rmtree(psls_root_created, ignore_errors=True)

    injected_df = pd.DataFrame(injected_rows)
    results_df = pd.DataFrame(result_rows)
    return injected_df, results_df


if __name__ == "__main__":
    inj, res = analyze_systems()
    print("=== Injected parameters ===")
    print(inj.to_string(index=False))
    print("\n=== Analysis ===")
    print(res.sort_values("rank_score", ascending=False).to_string(index=False))
