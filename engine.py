"""
engine.py  -  the physics behind the hysteresis simulator
==========================================================

What this file does
-------------------
It simulates how a POWDER of magnetic nanoparticles responds to a magnetic field
that changes in time, and returns the magnetisation loop M(H).

The same code is used for both instruments. Only the field programme differs:

    VSM                   slow, large sweep      (about 1 T, minutes per loop)
    AC hysteresis magn.   fast, smaller sinusoid (tens of mT, about 100 kHz)

The model (one particle)
------------------------
Each particle is a single magnetic domain with uniaxial anisotropy. Its energy is

    E(theta) = K V sin^2(theta)  -  mu0 Ms V H cos(theta - psi)

    theta : angle of the moment from the particle's easy axis
    psi   : angle between the easy axis and the applied field
    K     : effective anisotropy constant (J/m^3)
    V     : particle volume,  Ms : saturation magnetisation

For a field that is not too large this energy has TWO minima ("well A" with the
moment near +easy axis, "well B" near -easy axis) separated by an energy barrier.

  * Inside a well the moment tilts reversibly toward the field.
  * Between wells it jumps by thermal activation (Neel-Arrhenius), over either of the
    two saddle points (moment turning one way or the other):
        rate = f0/2 * [ exp(-barrier_1 / kB T) + exp(-barrier_2 / kB T) ]
    which equals the textbook  f0 exp(-K V / kB T)  at zero field.
  * The fraction p of particles sitting in well A obeys a rate equation
        dp/dt = -p * rate(A->B) + (1 - p) * rate(B->A)

The barriers, the well positions and the field at which a well disappears are
found numerically for every angle psi (no closed-form approximation), so the
low-field ellipse and the high-field Stoner-Wohlfarth-like loop come out of the
SAME equation. This follows the two-level approach of Carrey, Mehdaoui and
Respaud, J. Appl. Phys. 109, 083921 (2011).

From one particle to the whole sample
-------------------------------------
  * sizes:        log-normal distribution on a fine grid in ln D
  * orientations: easy axes random in 3D (uniform in cos psi), Gauss-Legendre
  * each particle counts in proportion to its VOLUME (that is what the
    magnetometer measures)

What is NOT included (see the "How it works" tab in the app)
------------------------------------------------------------
particle-particle interactions, self-heating during the run, Brownian rotation
(irrelevant for immobilised powders), cubic anisotropy (K is one effective
uniaxial number), surface spin disorder beyond a reduced Ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

K_B = 1.380649e-23          # Boltzmann constant, J/K
MU0 = 4e-7 * np.pi          # vacuum permeability, T m/A
_BIG = 1e4                  # "barrier" used for a well that no longer exists


# --------------------------------------------------------------------------
# Sample description
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    Ms: float               # saturation magnetisation, A/m
    K: float                # effective uniaxial anisotropy, J/m^3
    D_mean: float           # median particle diameter, m
    sigma: float            # width of the log-normal distribution (ln D)
    rho: float              # density of the magnetic material, kg/m^3
    T: float = 300.0        # temperature, K
    f0: float = 1e9         # attempt frequency, Hz

    @property
    def Hk(self) -> float:
        """Anisotropy field in A/m."""
        return 2.0 * self.K / (MU0 * self.Ms)


def _newton(th0, h, psi, dth, iters=4):
    """Polish a stationary point of e(theta) = 0.5 sin^2 - h cos(theta - psi); never moves more than one grid cell."""
    t = th0.copy()
    for _ in range(iters):
        g = np.sin(t) * np.cos(t) + h * np.sin(t - psi)
        gp = np.cos(2 * t) + h * np.cos(t - psi)
        step = np.where(np.abs(gp) > 1e-9, g / np.where(np.abs(gp) > 1e-9, gp, 1.0), 0.0)
        t = t - np.clip(step, -dth, dth)
    return t


# --------------------------------------------------------------------------
# Step 1: energy landscape of one particle (tabulated once, material-free)
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def build_table(n_angles: int = 16, n_theta: int = 1440) -> dict:
    """
    For every field angle psi and reduced field h = H/Hk, find numerically:
        dA, dB  : barrier out of well A / well B, in units of 2KV
        mA, mB  : moment component along the field in well A / well B (m = M/Ms)

    The energy in units of 2KV is   e(theta) = 0.5 sin^2(theta) - h cos(theta - psi),
    so the table does not depend on the material or on the particle size.
    """
    x, w = np.polynomial.legendre.leggauss(n_angles)
    u = 0.5 * (x + 1.0)                 # cos(psi) in (0, 1)
    wa = 0.5 * w                        # weights, sum to 1
    psi = np.arccos(u)

    pos = np.concatenate([np.linspace(0.0, 1.5, 376),
                          np.geomspace(1.5, 200.0, 120)[1:]])
    hg = np.concatenate([-pos[:0:-1], pos])          # ascending, symmetric

    N = n_theta
    half = N // 2
    dth = 2 * np.pi / N
    th = -np.pi / 2 + dth * np.arange(N)             # well A: first half, well B: second half
    idx = np.arange(N)[None, :]

    dA = np.empty((n_angles, 2, len(hg)))     # barrier out of well A over each of the two saddles
    dB = np.empty_like(dA)
    mA = np.empty((n_angles, len(hg)))
    mB = np.empty_like(mA)

    for j, ps in enumerate(psi):
        e = 0.5 * np.sin(th)[None, :] ** 2 - hg[:, None] * np.cos(th[None, :] - ps)
        # true local minima of the energy (circular), one per half-circle at most
        is_min = (e < np.roll(e, 1, axis=1)) & (e <= np.roll(e, -1, axis=1))
        candA = is_min & (idx < half)
        candB = is_min & (idx >= half)
        iA = np.where(candA, e, np.inf).argmin(axis=1)
        iB = np.where(candB, e, np.inf).argmin(axis=1)
        van_A = ~candA.any(axis=1)                   # no minimum near +easy axis: well A is gone
        van_B = ~candB.any(axis=1)
        iA = np.where(van_A, 0, iA)                  # placeholders; values replaced below
        iB = np.where(van_B, N - 1, iB)

        arc1 = (idx > iA[:, None]) & (idx < iB[:, None])
        arc2 = (idx < iA[:, None]) | (idx > iB[:, None])
        e1 = np.where(arc1, e, -np.inf)
        e2 = np.where(arc2, e, -np.inf)
        iS1, iS2 = e1.argmax(axis=1), e2.argmax(axis=1)      # the two saddles = the two ways round

        # refine the stationary points below the grid spacing (Newton on de/dtheta = 0);
        # without this the small reversible tilt of the moment at low field is lost
        thA = _newton(th[iA], hg, ps, dth)
        thB = _newton(th[iB], hg, ps, dth)
        thS1 = _newton(th[iS1], hg, ps, dth)
        thS2 = _newton(th[iS2], hg, ps, dth)
        e_at = lambda t_: 0.5 * np.sin(t_) ** 2 - hg * np.cos(t_ - ps)
        eA, eB = e_at(thA), e_at(thB)
        bA1 = np.clip(e_at(thS1) - eA, 0.0, None)
        bA2 = np.clip(e_at(thS2) - eA, 0.0, None)
        bB1 = np.clip(e_at(thS1) - eB, 0.0, None)
        bB2 = np.clip(e_at(thS2) - eB, 0.0, None)

        a_m = np.cos(thA - ps)
        b_m = np.cos(thB - ps)

        # a vanished well: its barriers are zero and the other well can no longer be left
        bA1, bA2 = [np.where(van_A, 0.0, np.where(van_B, _BIG, x)) for x in (bA1, bA2)]
        bB1, bB2 = [np.where(van_B, 0.0, np.where(van_A, _BIG, x)) for x in (bB1, bB2)]
        a_m = np.where(van_A, b_m, a_m)
        b_m = np.where(van_B, a_m, b_m)

        dA[j, 0], dA[j, 1], dB[j, 0], dB[j, 1] = bA1, bA2, bB1, bB2
        mA[j], mB[j] = a_m, b_m

    return dict(h=hg, wa=wa, psi=psi, dA=dA, dB=dB, mA=mA, mB=mB)


# --------------------------------------------------------------------------
# Step 2: size distribution
# --------------------------------------------------------------------------
def size_nodes(D_mean: float, sigma: float, n: int = 61):
    """
    Diameters and VOLUME weights (sum to 1) for a log-normal distribution of median D_mean.

    A fine uniform grid in ln D is used on purpose: the heating at low field comes from a
    narrow band of sizes whose Neel time is close to the field period (tau grows like
    exp(K V / kT)), and a coarse quadrature misses that band.
    """
    if sigma < 1e-3 or n < 2:
        return np.array([D_mean]), np.array([1.0])
    # the volume-weighted distribution is a Gaussian in ln D centred at ln D0 + 3 sigma^2
    x = np.linspace(3 * sigma - 4.5, 3 * sigma + 4.5, n)         # ln D = ln D0 + sigma x
    D = D_mean * np.exp(sigma * x)
    wv = np.exp(-0.5 * x ** 2) * D ** 3                           # number density x volume
    return D, wv / wv.sum()


# --------------------------------------------------------------------------
# Step 3: time evolution of the whole sample
# --------------------------------------------------------------------------
def simulate(sample: Sample, t: np.ndarray, H: np.ndarray,
             n_sizes: int = 61, n_angles: int = 16, p_init: float = 0.5):
    """
    Returns M(t)/Ms of the whole powder for the field history H(t) (A/m).
    The sample starts demagnetised (half the particles in each well).
    """
    tab = build_table(n_angles)
    hg = tab["h"]
    na = len(tab["wa"])

    h = H / sample.Hk
    h_mid = 0.5 * (h[1:] + h[:-1])
    dt = np.diff(t)
    nt = len(t)

    D, wv = size_nodes(sample.D_mean, sample.sigma, n_sizes)
    V = math.pi / 6.0 * D ** 3
    c = (2.0 * sample.K * V / (K_B * sample.T))[:, None]                      # (ns, 1): barrier scale / kT

    m = np.zeros(nt)
    for j in range(na):
        mA = np.interp(h, hg, tab["mA"][j])                                   # (nt,)
        mB = np.interp(h, hg, tab["mB"][j])

        def rate(key):      # escape over both saddles; /2 so that H=0 gives exactly f0 exp(-KV/kT)
            r = 0.0
            for k in (0, 1):
                d = np.interp(h_mid, hg, tab[key][j, k])[None, :]             # (1, nt-1)
                r = r + np.exp(-np.clip(c * d, 0.0, 700.0))
            return 0.5 * sample.f0 * r                                        # (ns, nt-1)

        nuAB, nuBA = rate("dA"), rate("dB")
        lam = nuAB + nuBA
        peq = np.where(lam > 0, nuBA / np.where(lam > 0, lam, 1.0), 0.5)
        dec = np.exp(-lam * dt[None, :])
        peq, dec = np.ascontiguousarray(peq.T), np.ascontiguousarray(dec.T)   # (nt-1, ns)

        p = np.empty((nt, len(D)))
        p[0] = p_init
        for n in range(nt - 1):
            # exact solution of the two-state rate equation over one step (stable for any rate)
            p[n + 1] = peq[n] + (p[n] - peq[n]) * dec[n]

        pm = p @ wv                                                           # volume-weighted fraction in well A
        m += tab["wa"][j] * (pm * mA + (1.0 - pm) * mB)
    return m


# --------------------------------------------------------------------------
# Field programmes
# --------------------------------------------------------------------------
def ac_field(H_amp: float, freq: float, cycles: int = 3, steps_per_cycle: int = 400):
    """Sinusoid. The window returned for the loop is the last full cycle, starting at the positive peak."""
    n = int((cycles + 0.25) * steps_per_cycle)
    t = np.arange(n + 1) / (steps_per_cycle * freq)
    H = H_amp * np.sin(2 * np.pi * freq * t)
    loop = slice(n + 1 - (steps_per_cycle + 1), n + 1)
    return t, H, loop


def vsm_field(H_max: float, loop_time: float, steps: int = 3000):
    """0 -> +Hmax -> -Hmax -> +Hmax at constant sweep rate. The loop is the last two legs."""
    rate = 4.0 * H_max / loop_time
    knots = np.array([0.0, H_max, -H_max, H_max])
    seg = np.abs(np.diff(knots))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], steps + 1)
    H = np.interp(s, cum, knots)
    t = s / rate
    start = int(np.searchsorted(s, H_max))
    return t, H, slice(start, steps + 1)


# --------------------------------------------------------------------------
# Quantities read off a loop (the same ones you read off a measurement)
# --------------------------------------------------------------------------
def _zero_crossings(x, y):
    """x values where y changes sign (linear interpolation)."""
    s = np.sign(y)
    i = np.where(s[:-1] * s[1:] < 0)[0]
    if len(i) == 0:
        return np.array([])
    return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def loop_metrics(H, m, Ms):
    """Coercivity, remanence, saturation reached and loop area for a closed loop."""
    Hc = _zero_crossings(H, m)
    # m at H = 0 : interpolate the other way round
    s = np.sign(H)
    i = np.where(s[:-1] * s[1:] < 0)[0]
    mr_vals = m[i] - H[i] * (m[i + 1] - m[i]) / (H[i + 1] - H[i]) if len(i) else np.array([])
    area = abs(MU0 * np.sum(0.5 * (m[1:] + m[:-1]) * Ms * np.diff(H)))      # J/m^3 per loop
    return dict(
        Hc=float(np.mean(np.abs(Hc))) if len(Hc) else 0.0,                  # A/m
        mr=float(np.mean(np.abs(mr_vals))) if len(mr_vals) else 0.0,        # M_r / Ms
        sat=float(np.max(np.abs(m))),                                       # M_max / Ms
        area=float(area),                                                   # J/m^3 per cycle
    )


@dataclass
class Run:
    mode: str
    t: np.ndarray
    H: np.ndarray            # A/m
    M: np.ndarray            # A/m
    loop: slice
    metrics: dict
    freq: float | None = None


def run_ac(sample: Sample, H_amp: float, freq: float, cycles: int = 3,
           steps_per_cycle: int = 400, n_sizes: int = 61, n_angles: int = 16) -> Run:
    t, H, loop = ac_field(H_amp, freq, cycles, steps_per_cycle)
    m = simulate(sample, t, H, n_sizes, n_angles)
    met = loop_metrics(H[loop], m[loop], sample.Ms)
    met["sar"] = met["area"] * freq / sample.rho / 1e3                     # W per gram of magnetic material
    return Run("ac", t, H, m * sample.Ms, loop, met, freq)


def run_vsm(sample: Sample, H_max: float, loop_time: float = 600.0,
            steps: int = 3000, n_sizes: int = 61, n_angles: int = 16) -> Run:
    t, H, loop = vsm_field(H_max, loop_time, steps)
    m = simulate(sample, t, H, n_sizes, n_angles)
    met = loop_metrics(H[loop], m[loop], sample.Ms)
    return Run("vsm", t, H, m * sample.Ms, loop, met, None)


# --------------------------------------------------------------------------
# Which particles can follow the field?  (blocked vs superparamagnetic)
# --------------------------------------------------------------------------
def regime(sample: Sample, t_meas: float) -> dict:
    """
    t_meas : time the field takes to change appreciably (1/f for AC; time to sweep Hk for VSM).
    A particle is 'blocked' if its Neel time exp(KV/kT)/f0 is longer than t_meas.
    """
    V0 = math.pi / 6.0 * sample.D_mean ** 3
    xi0 = sample.K * V0 / (K_B * sample.T)
    xi_B = math.log(max(sample.f0 * t_meas, 1.0001))
    D_B = (6.0 * xi_B * K_B * sample.T / (math.pi * sample.K)) ** (1.0 / 3.0)
    if sample.sigma < 1e-3:
        blocked = 1.0 if sample.D_mean > D_B else 0.0
    else:   # volume-weighted log-normal: ln D ~ N(ln D0 + 3 sigma^2, sigma^2)
        z = (math.log(D_B) - math.log(sample.D_mean) - 3 * sample.sigma ** 2) / (math.sqrt(2) * sample.sigma)
        blocked = 0.5 * math.erfc(z)
    return dict(xi=xi0, tau=math.exp(min(xi0, 700.0)) / sample.f0, D_block=D_B, blocked_fraction=blocked)
