"""
test_engine.py - physics checks for the simulator.   Run:  python test_engine.py

Each check is a statement you can also verify by eye in the app, and one you can
defend to your professor: the loop shape must come out of the physics, not be put in.
"""
import numpy as np
import engine as eng

MU0 = eng.MU0
mT = lambda x: x * 1e-3 / MU0            # mT -> A/m


def boltzmann_powder_m(sample, H, n_u=64, n_th=2000):
    """Exact thermal-equilibrium M/Ms of a monodisperse random powder (no two-state approximation)."""
    V = np.pi / 6 * sample.D_mean ** 3
    c = 2 * sample.K * V / (eng.K_B * sample.T)
    x, w = np.polynomial.legendre.leggauss(n_u)
    u, wu = 0.5 * (x + 1), 0.5 * w
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    out = []
    for h in H / sample.Hk:
        tot = 0.0
        for ui, wi in zip(u, wu):
            ps = np.arccos(ui)
            e = 0.5 * np.sin(th) ** 2 - h * np.cos(th - ps)
            wgt = np.exp(-c * (e - e.min()))
            tot += wi * np.sum(np.cos(th - ps) * wgt) / np.sum(wgt)
        out.append(tot)
    return np.array(out)


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    return ok


def main():
    results = []
    mag = dict(Ms=4.8e5, K=2e4, rho=5170.0)

    # 1. Low field: area grows as H^2 (the loop is an ellipse in linear response)
    s = eng.Sample(D_mean=14e-9, sigma=0.2, **mag)
    a = [eng.run_ac(s, mT(b), 100e3).metrics["area"] for b in (0.5, 1.0, 2.0)]
    slope = np.log(a[2] / a[0]) / np.log(4)
    results.append(check("low field: loop area ~ H^2", abs(slope - 2) < 0.1, f"log-log slope = {slope:.2f}"))

    # 2. Low field: the loop is a tilted ellipse (M follows H with a phase lag): compare with best-fit ellipse
    r = eng.run_ac(s, mT(1.0), 100e3, cycles=12)
    H, M = r.H[r.loop], r.M[r.loop] / s.Ms
    # fit M = chi' H0 sin(wt) + chi'' H0 cos(wt) on the last cycle
    ph = np.linspace(0, 2 * np.pi, len(H))
    A = np.c_[np.sin(ph + np.pi / 2), np.cos(ph + np.pi / 2)]
    coef, *_ = np.linalg.lstsq(A, M, rcond=None)
    resid = np.max(np.abs(A @ coef - M)) / np.max(np.abs(M))
    results.append(check("low field: M(t) is a pure sinusoid (ellipse)", resid < 0.01, f"max deviation = {resid*100:.2f}% of peak"))

    # 3. Equilibrium limit equals exact Boltzmann statistics (superparamagnetic, very slow field)
    s2 = eng.Sample(D_mean=16e-9, sigma=0.0, **mag)
    xi = s2.K * np.pi / 6 * s2.D_mean ** 3 / (eng.K_B * s2.T)
    t, Hf, loop = eng.ac_field(mT(20), 10.0, cycles=2, steps_per_cycle=400)
    m = eng.simulate(s2, t, Hf)
    sel = np.arange(len(t))[loop][::25]
    exact = boltzmann_powder_m(s2, Hf[sel])
    err = np.max(np.abs(m[sel] - exact))
    results.append(check("slow field: matches exact Boltzmann equilibrium", err < 0.03,
                         f"KV/kT = {xi:.1f}, max |difference| = {err:.3f} of Ms"))

    # 4. Strongly blocked powder: random-axis Stoner-Wohlfarth values  (Mr/Ms ~ 0.5, Hc below 0.48 Hk)
    s3 = eng.Sample(D_mean=30e-9, sigma=0.0, Ms=4.8e5, K=1e5, rho=5170.0)
    r3 = eng.run_vsm(s3, 1.5 / MU0, loop_time=600.0)
    hc_over_hk = r3.metrics["Hc"] / s3.Hk
    results.append(check("blocked powder: Mr/Ms about 0.5", 0.45 < r3.metrics["mr"] < 0.52, f"Mr/Ms = {r3.metrics['mr']:.3f}"))
    results.append(check("blocked powder: 0.25 Hk < Hc < 0.48 Hk", 0.25 < hc_over_hk < 0.48, f"Hc/Hk = {hc_over_hk:.3f}"))

    # 5. Bounds and closure
    r5 = eng.run_ac(eng.Sample(D_mean=20e-9, sigma=0.3, **mag), mT(90), 100e3)
    mm = r5.M / 4.8e5
    results.append(check("|M| never exceeds Ms", np.max(np.abs(mm)) <= 1.0 + 1e-9, f"max |M|/Ms = {np.max(np.abs(mm)):.4f}"))
    Hl, Ml = r5.H[r5.loop], r5.M[r5.loop]
    results.append(check("steady-state loop is closed", abs(Ml[0] - Ml[-1]) / 4.8e5 < 0.01, f"gap = {abs(Ml[0]-Ml[-1])/4.8e5:.4f} Ms"))

    # 6. Trends: bigger particles / higher frequency / lower temperature widen the loop
    base = dict(sigma=0.2, **mag)
    A = lambda **k: eng.run_ac(eng.Sample(**{**dict(D_mean=14e-9, **base), **{kk: v for kk, v in k.items() if kk in ('D_mean', 'T')}}),
                               mT(40), k.get('f', 100e3)).metrics["area"]
    results.append(check("area increases with particle size", A(D_mean=10e-9) < A(D_mean=14e-9) < A(D_mean=18e-9)))
    results.append(check("area increases with frequency (in the relaxing regime)", A(f=20e3) < A(f=100e3) < A(f=500e3)))
    results.append(check("area increases when cooled", A(T=450) < A(T=300) < A(T=200)))

    # 7. Smooth, single-equation transition: ellipse -> hysteresis loop as amplitude grows
    s7 = eng.Sample(D_mean=22e-9, sigma=0.15, **mag)
    amps = [5, 10, 20, 40, 60, 90]
    met = [eng.run_ac(s7, mT(b), 100e3).metrics for b in amps]
    sat = [x["sat"] for x in met]
    results.append(check("saturation reached rises smoothly with field", all(np.diff(sat) > 0), "M_max/Ms = " + ", ".join(f"{v:.2f}" for v in sat)))
    results.append(check("area stops growing as H^2 at high field",
                         np.log(met[-1]["area"] / met[-2]["area"]) / np.log(90 / 60) < 1.5))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
