"""
app.py  -  the interface (no physics in here)
=============================================
Pattern: read the widgets -> build a Sample -> call engine.run_ac / engine.run_vsm -> plot.
All physics lives in engine.py.

Run locally:   streamlit run app.py
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import engine as eng

MU0 = eng.MU0
INK, TEAL, SLATE, AMBER = "#1F2A44", "#0B7A75", "#6B7A90", "#B7791F"

st.set_page_config(page_title="Nanoparticle hysteresis simulator", page_icon="🧲", layout="wide")
st.markdown("<style>.block-container{padding-top:2.2rem;max-width:1250px}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ database
DB = json.loads((Path(__file__).with_name("materials.json")).read_text(encoding="utf-8"))
CUSTOM = dict(name="Custom material", Ms_kA_m=400.0, K_kJ_m3=20.0, rho_kg_m3=5000.0,
              note="Enter your own values. Ms from your VSM plateau; K is the effective uniaxial anisotropy.")

# ------------------------------------------------------------------ units
FIELD_UNITS = {"mT": 1e3, "Oe": 1e4}                       # mu0*H in tesla x factor
MOMENT_UNITS = ["M / Ms", "emu/g", "emu (whole sample)"]


def field_to_display(H_Apm, unit):
    return np.asarray(H_Apm) * MU0 * FIELD_UNITS[unit]


def moment_to_display(M_Apm, unit, p):
    M = np.asarray(M_Apm)
    if unit == "M / Ms":
        return M / p["Ms"]
    if unit == "emu/g":
        return M / p["rho"]                                  # A/m / (kg/m3) = A m2/kg = emu/g
    return M / p["rho"] * p["mag_mass_g"]


def moment_label(unit):
    return {"M / Ms": "Magnetisation M / Ms", "emu/g": "Magnetisation (emu/g of magnetic phase)",
            "emu (whole sample)": "Moment of the whole sample (emu)"}[unit]


def fmt(x, digits=3):
    return f"{x:.{digits}g}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Inputs")

    st.subheader("Material")
    names = [m["name"] for m in DB] + [CUSTOM["name"]]
    choice = st.selectbox("Material", names, help="Fills in typical values. Overwrite them with your own measurements.")
    base = next((m for m in DB if m["name"] == choice), CUSTOM)
    Ms_k = st.number_input("Saturation magnetisation Ms (kA/m)", 10.0, 2000.0, float(base["Ms_kA_m"]), 5.0,
                           key=f"Ms_{choice}", help="Read it from the high-field plateau of your VSM loop, per volume of magnetic material.")
    K_k = st.number_input("Effective anisotropy K (kJ/m³)", 0.1, 1000.0, float(base["K_kJ_m3"]), 1.0,
                          key=f"K_{choice}", help="One effective uniaxial number. It also absorbs shape, surface and interaction effects.")
    rho = st.number_input("Density (kg/m³)", 1000.0, 20000.0, float(base["rho_kg_m3"]), 10.0, key=f"rho_{choice}")
    st.caption(f"Ms = {Ms_k * 1e3 / rho:.0f} emu/g.  {base['note']}")

    st.subheader("Particles")
    D_nm = st.slider("Median diameter (nm)", 3.0, 45.0, 12.0, 0.5, help="From TEM.")
    sigma = st.slider("Size spread σ (log-normal)", 0.0, 0.6, 0.20, 0.01,
                      help="Width of ln(D). About 0.1 is very narrow, 0.3 is broad. 0 = every particle the same size.")

    st.subheader("Sample")
    mass_mg = st.number_input("Powder mass (mg)", 0.01, 1000.0, 1.0, 0.1)
    frac = st.number_input("Magnetic fraction (%)", 1.0, 100.0, 100.0, 1.0,
                           help="Less than 100 if the particles sit on a support such as CNC/CNF or in sugar.")
    T = st.number_input("Temperature (K)", 4.0, 700.0, 300.0, 5.0)

    st.subheader("Measurement")
    mode_label = st.radio("Instrument", ["AC magnetometer", "VSM"], horizontal=True)
    mode = "ac" if mode_label == "AC magnetometer" else "vsm"
    if mode == "ac":
        amp_mT = st.slider("Field amplitude μ₀H (mT)", 1.0, 150.0, 30.0, 1.0)
        freq_kHz = st.number_input("Frequency (kHz)", 0.1, 2000.0, 100.0, 10.0)
        hmax_T, loop_time = 1.5, 600.0
    else:
        hmax_T = st.slider("Maximum field (T)", 0.1, 5.0, 1.5, 0.1)
        loop_time = st.number_input("Time for one full loop (s)", 30.0, 7200.0, 600.0, 30.0,
                                    help="Slower sweeps give particles more time to relax.")
        amp_mT, freq_kHz = 30.0, 100.0

    with st.expander("Advanced"):
        f0 = st.select_slider("Attempt frequency f₀ (Hz)", options=[1e8, 1e9, 1e10, 1e11, 1e12], value=1e9,
                              format_func=lambda x: f"{x:.0e}",
                              help="Rate prefactor in the Néel law. 10⁹ Hz is the usual choice; literature values range from 10⁹ to 10¹².")
        cycles = st.slider("Simulated cycles (AC)", 2, 20, 4, help="The last cycle is shown. More cycles remove the start-up transient.")

    st.subheader("Display")
    f_unit = st.radio("Field unit", list(FIELD_UNITS), horizontal=True)
    m_unit = st.radio("Magnetisation unit", MOMENT_UNITS)

P = dict(Ms=Ms_k * 1e3, K=K_k * 1e3, rho=rho, D=D_nm * 1e-9, sigma=sigma, T=T, f0=f0,
         mag_mass_g=mass_mg * 1e-3 * frac / 100.0)
sample = eng.Sample(Ms=P["Ms"], K=P["K"], D_mean=P["D"], sigma=sigma, rho=rho, T=T, f0=f0)


# ------------------------------------------------------------------ cached simulations
@st.cache_data(show_spinner=False, max_entries=64)
def cached_run(mode, Ms, K, D, sigma, rho, T, f0, H_Apm, freq, loop_time, cycles):
    s = eng.Sample(Ms=Ms, K=K, D_mean=D, sigma=sigma, rho=rho, T=T, f0=f0)
    if mode == "ac":
        return eng.run_ac(s, H_Apm, freq, cycles=cycles)
    return eng.run_vsm(s, H_Apm, loop_time)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_series(Ms, K, D, sigma, rho, T, f0, amps_Apm, freq, cycles):
    s = eng.Sample(Ms=Ms, K=K, D_mean=D, sigma=sigma, rho=rho, T=T, f0=f0)
    out = []
    for a in amps_Apm:
        r = eng.run_ac(s, a, freq, cycles=cycles)
        out.append((r.H[r.loop], r.M[r.loop], r.metrics))
    return out


# ------------------------------------------------------------------ header
st.title("Nanoparticle hysteresis simulator")
st.write("Describe the powder and the field. The model draws the magnetisation loop that an AC magnetometer or a VSM "
         "would record, from one set of equations, so the same sample parameters give the low-field ellipse and the "
         "high-field hysteresis loop.")

tab_loop, tab_series, tab_how = st.tabs(["Loop", "Several fields", "How it works"])


# ================================================================== TAB 1
def parse_upload(file):
    raw = file.getvalue().decode("utf-8", errors="ignore")
    if "[Data]" in raw:                                        # e.g. Quantum Design exports
        raw = raw.split("[Data]", 1)[1]
    df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", comment="#", skip_blank_lines=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    if df.shape[1] < 2 or len(df) < 5:
        raise ValueError("Found fewer than two numeric columns or too few rows.")
    return df


with tab_loop:
    H_Apm = amp_mT * 1e-3 / MU0 if mode == "ac" else hmax_T / MU0
    with st.spinner("Simulating…"):
        run = cached_run(mode, P["Ms"], P["K"], P["D"], sigma, rho, T, f0, H_Apm, freq_kHz * 1e3, loop_time, cycles)
    met = run.metrics
    reg = eng.regime(sample, 1.0 / (freq_kHz * 1e3) if mode == "ac"
                     else sample.Hk * loop_time / (4.0 * H_Apm))
    t_txt = f"{1e3 / freq_kHz:.3g} µs" if mode == "ac" else f"{sample.Hk * loop_time / (4.0 * H_Apm):.3g} s"

    # ---- numbers
    cols = st.columns(5 if mode == "ac" else 4)
    cols[0].metric("Peak magnetisation", f"{met['sat'] * 100:.0f}% of Ms", help="Largest M reached during the loop, relative to Ms.")
    cols[1].metric("Coercivity Hc", f"{fmt(float(field_to_display(met['Hc'], f_unit)))} {f_unit}",
                   help="Field at which M crosses zero.")
    mr_disp = float(moment_to_display(met["mr"] * P["Ms"], m_unit, P))
    cols[2].metric("Remanence Mr", f"{fmt(mr_disp)} {'Ms' if m_unit == 'M / Ms' else m_unit.split(' ')[0]}",
                   help="M at zero field.")
    cols[3].metric("Energy per cycle", f"{fmt(met['area'] / 1e3)} kJ/m³", help="Loop area, μ₀∮M dH, per volume of magnetic material.")
    if mode == "ac":
        cols[4].metric("Heating power (SAR)", f"{fmt(met['sar'])} W/g", help="Loop area × frequency ÷ density, per gram of magnetic material.")

    # ---- plot
    fig = go.Figure()
    full = slice(None)
    if mode == "vsm":
        fig.add_trace(go.Scatter(x=field_to_display(run.H[:run.loop.start + 1], f_unit),
                                 y=moment_to_display(run.M[:run.loop.start + 1], m_unit, P),
                                 mode="lines", line=dict(color=SLATE, width=1.5, dash="dot"), name="First magnetisation"))
    elif st.session_state.get("show_all", False) and cycles > 1:
        fig.add_trace(go.Scatter(x=field_to_display(run.H, f_unit), y=moment_to_display(run.M, m_unit, P),
                                 mode="lines", line=dict(color=SLATE, width=1, dash="dot"), name="All cycles"))
    Hl, Ml = run.H[run.loop], run.M[run.loop]
    fig.add_trace(go.Scatter(x=field_to_display(Hl, f_unit), y=moment_to_display(Ml, m_unit, P), mode="lines",
                             line=dict(color=TEAL, width=3.5), name="Simulated"))

    # coercivity / remanence markers
    hc_d = float(field_to_display(met["Hc"], f_unit))
    fig.add_trace(go.Scatter(x=[hc_d, -hc_d], y=[0, 0], mode="markers", marker=dict(color=INK, size=9, symbol="diamond"),
                             name="±Hc", hovertemplate=f"Hc = {fmt(hc_d)} {f_unit}<extra></extra>"))
    mr_d = float(moment_to_display(met["mr"] * P["Ms"], m_unit, P))
    fig.add_trace(go.Scatter(x=[0, 0], y=[mr_d, -mr_d], mode="markers", marker=dict(color=AMBER, size=9, symbol="square"),
                             name="±Mr", hovertemplate=f"Mr = {fmt(mr_d)}<extra></extra>"))

    # ---- optional measured overlay
    with st.expander("Overlay measured data (optional)"):
        st.caption("Upload a background-corrected loop as CSV or text, pick the columns and units, and it is drawn over the simulation.")
        up = st.file_uploader("Measured loop", type=["csv", "txt", "dat"], label_visibility="collapsed")
        meas = None
        if up is not None:
            try:
                df = parse_upload(up)
                c1, c2, c3, c4 = st.columns(4)
                hcol = c1.selectbox("Field column", df.columns, index=0)
                mcol = c2.selectbox("Moment column", df.columns, index=1)
                hu = c3.selectbox("Field unit", ["Oe", "mT", "T", "kA/m"])
                mu = c4.selectbox("Moment unit", ["emu", "emu/g (magnetic phase)", "A/m", "M / Ms (normalised)"])
                H_T = df[hcol].to_numpy() * {"Oe": 1e-4, "mT": 1e-3, "T": 1.0, "kA/m": 1e3 * MU0}[hu]
                Mraw = df[mcol].to_numpy()
                M_SI = {"emu": Mraw / P["mag_mass_g"] * rho, "emu/g (magnetic phase)": Mraw * rho,
                        "A/m": Mraw, "M / Ms (normalised)": Mraw * P["Ms"]}[mu]
                meas = (H_T / MU0, M_SI)
                if mu == "emu":
                    st.caption(f"Using a magnetic mass of {P['mag_mass_g'] * 1e3:.3g} mg (powder mass × magnetic fraction) to convert emu to emu/g.")
            except Exception as err:                                             # noqa: BLE001
                st.error(f"Could not read that file: {err}")
    if meas is not None:
        fig.add_trace(go.Scatter(x=field_to_display(meas[0], f_unit), y=moment_to_display(meas[1], m_unit, P), mode="markers",
                                 marker=dict(color="rgba(0,0,0,0)", size=6, line=dict(color=INK, width=1.2)), name="Measured"))

    fig.add_hline(y=0, line=dict(color="#C9D2DA", width=1))
    fig.add_vline(x=0, line=dict(color="#C9D2DA", width=1))
    fig.update_layout(template="plotly_white", height=540, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=f"Applied field μ₀H ({f_unit})", yaxis_title=moment_label(m_unit),
                      legend=dict(orientation="h", y=1.07, x=0), hovermode="closest",
                      font=dict(color=INK, size=14))
    st.plotly_chart(fig, width="stretch")
    if mode == "ac":
        st.checkbox("Also show the start-up cycles", key="show_all")

    # ---- plain-language reading of the result
    Hk_mT = sample.Hk * MU0 * 1e3
    if mode == "ac":
        ratio = amp_mT / Hk_mT
        shape = ("low-field regime: expect a tilted ellipse" if ratio < 0.25 else
                 "intermediate regime: the ellipse starts to distort" if ratio < 0.6 else
                 "high-field regime: expect a full hysteresis loop with a clear coercive field")
        st.markdown(f"**Reading the loop.** The field amplitude is {ratio * 100:.0f}% of the anisotropy field "
                    f"μ₀H<sub>k</sub> = {Hk_mT:.0f} mT, which is the {shape}.", unsafe_allow_html=True)
    else:
        st.markdown(f"**Reading the loop.** The anisotropy field is μ₀H<sub>k</sub> = {Hk_mT:.0f} mT; the sweep goes well past it, "
                    f"so the loop shows whether the sample reaches saturation.", unsafe_allow_html=True)
    st.markdown(
        f"**Which particles can follow the field?** The field changes on a time scale of about {t_txt}. Particles larger than "
        f"≈{reg['D_block'] * 1e9:.0f} nm have a Néel relaxation time longer than that, so they are blocked and make the loop open. "
        f"For your size distribution, **{reg['blocked_fraction'] * 100:.0f}%** of the magnetic volume is above that size "
        f"(median particle: KV/k<sub>B</sub>T = {reg['xi']:.1f}).", unsafe_allow_html=True)
    if reg["xi"] < 3:
        st.warning("The median particle has a very small energy barrier (KV/k_BT < 3). The two-level model underestimates the "
                   "low-field response there, so treat the result as qualitative.")
    if sample.Hk * MU0 * 1e3 < 5 and mode == "ac":
        st.info("The anisotropy field is only a few mT, so even small AC fields saturate the sample.")

    loop_df = pd.DataFrame({"mu0H_mT": run.H[run.loop] * MU0 * 1e3, "H_Oe": run.H[run.loop] * MU0 * 1e4,
                            "M_A_per_m": run.M[run.loop], "M_over_Ms": run.M[run.loop] / P["Ms"],
                            "M_emu_per_g": run.M[run.loop] / rho,
                            "moment_emu": run.M[run.loop] / rho * P["mag_mass_g"]})
    st.download_button("Download simulated loop (CSV)", loop_df.to_csv(index=False), "simulated_loop.csv", "text/csv")


# ================================================================== TAB 2
with tab_series:
    if mode != "ac":
        st.info("This view needs the AC magnetometer. Switch the instrument in the sidebar.")
    else:
        st.write("The same sample at several field amplitudes. At small amplitudes the loop is an ellipse, and as the amplitude "
                 "grows it opens into a full hysteresis loop.")
        c1, c2, c3 = st.columns(3)
        a_min = c1.number_input("Smallest amplitude (mT)", 0.5, 100.0, 5.0, 1.0)
        a_max = c2.number_input("Largest amplitude (mT)", 2.0, 200.0, 90.0, 5.0)
        n_curves = c3.slider("Number of loops", 3, 8, 6)
        amps_mT = np.unique(np.round(np.geomspace(a_min, max(a_max, a_min * 1.5), n_curves), 2))
        with st.spinner("Simulating several fields…"):
            res = cached_series(P["Ms"], P["K"], P["D"], sigma, rho, T, f0, tuple(float(a * 1e-3 / MU0) for a in amps_mT),
                                freq_kHz * 1e3, cycles)
        colors = sample_colorscale("Viridis", list(np.linspace(0.05, 0.85, len(amps_mT))))

        fig2 = go.Figure()
        for a, (Hs, Ms_, _), col in zip(amps_mT, res, colors):
            fig2.add_trace(go.Scatter(x=field_to_display(Hs, f_unit), y=moment_to_display(Ms_, m_unit, P), mode="lines",
                                      line=dict(color=col, width=2.5), name=f"{a:g} mT"))
        fig2.add_hline(y=0, line=dict(color="#C9D2DA", width=1))
        fig2.add_vline(x=0, line=dict(color="#C9D2DA", width=1))
        fig2.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title=f"Applied field μ₀H ({f_unit})", yaxis_title=moment_label(m_unit),
                           legend_title="Amplitude", font=dict(color=INK, size=14))
        st.plotly_chart(fig2, width="stretch")

        area = np.array([r[2]["area"] for r in res])
        hc = np.array([float(field_to_display(r[2]["Hc"], f_unit)) for r in res])
        mr = np.array([r[2]["mr"] for r in res])
        sat = np.array([r[2]["sat"] for r in res])

        sub = make_subplots(rows=1, cols=3, subplot_titles=("Energy per cycle (J/m³)", f"Coercivity Hc ({f_unit})",
                                                            "Mr and peak M (fraction of Ms)"))
        sub.add_trace(go.Scatter(x=amps_mT, y=area, mode="lines+markers", line=dict(color=TEAL, width=3), name="Loop area"), 1, 1)
        sub.add_trace(go.Scatter(x=amps_mT, y=area[0] * (amps_mT / amps_mT[0]) ** 2, mode="lines",
                                 line=dict(color=SLATE, dash="dash"), name="∝ H² guide"), 1, 1)
        sub.add_trace(go.Scatter(x=amps_mT, y=hc, mode="lines+markers", line=dict(color=INK, width=3), name="Hc", showlegend=False), 1, 2)
        sub.add_trace(go.Scatter(x=amps_mT, y=sat, mode="lines+markers", line=dict(color=TEAL, width=3), name="M_max / Ms"), 1, 3)
        sub.add_trace(go.Scatter(x=amps_mT, y=mr, mode="lines+markers", line=dict(color=AMBER, width=3), name="Mr / Ms"), 1, 3)
        sub.update_xaxes(title_text="Amplitude (mT)", type="log")
        sub.update_yaxes(type="log", row=1, col=1)
        sub.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.35), font=dict(color=INK, size=13))
        st.plotly_chart(sub, width="stretch")

        if len(amps_mT) >= 2 and area[0] > 0 and area[1] > 0:
            slope = np.log(area[1] / area[0]) / np.log(amps_mT[1] / amps_mT[0])
            st.caption(f"Between {amps_mT[0]:g} and {amps_mT[1]:g} mT the loop area grows as H^{slope:.2f}. "
                       "In the linear (ellipse) regime it should be close to H², and the curve should bend away from the dashed "
                       "guide as the loop opens.")


# ================================================================== TAB 3
with tab_how:
    st.markdown("""
### What goes in, what comes out

| You provide | The model returns |
|---|---|
| **Material:** Ms and K (the only material-specific numbers) | The loop **M(H)**: ellipse at low field, hysteresis loop at high field |
| **Particles:** median size and size spread (from TEM) | **Hc**, **Mr** and how close the sample gets to **saturation** |
| **Sample:** mass, magnetic fraction, temperature | **Energy per cycle** and **heating power (SAR)** |
| **Field:** AC amplitude and frequency, or a VSM sweep | The same loop in **emu/g** or **emu**, to compare with your instrument |

Ms is an **input**: it is a property of the material, and you read it from the VSM plateau. Hc, Mr and the loop shape are **outputs**, because they come from the competition between the field, the energy barrier and the thermal energy. K is rarely known for nanoparticles, so a practical use is to adjust K until one measured AC loop is matched, then check that the other amplitudes follow.

### One particle
Each particle is a single magnetic domain. Its energy depends on the angle θ of its moment from the easy axis and on the field angle ψ:
""")
    st.latex(r"E(\theta)=K V\sin^2\theta-\mu_0 M_s V H\cos(\theta-\psi)")
    st.markdown("""
For moderate fields this has two minima (moment along +axis or −axis) separated by a barrier. Inside a minimum the moment tilts reversibly toward the field. Between the minima it hops by thermal activation:
""")
    st.latex(r"\text{rate}=f_0\,e^{-\Delta E/k_BT},\qquad \frac{dp}{dt}=-p\,\nu_{A\to B}+(1-p)\,\nu_{B\to A}")
    st.markdown("""
The barriers, the positions of the minima and the field at which a minimum disappears are computed numerically for every angle, so one equation produces both the ellipse and the hysteresis loop.

### The whole sample
The program repeats this for 61 particle sizes (log-normal) and 16 easy-axis orientations (random in 3D), and adds them up weighted by volume, which is what a magnetometer measures. The result is multiplied by the magnetic mass to give emu.

### The two instruments
They differ only in the field programme. A **VSM** sweeps slowly to about 1 T, so most particles relax in time and the loop is nearly closed unless particles are large. An **AC magnetometer** oscillates at about 100 kHz, so particles whose relaxation time is longer than the period cannot follow, and the loop opens.

### What is not included
- **Interactions** between particles. Dry powders have them; sugar-dispersed samples much less. If a dry powder needs a larger K than the sugar-dispersed one, that is the signature of interactions.
- **Self-heating** during the run (temperature is fixed), and the temperature dependence of Ms and K.
- **Brownian rotation**, which is irrelevant for immobilised powders but matters in liquids.
- **Cubic anisotropy**: K is one effective uniaxial number.
- Very small barriers (KV/k<sub>B</sub>T below about 3), where the two-level picture is rough.

### Reference
J. Carrey, B. Mehdaoui, M. Respaud, *Simple models for dynamic hysteresis loop calculations of magnetic single-domain nanoparticles*, J. Appl. Phys. **109**, 083921 (2011).
""", unsafe_allow_html=True)
