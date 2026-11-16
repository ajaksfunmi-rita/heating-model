"""
app.py  -  the interface (no physics in here)
=============================================
Pattern: read the widgets -> build one Sample per magnetic phase -> call engine.run_ac / engine.run_vsm
for each phase -> add the loops (weighted by mass) -> plot.
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
INK, TEAL, SLATE, AMBER, PLUM = "#1F2A44", "#0B7A75", "#6B7A90", "#B7791F", "#7A3E8E"

st.set_page_config(page_title="Nanoparticle hysteresis simulator", page_icon="🧲", layout="wide")
st.markdown("<style>.block-container{padding-top:2.2rem;max-width:1250px}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ database
DB = json.loads((Path(__file__).with_name("materials.json")).read_text(encoding="utf-8"))
CUSTOM = dict(name="Custom material", Ms_kA_m=400.0, K_kJ_m3=20.0, rho_kg_m3=5000.0,
              note="Enter your own values. Ms from your VSM plateau; K is the effective uniaxial anisotropy.")
NAMES = [m["name"] for m in DB] + [CUSTOM["name"]]

# ------------------------------------------------------------------ units
FIELD_UNITS = {"mT": 1e3, "Oe": 1e4}                       # mu0*H in tesla x factor
MOMENT_UNITS = ["emu/g (whole sample)", "emu (whole sample)", "emu/g (magnetic material)", "M / Ms"]
SUFFIX = {"emu/g (whole sample)": "emu/g", "emu (whole sample)": "emu", "emu/g (magnetic material)": "emu/g", "M / Ms": "Ms"}
LABEL = {"emu/g (whole sample)": "Moment per gram of sample (emu/g)", "emu (whole sample)": "Moment of the whole sample (emu)",
         "emu/g (magnetic material)": "Moment per gram of magnetic material (emu/g)", "M / Ms": "Fraction of full saturation (M / Ms)"}


def fmt(x, digits=3):
    return f"{x:.{digits}g}"


def field_to_display(H_Apm, unit):
    return np.asarray(H_Apm) * MU0 * FIELD_UNITS[unit]


# ------------------------------------------------------------------ sidebar
def phase_inputs(tag, title, default_name, default_D, default_sigma):
    """Material + particle widgets for one magnetic phase."""
    st.subheader(title)
    choice = st.selectbox("Material", NAMES, index=NAMES.index(default_name), key=f"mat_{tag}",
                          help="Fills in typical values. Overwrite them with your own measurements.")
    base = next((m for m in DB if m["name"] == choice), CUSTOM)
    Ms_k = st.number_input("Saturation magnetisation Ms (kA/m)", 10.0, 2000.0, float(base["Ms_kA_m"]), 5.0,
                           key=f"Ms_{tag}_{choice}", help="Per volume of this magnetic material. The VSM plateau only gives it if the sample really saturates.")
    K_k = st.number_input("Effective anisotropy K (kJ/m³)", 0.1, 1000.0, float(base["K_kJ_m3"]), 1.0,
                          key=f"K_{tag}_{choice}", help="One effective uniaxial number. It also absorbs shape, surface and interaction effects.")
    rho = st.number_input("Density (kg/m³)", 1000.0, 20000.0, float(base["rho_kg_m3"]), 10.0, key=f"rho_{tag}_{choice}")
    st.caption(f"Ms = {Ms_k * 1e3 / rho:.0f} emu/g.  {base['note']}")
    D_nm = st.slider("Median diameter (nm)", 3.0, 100.0, float(default_D), 0.5, key=f"D_{tag}", help="From TEM, or an assumption if unknown.")
    sigma = st.slider("Size spread σ (log-normal)", 0.0, 0.6, float(default_sigma), 0.01, key=f"s_{tag}",
                      help="Width of ln(D). About 0.1 is very narrow, 0.3 is broad. 0 = every particle the same size.")
    return dict(name=choice, Ms=Ms_k * 1e3, K=K_k * 1e3, rho=rho, D=D_nm * 1e-9, sigma=sigma)


with st.sidebar:
    st.header("Inputs")
    two = st.toggle("Two-phase sample (soft + hard)", value=False,
                    help="For a mixture of two magnetic materials, for example nickel zinc ferrite + barium hexaferrite. "
                         "Each phase gets its own inputs and the two loops are added by mass.")
    ph1 = phase_inputs("1", "Phase 1 (soft)" if two else "Material and particles", "Nickel zinc ferrite NiZnFe2O4", 12.0, 0.20)
    ph2 = phase_inputs("2", "Phase 2 (hard)", "Barium hexaferrite BaFe12O19 (Sc-substituted)", 30.0, 0.30) if two else None

    st.subheader("Sample")
    mass_mg = st.number_input("Total sample mass (mg)", 0.01, 1000.0, 3.584, 0.1,
                              help="Everything that was weighed: magnetic powder plus sugar or other support.")
    frac = st.number_input("Magnetic fraction (% of sample mass)", 1.0, 100.0, 100.0, 1.0,
                           help="100 for pure powder. Less if the particles sit in sugar or on a support such as CNC/CNF.")
    share2 = st.slider("Hard phase share of the magnetic mass (%)", 0, 100, 50, 1) if two else 0
    T = st.number_input("Temperature (K)", 4.0, 700.0, 300.0, 5.0)

    st.subheader("Measurement")
    mode_label = st.radio("Instrument", ["VSM", "AC magnetometer"], horizontal=True)
    mode = "ac" if mode_label == "AC magnetometer" else "vsm"
    if mode == "ac":
        amp_mT = st.slider("Field amplitude μ₀H (mT)", 1.0, 150.0, 30.0, 1.0)
        freq_kHz = st.number_input("Frequency (kHz)", 0.1, 2000.0, 100.0, 10.0)
        hmax_T, loop_time = 1.0, 474.0
    else:
        hmax_T = st.slider("Maximum field (T)", 0.1, 5.0, 1.0, 0.1, help="1 T = 10 kOe.")
        loop_time = st.number_input("Time for one full loop (s)", 30.0, 7200.0, 474.0, 30.0,
                                    help="Elapsed time shown by the VSM software. Slower sweeps give particles more time to relax.")
        amp_mT, freq_kHz = 30.0, 100.0

    with st.expander("Advanced"):
        f0 = st.select_slider("Attempt frequency f₀ (Hz)", options=[1e8, 1e9, 1e10, 1e11, 1e12], value=1e9,
                              format_func=lambda x: f"{x:.0e}",
                              help="Rate prefactor in the Néel law. 10⁹ Hz is the usual choice; literature values range from 10⁹ to 10¹².")
        cycles = st.slider("Simulated cycles (AC)", 2, 20, 4, help="The last cycle is shown. More cycles remove the start-up transient.")

    st.subheader("Display")
    f_unit = st.radio("Field unit", list(FIELD_UNITS), horizontal=True)
    m_unit = st.radio("Moment unit", MOMENT_UNITS, help="The VSM software reports emu/g of the whole sample.")

# ------------------------------------------------------------------ assemble the phases
frac_tot = frac / 100.0
ph1["f"] = frac_tot * (1.0 - share2 / 100.0)
phases = [ph1] + ([dict(ph2, f=frac_tot * share2 / 100.0)] if two else [])
mass_g = mass_mg * 1e-3
G_sat = sum(p["f"] * p["Ms"] / p["rho"] for p in phases)       # full saturation, emu/g of sample
f_sum = sum(p["f"] for p in phases)


def sample_of(p):
    return eng.Sample(Ms=p["Ms"], K=p["K"], D_mean=p["D"], sigma=p["sigma"], rho=p["rho"], T=T, f0=f0)


def G_disp(G, unit):
    """emu/g of whole sample -> chosen display unit."""
    G = np.asarray(G)
    return {"emu/g (whole sample)": G, "emu (whole sample)": G * mass_g,
            "emu/g (magnetic material)": G / frac_tot, "M / Ms": G / G_sat}[unit]


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


def stats(H, G):
    """Hc (A/m), remanence (emu/g of sample) and peak (fraction of full saturation) of a combined loop."""
    hc = eng._at_zero_of(G, H)
    mr = eng._at_zero_of(H, G)
    return (float(np.mean(np.abs(hc))) if len(hc) else 0.0,
            float(np.mean(np.abs(mr))) if len(mr) else 0.0,
            float(np.max(np.abs(G)) / G_sat))


# ------------------------------------------------------------------ header
st.title("Nanoparticle hysteresis simulator")
st.write("Describe the powder and the field. The model draws the magnetisation loop that a VSM or an AC magnetometer "
         "would record, from one set of equations, so the same sample parameters give the low-field ellipse and the "
         "high-field hysteresis loop. Switch on the two-phase option for a soft + hard mixture.")

tab_loop, tab_series, tab_how = st.tabs(["Loop", "Several fields", "How it works"])


def parse_upload(file):
    raw = file.getvalue().decode("utf-8", errors="ignore")
    if "[Data]" in raw:                                        # e.g. Quantum Design exports
        raw = raw.split("[Data]", 1)[1]
    df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", comment="#", skip_blank_lines=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    if df.shape[1] < 2 or len(df) < 5:
        raise ValueError("Found fewer than two numeric columns or too few rows.")
    return df


# ================================================================== TAB 1
with tab_loop:
    H_Apm = amp_mT * 1e-3 / MU0 if mode == "ac" else hmax_T / MU0
    with st.spinner("Simulating…"):
        runs = [cached_run(mode, p["Ms"], p["K"], p["D"], p["sigma"], p["rho"], T, f0, H_Apm, freq_kHz * 1e3, loop_time, cycles)
                for p in phases]
    run0 = runs[0]
    H, loop = run0.H, run0.loop
    G_parts = [p["f"] * r.M / p["rho"] for p, r in zip(phases, runs)]      # emu/g of whole sample
    G = np.sum(G_parts, axis=0)
    Hc, Mr, sat = stats(H[loop], G[loop])
    energy = sum(p["f"] * r.metrics["area"] / p["rho"] for p, r in zip(phases, runs)) / f_sum        # mJ/g of magnetic material
    sar = sum(p["f"] * r.metrics.get("sar", 0.0) for p, r in zip(phases, runs)) / f_sum              # W/g of magnetic material

    # ---- numbers
    cols = st.columns(5 if mode == "ac" else 4)
    cols[0].metric("Peak magnetisation", f"{sat * 100:.0f}% of Ms", help="Largest moment reached, relative to full saturation of all phases.")
    cols[1].metric("Coercivity Hc", f"{fmt(float(field_to_display(Hc, f_unit)))} {f_unit}", help="Field at which M crosses zero.")
    cols[2].metric("Remanence Mr", f"{fmt(float(G_disp(Mr, m_unit)))} {SUFFIX[m_unit]}", help="Moment at zero field.")
    cols[3].metric("Energy per cycle", f"{fmt(energy)} mJ/g", help="Loop area per gram of magnetic material.")
    if mode == "ac":
        cols[4].metric("Heating power (SAR)", f"{fmt(sar)} W/g", help="Loop area × frequency, per gram of magnetic material.")

    # ---- plot
    fig = go.Figure()
    if mode == "vsm":
        k = loop.start + 1
        fig.add_trace(go.Scatter(x=field_to_display(H[:k], f_unit), y=G_disp(G[:k], m_unit), mode="lines",
                                 line=dict(color=SLATE, width=1.5, dash="dot"), name="First magnetisation"))
    elif st.session_state.get("show_all", False) and cycles > 1:
        fig.add_trace(go.Scatter(x=field_to_display(H, f_unit), y=G_disp(G, m_unit), mode="lines",
                                 line=dict(color=SLATE, width=1, dash="dot"), name="All cycles"))
    if two:
        for gp, p, col, lab in zip(G_parts, phases, (AMBER, PLUM), ("Phase 1 alone", "Phase 2 alone")):
            fig.add_trace(go.Scatter(x=field_to_display(H[loop], f_unit), y=G_disp(gp[loop], m_unit), mode="lines",
                                     line=dict(color=col, width=1.8, dash="dash"), name=f"{lab}: {p['name'].split(' (')[0]}"))
    fig.add_trace(go.Scatter(x=field_to_display(H[loop], f_unit), y=G_disp(G[loop], m_unit), mode="lines",
                             line=dict(color=TEAL, width=3.5), name="Simulated total" if two else "Simulated"))
    hc_d = float(field_to_display(Hc, f_unit))
    fig.add_trace(go.Scatter(x=[hc_d, -hc_d], y=[0, 0], mode="markers", marker=dict(color=INK, size=9, symbol="diamond"),
                             name="±Hc", hovertemplate=f"Hc = {fmt(hc_d)} {f_unit}<extra></extra>"))
    mr_d = float(G_disp(Mr, m_unit))
    fig.add_trace(go.Scatter(x=[0, 0], y=[mr_d, -mr_d], mode="markers", marker=dict(color=AMBER, size=9, symbol="square"),
                             name="±Mr", hovertemplate=f"Mr = {fmt(mr_d)}<extra></extra>"))

    # ---- optional measured overlay
    with st.expander("Overlay measured data (optional)"):
        st.caption("Upload a loop as CSV or text (background-corrected), pick the columns and units, and it is drawn over the simulation. "
                   "The VSM software reports emu/g of the whole sample, which is the default here.")
        up = st.file_uploader("Measured loop", type=["csv", "txt", "dat"], label_visibility="collapsed")
        meas = None
        if up is not None:
            try:
                df = parse_upload(up)
                c1, c2, c3, c4 = st.columns(4)
                hcol = c1.selectbox("Field column", df.columns, index=0)
                mcol = c2.selectbox("Moment column", df.columns, index=1)
                hu = c3.selectbox("Field unit", ["Oe", "kOe", "mT", "T", "kA/m"])
                mu = c4.selectbox("Moment unit", ["emu/g (whole sample)", "emu", "emu/g (magnetic material)", "M / Ms (normalised)"])
                H_T = df[hcol].to_numpy() * {"Oe": 1e-4, "kOe": 1e-1, "mT": 1e-3, "T": 1.0, "kA/m": 1e3 * MU0}[hu]
                raw = df[mcol].to_numpy()
                G_meas = {"emu": raw / mass_g, "emu/g (whole sample)": raw,
                          "emu/g (magnetic material)": raw * frac_tot, "M / Ms (normalised)": raw * G_sat}[mu]
                meas = (H_T / MU0, G_meas)
            except Exception as err:                                             # noqa: BLE001
                st.error(f"Could not read that file: {err}")
    if meas is not None:
        fig.add_trace(go.Scatter(x=field_to_display(meas[0], f_unit), y=G_disp(meas[1], m_unit), mode="markers",
                                 marker=dict(color="rgba(0,0,0,0)", size=6, line=dict(color=INK, width=1.2)), name="Measured"))

    fig.add_hline(y=0, line=dict(color="#C9D2DA", width=1))
    fig.add_vline(x=0, line=dict(color="#C9D2DA", width=1))
    fig.update_layout(template="plotly_white", height=540, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=f"Applied field μ₀H ({f_unit})", yaxis_title=LABEL[m_unit],
                      legend=dict(orientation="h", y=1.09, x=0), hovermode="closest", font=dict(color=INK, size=14))
    st.plotly_chart(fig, width="stretch")
    if mode == "ac":
        st.checkbox("Also show the start-up cycles", key="show_all")

    # ---- plain-language reading of the result
    for i, p in enumerate(phases, start=1):
        s = sample_of(p)
        Hk_mT = s.Hk * MU0 * 1e3
        t_meas = 1.0 / (freq_kHz * 1e3) if mode == "ac" else s.Hk * loop_time / (4.0 * H_Apm)
        t_txt = f"{t_meas * 1e6:.3g} µs" if t_meas < 1e-3 else f"{t_meas:.3g} s"
        reg = eng.regime(s, t_meas)
        who = f"**Phase {i} ({p['name'].split(' (')[0]}).** " if two else ""
        if mode == "ac":
            ratio = amp_mT / Hk_mT
            shape = ("low-field regime: expect a tilted ellipse" if ratio < 0.25 else
                     "intermediate regime: the ellipse starts to distort" if ratio < 0.6 else
                     "high-field regime: expect a full hysteresis loop with a clear coercive field")
            line1 = f"The field amplitude is {ratio * 100:.0f}% of its anisotropy field μ₀H<sub>k</sub> = {Hk_mT:.0f} mT, which is the {shape}. "
        else:
            reach = hmax_T * 1e3 / Hk_mT
            line1 = (f"Its anisotropy field is μ₀H<sub>k</sub> = {Hk_mT:.0f} mT and the sweep reaches {reach:.1f} times that"
                     f"{', so this phase is fully saturated.' if reach > 3 else ', so this phase does not fully saturate.'} ")
        st.markdown(f"{who}{line1}The field changes on a time scale of about {t_txt}; particles larger than ≈{reg['D_block'] * 1e9:.0f} nm "
                    f"cannot follow it and are blocked. **{reg['blocked_fraction'] * 100:.0f}%** of this phase's volume is above that size "
                    f"(median particle KV/k<sub>B</sub>T = {reg['xi']:.1f}).", unsafe_allow_html=True)
        if reg["xi"] < 3:
            st.warning(f"Phase {i}: the median particle has a very small energy barrier (KV/k_BT < 3). The two-level model "
                       "underestimates the low-field response there, so treat the result as qualitative.")

    loop_df = pd.DataFrame({"mu0H_mT": H[loop] * MU0 * 1e3, "H_Oe": H[loop] * MU0 * 1e4,
                            "moment_emu_per_g_sample": G[loop], "moment_emu": G[loop] * mass_g,
                            **{f"phase{i}_emu_per_g_sample": gp[loop] for i, gp in enumerate(G_parts, start=1)}})
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
            res = [cached_series(p["Ms"], p["K"], p["D"], p["sigma"], p["rho"], T, f0,
                                 tuple(float(a * 1e-3 / MU0) for a in amps_mT), freq_kHz * 1e3, cycles) for p in phases]
        colors = sample_colorscale("Viridis", list(np.linspace(0.05, 0.85, len(amps_mT))))

        fig2 = go.Figure()
        area, hc, mr, sat_ = [], [], [], []
        for k, (a, col) in enumerate(zip(amps_mT, colors)):
            Hs = res[0][k][0]
            Gs = sum(p["f"] * res[i][k][1] / p["rho"] for i, p in enumerate(phases))
            h_c, m_r, s_ = stats(Hs, Gs)
            hc.append(float(field_to_display(h_c, f_unit))); mr.append(float(G_disp(m_r, "M / Ms"))); sat_.append(s_)
            area.append(sum(p["f"] * res[i][k][2]["area"] / p["rho"] for i, p in enumerate(phases)) / f_sum)
            fig2.add_trace(go.Scatter(x=field_to_display(Hs, f_unit), y=G_disp(Gs, m_unit), mode="lines",
                                      line=dict(color=col, width=2.5), name=f"{a:g} mT"))
        fig2.add_hline(y=0, line=dict(color="#C9D2DA", width=1))
        fig2.add_vline(x=0, line=dict(color="#C9D2DA", width=1))
        fig2.update_layout(template="plotly_white", height=500, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title=f"Applied field μ₀H ({f_unit})", yaxis_title=LABEL[m_unit],
                           legend_title="Amplitude", font=dict(color=INK, size=14))
        st.plotly_chart(fig2, width="stretch")

        area, hc, mr, sat_ = map(np.array, (area, hc, mr, sat_))
        sub = make_subplots(rows=1, cols=3, subplot_titles=("Energy per cycle (mJ/g)", f"Coercivity Hc ({f_unit})",
                                                            "Mr and peak M (fraction of Ms)"))
        sub.add_trace(go.Scatter(x=amps_mT, y=area, mode="lines+markers", line=dict(color=TEAL, width=3), name="Loop area"), 1, 1)
        sub.add_trace(go.Scatter(x=amps_mT, y=area[0] * (amps_mT / amps_mT[0]) ** 2, mode="lines",
                                 line=dict(color=SLATE, dash="dash"), name="∝ H² guide"), 1, 1)
        sub.add_trace(go.Scatter(x=amps_mT, y=hc, mode="lines+markers", line=dict(color=INK, width=3), name="Hc", showlegend=False), 1, 2)
        sub.add_trace(go.Scatter(x=amps_mT, y=sat_, mode="lines+markers", line=dict(color=TEAL, width=3), name="M_max / Ms"), 1, 3)
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
| **Particles:** median size and size spread (from TEM, or an assumption) | **Hc**, **Mr** and how close the sample gets to **saturation** |
| **Sample:** total mass, magnetic fraction, temperature | **Energy per cycle** and **heating power (SAR)** |
| **Field:** AC amplitude and frequency, or a VSM sweep | The loop in **emu/g** or **emu**, to compare with your instrument |

Ms is an **input**: it is a property of the material. Hc, Mr and the loop shape are **outputs**, because they come from the competition between the field, the energy barrier and the thermal energy. K is rarely known for nanoparticles, so a practical use is to adjust K until a measured loop is matched.

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
The program repeats this for 61 particle sizes (log-normal) and 16 easy-axis orientations (random in 3D), and adds them up weighted by volume, which is what a magnetometer measures. The result is scaled by the magnetic mass to give emu/g of the whole sample, the quantity your VSM software reports.

### Two phases
For a mixture, each phase is simulated on its own and the loops are added, weighted by the mass of each phase. A soft phase (small K) flips at low field and a hard phase (large K) only at high field, which produces a kinked or pinched loop. **Exchange coupling between the phases is not included**, so a strongly coupled composite would deviate.

### What matters when the size is unknown
Particles that are blocked (large KV/k<sub>B</sub>T) give a loop set by K and not by size, so for a hard phase the size hardly matters above a few tens of nm. For a soft phase near the blocking size, the loop changes strongly with size. Try a few sizes and see which features move.

### The two instruments
They differ only in the field programme. A **VSM** sweeps slowly to about 1 T. An **AC magnetometer** oscillates at about 100 kHz, so particles whose relaxation time is longer than the period cannot follow, and the loop opens.

### What is not included
- **Interactions** between particles, and **exchange coupling** between phases.
- **Self-heating** during the run (temperature is fixed), and the temperature dependence of Ms and K.
- **Brownian rotation**, which is irrelevant for immobilised powders but matters in liquids.
- **Cubic anisotropy**: K is one effective uniaxial number.
- **Texture**: the model assumes randomly oriented particles, so a pressed cube with aligned platelets will differ along different axes.
- Very small barriers (KV/k<sub>B</sub>T below about 3), where the two-level picture is rough.

### Reference
J. Carrey, B. Mehdaoui, M. Respaud, *Simple models for dynamic hysteresis loop calculations of magnetic single-domain nanoparticles*, J. Appl. Phys. **109**, 083921 (2011).
""", unsafe_allow_html=True)
