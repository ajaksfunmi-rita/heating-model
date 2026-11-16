# Nanoparticle hysteresis simulator

Enter the material, particles, sample and field. The app draws the magnetisation loop an AC magnetometer or a VSM would record
(low-field ellipse, high-field hysteresis loop), plus Hc, Mr, saturation reached, energy per cycle and SAR.

## Files
| File | Job |
|---|---|
| `engine.py` | All the physics. No Streamlit in here. |
| `app.py` | The interface. Reads widgets, calls the engine, plots. |
| `materials.json` | Material database (typical values to verify). Add your own entries here. |
| `test_engine.py` | Physics checks (ellipse, area ∝ H², equilibrium limit, blocked limit, trends). Run `python test_engine.py`. |
| `requirements.txt`, `.streamlit/config.toml` | Dependencies and theme. |

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Put it online (Streamlit Community Cloud)
1. In your GitHub repository, delete the old files and upload these (keep the `.streamlit` folder).
2. Go to share.streamlit.io, sign in with GitHub, choose New app.
3. Pick the repository and branch, set the main file to `app.py`, and press Deploy.
Every later commit to GitHub updates the app automatically.
