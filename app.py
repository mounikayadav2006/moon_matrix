"""
MoonMatrix AI — Multi-modal Lunar Image Correspondence & Registration
Simplified single-flow prototype for SIH26166.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.sample_data import ensure_sample_data
from src.utils.image_io import load_image_from_upload, load_image_from_path, InvalidImageError
from src.features.sift import draw_keypoints
from src.matching.matcher import draw_matches
from src.matching.cross_modal import classify_sensor_pair
from src.geometry.ransac import draw_inliers_outliers
from src.spatial.grid import draw_grid_overlay
from src.evaluation.robustness import run_scale_experiment, run_illumination_experiment
from src.geospatial.coords import load_geo_metadata
from src.mapping.lunar_map import plot_footprint_on_basemap
from src.pipeline import run_full_pipeline

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

st.set_page_config(page_title="MoonMatrix AI", page_icon="🌙", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

div[data-testid="stMetric"] {
    background-color: #161B22;
    border: 1px solid #262C36;
    border-radius: 10px;
    padding: 12px 14px;
}
.status-pill {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-weight: 600; font-size: 0.9rem;
}
.status-ok { background-color: #10331f; color: #4ADE80; border: 1px solid #1f5c37; }
.status-fail { background-color: #331010; color: #F87171; border: 1px solid #5c1f1f; }

.mm-hero {
    background: radial-gradient(circle at 15% 20%, #1b2a4a 0%, #0E1117 55%),
                radial-gradient(circle at 85% 80%, #2a1b3d 0%, transparent 40%);
    border: 1px solid #262C36;
    border-radius: 20px;
    padding: 48px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.mm-hero::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,229,255,0.18) 0%, transparent 70%);
}
.mm-hero-badge {
    display: inline-block; background: #10202b; color: #00E5FF;
    border: 1px solid #12384a; padding: 5px 14px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 18px;
}
.mm-hero h1 {
    font-size: 2.6rem; font-weight: 700; margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff 0%, #9fd8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.mm-hero p.sub {
    font-size: 1.1rem; color: #9BA3AF; max-width: 620px; margin-bottom: 0;
}

.mm-step-card {
    background-color: #161B22; border: 1px solid #262C36; border-radius: 14px;
    padding: 18px 18px 16px 18px; height: 100%;
}
.mm-step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: #10202b; color: #00E5FF; font-size: 0.8rem; font-weight: 700;
    margin-bottom: 10px;
}
.mm-step-card h4 { margin: 0 0 6px 0; font-size: 1rem; color: #E6E6E6; }
.mm-step-card p { margin: 0; font-size: 0.85rem; color: #8B93A1; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "reference": None, "source": None,
        "sensor_ref": "OHRC", "sensor_src": "TMC-2",
        "ground_truth_H": None,
        "result": None,
        "scale_results": None, "illum_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def load_demo_data():
    paths = ensure_sample_data(SAMPLES_DIR)
    st.session_state.reference = load_image_from_path(paths["reference"])
    st.session_state.source = load_image_from_path(paths["source"])
    gt_path = os.path.join(SAMPLES_DIR, "demo_ground_truth_H.npy")
    st.session_state.ground_truth_H = np.load(gt_path) if os.path.exists(gt_path) else None
    st.session_state.result = None
    st.session_state.scale_results = None
    st.session_state.illum_results = None


# ==========================================================================
# Sidebar — everything needed to run a full analysis, in one place
# ==========================================================================
with st.sidebar:
    st.markdown("## 🌙 MoonMatrix AI")
    st.caption("Multi-modal Lunar Image Correspondence & Registration")
    st.caption("SIH26166 · Hackathon Prototype")
    st.divider()

    st.markdown("### 1. Images")
    tab1, tab2 = st.tabs(["Sample", "Upload"])
    with tab1:
        st.caption("Synthetic lunar-like demo pair with a known transform.")
        if st.button("Load Sample Images", use_container_width=True):
            load_demo_data()
            st.success("Loaded.")
    with tab2:
        ref_file = st.file_uploader("Reference", type=["png", "jpg", "jpeg", "tif", "tiff"], key="ref_up")
        src_file = st.file_uploader("Source", type=["png", "jpg", "jpeg", "tif", "tiff"], key="src_up")
        if ref_file and src_file:
            try:
                st.session_state.reference = load_image_from_upload(ref_file)
                st.session_state.source = load_image_from_upload(src_file)
                st.session_state.ground_truth_H = None
                st.session_state.result = None
                st.success("Loaded.")
            except InvalidImageError as e:
                st.error(str(e))

    st.markdown("### 2. Sensors")
    sensor_options = ["OHRC", "TMC-2", "IIRS", "Unknown / Other"]
    st.session_state.sensor_ref = st.selectbox("Reference sensor", sensor_options,
                                                 index=sensor_options.index(st.session_state.sensor_ref))
    st.session_state.sensor_src = st.selectbox("Source sensor", sensor_options,
                                                 index=sensor_options.index(st.session_state.sensor_src))

    with st.expander("Advanced settings"):
        grid_size = st.slider("Spatial grid size", 2, 8, 4)

    st.divider()
    ready = st.session_state.reference is not None and st.session_state.source is not None
    run_clicked = st.button("▶ Run Full Analysis", type="primary", use_container_width=True, disabled=not ready)
    if not ready:
        st.caption("Load or upload both images to enable this.")

    if run_clicked:
        with st.spinner("Running detection, matching, RANSAC, registration, refinement..."):
            result = run_full_pipeline(
                st.session_state.reference, st.session_state.source,
                ground_truth_H=st.session_state.ground_truth_H, grid_size=grid_size,
            )
        st.session_state.result = result
        st.session_state.scale_results = None
        st.session_state.illum_results = None


# ==========================================================================
# Main area
# ==========================================================================
result = st.session_state.result

if st.session_state.reference is None:
    st.markdown("""
    <div class="mm-hero">
        <div class="mm-hero-badge">SIH26166 · Hackathon Prototype</div>
        <h1>🌙 MoonMatrix AI</h1>
        <p class="sub">A complete correspondence &amp; registration pipeline for Chandrayaan-2
        optical imagery — SIFT detection, RANSAC-verified matching, spatial coverage analysis,
        registration and sub-pixel refinement, all in one run.</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Load Images", "Use the built-in synthetic sample pair, or upload your own reference + source images."),
        ("2", "Pick Sensors", "Tell MoonMatrix which Chandrayaan-2 instrument each image is from — OHRC, TMC-2, or IIRS."),
        ("3", "Run Analysis", "One click runs detection, matching, RANSAC, registration and refinement end-to-end."),
        ("4", "Explore Results", "Browse features, matches, spatial coverage, registration and robustness in clean tabs."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="mm-step-card">
                <div class="mm-step-num">{num}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.info("👈 Start in the sidebar: **Load Sample Images**, then **▶ Run Full Analysis**.")
    st.stop()

if result is None:
    st.markdown("""
    <div class="mm-hero">
        <div class="mm-hero-badge">Images Loaded</div>
        <h1>Ready to Analyze</h1>
        <p class="sub">Reference and source images are loaded below. Click
        <b>▶ Run Full Analysis</b> in the sidebar to run the full pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(st.session_state.reference, caption="Reference", use_container_width=True)
    with c2:
        st.image(st.session_state.source, caption="Source", use_container_width=True)
    st.stop()

st.title("🌙 MoonMatrix AI")
st.caption("Multi-modal Lunar Image Correspondence & Registration — SIH26166")

# ---- Result is available: show everything in tabs ----
tabs = st.tabs([
    "📊 Overview", "🔍 Features", "🔗 Matching & RANSAC", "🗺️ Spatial Coverage",
    "🛰️ Registration", "🎯 Sub-pixel", "🧪 Robustness", "🌙 Lunar Map", "ℹ️ About",
])

m = result.metrics
pair_info = classify_sensor_pair(st.session_state.sensor_ref, st.session_state.sensor_src)

# ---- Overview ----
with tabs[0]:
    status_html = (
        '<span class="status-pill status-ok">✔ SUCCESS</span>' if m.success
        else '<span class="status-pill status-fail">✕ FAILED</span>'
    )
    st.markdown(f"### Result: {status_html}", unsafe_allow_html=True)

    if pair_info.is_hard_cross_modal:
        st.caption(f"⚠️ {pair_info.note}")

    if not m.success:
        st.error(m.failure_reason or "Pipeline could not complete.")
        if result.notes:
            with st.expander("What was tried"):
                for n in result.notes:
                    st.write("• " + n)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Keypoints", f"{m.num_keypoints_ref} / {m.num_keypoints_src}")
        c2.metric("Good Matches", m.num_good_matches)
        c3.metric("Inliers", m.num_inliers)
        c4.metric("Inlier Ratio", f"{m.inlier_ratio:.1%}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Spatial Coverage", f"{m.spatial_coverage_pct:.0f}%")
        c6.metric("Distribution Score", f"{m.distribution_score:.2f}")
        c7.metric("Registration Error", f"{m.registration_error_px:.2f} px" if m.registration_error_px is not None else "N/A")
        c8.metric("Runtime", f"{m.runtime_sec:.2f}s")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.image(st.session_state.reference, caption="Reference", use_container_width=True)
        with c2:
            st.image(result.reg_res.overlay if result.reg_res else st.session_state.source,
                     caption="Registered overlay (green=reference, red=aligned source)", use_container_width=True)

# ---- Features ----
with tabs[1]:
    if result.ref_feat is None:
        st.warning("Run the analysis first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.image(draw_keypoints(result.reference_proc, result.ref_feat),
                     caption=f"Reference — {result.ref_feat.num_keypoints} keypoints", use_container_width=True)
        with c2:
            st.image(draw_keypoints(result.source_proc, result.src_feat),
                     caption=f"Source — {result.src_feat.num_keypoints} keypoints", use_container_width=True)

# ---- Matching & RANSAC ----
with tabs[2]:
    if result.match_res is None:
        st.warning("Not enough keypoints to attempt matching.")
    else:
        st.image(
            draw_matches(result.reference_proc, result.ref_feat.keypoints,
                        result.source_proc, result.src_feat.keypoints, result.match_res.good_matches),
            caption=f"{result.match_res.num_good} good matches (of {result.match_res.num_candidates} candidates)",
            use_container_width=True,
        )
        if result.ransac_res is not None:
            rr = result.ransac_res
            st.caption(f"RANSAC ({rr.transform_type}): {rr.num_inliers} inliers, {rr.num_outliers} outliers "
                       f"— {rr.inlier_ratio:.1%} inlier ratio.")
            st.image(
                draw_inliers_outliers(result.reference_proc, result.ref_feat.keypoints,
                                      result.source_proc, result.src_feat.keypoints,
                                      result.match_res.good_matches, rr.inlier_mask),
                caption="Green = inliers, Red = outliers", use_container_width=True,
            )
        else:
            st.warning("Not enough good matches for RANSAC verification.")

# ---- Spatial Coverage ----
with tabs[3]:
    if result.spatial_res is None:
        st.warning("Spatial analysis needs a successful RANSAC verification.")
    else:
        sres = result.spatial_res
        c1, c2, c3 = st.columns(3)
        c1.metric("Occupied cells", f"{sres.occupied_cells}/{sres.total_cells}")
        c2.metric("Coverage", f"{sres.spatial_coverage_pct:.1f}%")
        c3.metric("Distribution score", f"{sres.distribution_score:.2f}")

        inlier_pts = np.float32(
            [result.ref_feat.keypoints[mm.queryIdx].pt for mm, keep in
             zip(result.match_res.good_matches, result.ransac_res.inlier_mask) if keep]
        )
        st.image(draw_grid_overlay(result.reference_proc, inlier_pts, grid_size),
                 caption="Inlier correspondences over spatial grid", use_container_width=True)
        st.dataframe(pd.DataFrame(sres.matches_per_cell), use_container_width=True)

# ---- Registration ----
with tabs[4]:
    if result.reg_res is None:
        st.warning("Registration needs a successful RANSAC verification.")
    else:
        reg = result.reg_res
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result.reference_proc, caption="Reference", use_container_width=True)
        with c2:
            st.image(result.source_proc, caption="Source (original)", use_container_width=True)
        with c3:
            st.image(reg.registered_image, caption="Registered source", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.image(reg.overlay, caption="Overlay", use_container_width=True)
        with c2:
            st.image(reg.difference, caption="Difference map", use_container_width=True)

        if m.registration_error_px is not None:
            st.metric("Registration error vs ground truth", f"{m.registration_error_px:.3f} px")
        else:
            st.caption("No ground truth available for this image pair — use the overlay/difference "
                       "map above for a visual quality check.")

# ---- Sub-pixel ----
with tabs[5]:
    if result.subpixel_res is None or result.subpixel_res.num_refined == 0:
        st.warning("Sub-pixel refinement needs a successful RANSAC verification with usable points.")
    else:
        sr = result.subpixel_res
        st.caption("Prototype: local phase-correlation based sub-pixel refinement.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Points refined", sr.num_refined)
        c2.metric("Points skipped", sr.skipped)
        c3.metric("Mean displacement", f"{sr.mean_displacement_px:.3f} px")
        df = pd.DataFrame({
            "Initial X": sr.initial_points[:, 0], "Initial Y": sr.initial_points[:, 1],
            "Refined X": sr.refined_points[:, 0], "Refined Y": sr.refined_points[:, 1],
            "Displacement (px)": np.linalg.norm(sr.displacements, axis=1),
        })
        st.dataframe(df, use_container_width=True, height=250)

# ---- Robustness ----
with tabs[6]:
    st.caption("Reruns the pipeline on scaled / illumination-varied copies of the source image.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run Scale Experiment", use_container_width=True):
            with st.spinner("Testing scale robustness..."):
                st.session_state.scale_results = run_scale_experiment(
                    result.reference_proc, result.source_proc, grid_size=grid_size,
                )
    with c2:
        if st.button("Run Illumination Experiment", use_container_width=True):
            with st.spinner("Testing illumination robustness..."):
                st.session_state.illum_results = run_illumination_experiment(
                    result.reference_proc, result.source_proc, grid_size=grid_size,
                )

    if st.session_state.scale_results:
        st.write("**Scale robustness**")
        df = pd.DataFrame([r.as_dict() for r in st.session_state.scale_results])
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index("Experiment")[["Inlier Ratio", "Distribution Score"]])

    if st.session_state.illum_results:
        st.write("**Illumination robustness**")
        df = pd.DataFrame([r.as_dict() for r in st.session_state.illum_results])
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index("Experiment")[["Inlier Ratio", "Distribution Score"]])

# ---- Lunar Map ----
with tabs[7]:
    st.caption("Optional geospatial visualization — shown only when footprint metadata is supplied.")
    meta_file = st.file_uploader("Metadata JSON (lat/lon corners)", type=["json"], key="meta_up")
    if meta_file is not None:
        tmp_path = os.path.join(DATA_DIR, "metadata", "uploaded_metadata.json")
        with open(tmp_path, "wb") as f:
            f.write(meta_file.getbuffer())
        geo = load_geo_metadata(tmp_path)
    else:
        geo = load_geo_metadata(None)

    if not geo.available:
        st.info("Geospatial metadata unavailable for this dataset.")
    else:
        st.pyplot(plot_footprint_on_basemap(geo))

# ---- About ----
with tabs[8]:
    st.markdown("""
**MoonMatrix AI** runs a classical computer-vision pipeline — SIFT feature
detection, BFMatcher + Lowe's ratio matching, RANSAC geometric verification,
grid-based spatial coverage analysis, affine/homography registration, and
phase-correlation sub-pixel refinement — with automatic parameter fallbacks
so a reasonable image pair reliably produces a result.

**Built for:** SIH26166 — Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).

**Honest scope:**
- Cross-modal pairs involving IIRS use the same classical baseline and are
  flagged above as an advanced/research-mode preview, not a validated result.
- Illumination robustness uses simulated brightness/contrast/gamma changes,
  not real Sun-angle observations.
- Sub-pixel refinement is a local, patch-based prototype.
- Registration error is only numeric when a known ground truth exists
  (the bundled synthetic demo pair); user-uploaded pairs show N/A.

See `LIMITATIONS.md` and `ARCHITECTURE.md` in the project folder for full detail.
""")
