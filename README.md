# 🌙 MoonMatrix AI

**Multi-modal Lunar Image Correspondence & Registration**
Internal hackathon prototype for **SIH26166** — *Multi-modal, Sun angle and scale
invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

---

## Quick start

```bash
# 1. (recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Example workflow (one button, all results)
1. In the sidebar, click **"Load Sample Images"** (or upload your own reference/source pair)
2. Click **▶ Run Full Analysis**
3. Everything — features, matches, RANSAC, spatial coverage, registration,
   sub-pixel refinement — appears automatically across the result tabs
4. Optionally run the **🧪 Robustness** tab's scale/illumination experiments

The pipeline (`src/pipeline.py`) automatically retries with more permissive
(but still principled) parameters if a stage under-delivers — e.g. relaxing
the SIFT contrast threshold if too few keypoints are found, relaxing Lowe's
ratio if too few matches survive, or falling back from homography to affine
if RANSAC can't fit a homography — so a reasonable image pair reliably
produces a complete result without manual tuning.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project structure

```text
moonmatrix-ai/
├── app.py                  # Streamlit application (all pages)
├── requirements.txt
├── README.md / DATA_GUIDE.md / ARCHITECTURE.md / EVALUATION.md / LIMITATIONS.md
├── data/
│   ├── raw/ processed/ samples/ metadata/
├── src/
│   ├── pipeline.py                     # orchestrator: runs all stages with auto-tuning fallbacks
│   ├── preprocessing/preprocess.py     # grayscale, normalize, CLAHE, contrast, denoise
│   ├── features/sift.py                # SIFT detection
│   ├── matching/matcher.py             # BFMatcher + KNN + Lowe ratio test
│   ├── matching/cross_modal.py         # sensor-pair classification + advanced-matcher interface
│   ├── geometry/ransac.py              # RANSAC verification (homography/affine)
│   ├── spatial/grid.py                 # grid-based spatial distribution analysis
│   ├── registration/register.py        # warp + overlay + difference + ground-truth error
│   ├── subpixel/refine.py              # phase-correlation sub-pixel refinement (prototype)
│   ├── geospatial/coords.py            # metadata-gated pixel→lat/lon (never fabricated)
│   ├── mapping/lunar_map.py            # footprint plotting when metadata exists
│   ├── evaluation/metrics.py           # PipelineMetrics dataclass
│   ├── evaluation/robustness.py        # scale + illumination experiments
│   └── utils/                          # sample-data generator, image IO
└── tests/                              # pytest unit tests for core modules
```

---

## IMPLEMENTED NOW

- Grayscale conversion, normalization, CLAHE, contrast adjustment, denoising (toggleable)
- SIFT keypoint detection with count + visualization
- BFMatcher + KNN + Lowe's ratio test, with candidate/good match counts and visualization
- RANSAC-based homography **and** affine estimation, with inlier/outlier separation,
  inlier ratio, and explicit **rejection** (not forcing) when matches/inliers are insufficient
- 4×4 (adjustable N×N) grid-based spatial distribution analysis: occupied cells,
  coverage %, matches-per-cell, entropy-based distribution score
- Image registration (affine/homography warp), overlay, absolute-difference map
- Registration error **vs known ground truth** for the synthetic demo pair (real numeric
  error, not fabricated) — reported as "N/A" honestly for user-uploaded pairs with no ground truth
- Sub-pixel refinement prototype using local phase correlation, clearly labeled as a prototype
- Scale robustness experiment (0.5×–1.5×) with real recomputed metrics per scale
- Illumination robustness experiment (brightness/contrast/gamma) — labeled **simulated**
- Sensor-pair classification (OHRC/TMC-2/IIRS) that flags hard cross-modal pairs as
  "advanced/research mode" rather than pretending they're solved
- Evaluation dashboard + experiment comparison table with real computed values
- Hackathon "one-screen" demo summary
- Synthetic demo dataset generated on first run, clearly labeled as non-Chandrayaan-2 data
- Error handling for: invalid images, too-small images, insufficient keypoints,
  insufficient matches, RANSAC failure, missing geospatial metadata

## PROTOTYPE / APPROXIMATION

- Sub-pixel refinement is a local patch-based phase-correlation estimate — not a
  validated, production-grade sub-pixel geodetic registration method
- Illumination robustness uses **simulated** brightness/contrast/gamma changes,
  not real Sun-angle-varying Chandrayaan-2 image pairs
- Cross-modal results (any pair involving IIRS) reuse the classical SIFT baseline
  and are explicitly labeled as an advanced/research-mode preview
- Lunar map footprint uses a simple linear lat/lon interpolation across the image,
  not a rigorous cartographic projection

## FUTURE (AFTER SIH SELECTION)

- SuperPoint / LightGlue / LoFTR integration for learned, illumination- and
  cross-modal-robust feature matching
- Real Chandrayaan-2 PDS3/PDS4 label parsing for accurate geospatial metadata
- DEM-aware / ortho-rectified registration
- Large-scale benchmarking on real Chandrayaan-2 OHRC/TMC-2/IIRS datasets
- Real Sun-angle-varying dataset collection and evaluation
- Improved, validated sub-pixel correspondence estimation

See `LIMITATIONS.md` for the itemized detail behind each of the above.
