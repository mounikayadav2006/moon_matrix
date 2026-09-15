# Data Guide

## Demo data (bundled)

On first run, `src/utils/sample_data.py` procedurally generates a
`demo_reference.png` / `demo_source.png` pair under `data/samples/` — a
grayscale texture with painted "craters" and a **known** rotation + scale +
translation applied to create the source image from the reference. A
`demo_ground_truth_H.npy` file stores the exact transform, which is why the
**Registration** page can report a genuine registration error in pixels for
this pair specifically.

**These images are not Chandrayaan-2 data.** The UI labels them
"Demo Dataset — Synthetic Test Data" everywhere they appear.

## Using real Chandrayaan-2 imagery

1. Download OHRC / TMC-2 / IIRS products from ISRO's **Pradan** portal
   (https://pradan.issdc.gov.in/) for the sensor(s) you want to test.
2. Most Chandrayaan-2 products are distributed as PDS3/PDS4-labelled image
   files (often `.img` / `.qub` alongside a `.lbl` / `.xml` label). Convert
   the image data to a standard 8-bit grayscale PNG/TIFF before uploading —
   this prototype's uploader accepts PNG/JPG/TIFF only (`src/utils/image_io.py`).
   A GDAL-based conversion script is a natural first addition post-selection
   (see LIMITATIONS.md).
3. On the **📤 Image Input** page, use the "Upload Your Own" tab to load your
   converted reference and source images, and set the correct sensor labels.
4. Because uploaded pairs have no known ground-truth transform, the
   Registration page will correctly report registration error as **N/A** —
   use the overlay/difference visualizations for a qualitative check instead.

## Supplying geospatial metadata for the Lunar Map page

The Lunar Map page (`src/geospatial/coords.py`) accepts a simple JSON sidecar:

```json
{
  "lat_top_left": -12.34,
  "lon_top_left": 45.67,
  "lat_bottom_right": -12.50,
  "lon_bottom_right": 45.90
}
```

If you don't supply one, the app explicitly displays
**"Geospatial metadata unavailable for this dataset."** rather than guessing
coordinates. Parsing these values directly out of Chandrayaan-2 PDS labels is
listed under FUTURE WORK in `LIMITATIONS.md`.
