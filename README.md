# Stylus Effect

Turn photos into **pencil-style vector engravings**: selectively broken contours, small clusters of closely spaced cuts, and a few internal lines describing surface volume. Exports editable **SVG** and **transparent PNG**. Everything runs locally with Python and OpenCV; no API key is needed.

The default settings reproduce the charcoal/grey, densely grouped engraving style used in development. The SVG and transparent PNG have **no background by default**. An optional dark preview is produced separately.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/andreadada/stylus-effect.git
cd stylus-effect
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Create `input/`, add one or more photographs, then run:

```bash
python photo_to_svg.py
```

Files are saved to `output/`:

- `photo.svg` — editable vector paths, transparent background
- `photo_art_transparent.png` — transparent raster image containing only ink strokes
- `photo_art.png` — ink on a dark preview background
- `photo_transparent.png` and `photo_preview.png` — equivalent filenames retained for compatibility

The folders and their contents are intentionally not committed to Git.

## Examples (terminal commands)

```bash
# Your original v7 command works unchanged (PowerShell: keep it on one line)
python photo_to_svg.py --input input/cornicello.jpg --output output --preset balanced --style accent --accent-local-density 4.2 --outline-style cut --outline-cut-density 1.6 --outline-cut-length 11 --export-transparent --svg-background

# One photo, default style
python photo_to_svg.py --input "photos/pepper.jpg" --output output

# Process all images in a directory
python photo_to_svg.py --input photos --output output

# More cuts inside the same groups; do NOT create more groups
python photo_to_svg.py --input photos/pepper.jpg --accent-local-density 5.5

# Fewer individual groups, but very dense cuts in each
python photo_to_svg.py --input photos/pepper.jpg --accent-clusters 4 --accent-local-density 5

# Sparse, minimal engraving
python photo_to_svg.py --input photos/pepper.jpg --accent-clusters 3 --accent-local-density 1.4 --accent-inner-lines 1

# Rougher and more fragmented pencil perimeter
python photo_to_svg.py --input photos/pepper.jpg --outline-cut-density 2 --outline-jitter .35

# Clean uniform outline, but keep grouped interior cuts
python photo_to_svg.py --input photos/pepper.jpg --outline-style smooth

# Custom ink and dark preview color (transparent SVG/PNG are unaffected by the background)
python photo_to_svg.py --input photos/pepper.jpg --stroke "#B1A38A" --background "#141414"

# Add an actual background rectangle to the SVG only
python photo_to_svg.py --input photos/pepper.jpg --svg-background

# Disable exporting the transparent PNG when you only need the SVG and preview
python photo_to_svg.py --input photos/pepper.jpg --no-export-transparent

# Detailed object, higher processing resolution
python photo_to_svg.py --input photos/object.jpg --max-side 2400 --preset detailed

# Choose a particular disconnected object (by descending foreground area)
python photo_to_svg.py --input photos/objects.jpg --region 2

# Complex photo: isolate center object with GrabCut
python photo_to_svg.py --input photos/object.jpg --segment grabcut

# Use a manual mask: white = object, black = background
python photo_to_svg.py --input photos/object.jpg --mask masks/object.png

# Inspect the segmentation and tonal maps
python photo_to_svg.py --input photos/object.jpg --debug

# Print every CLI option and its default
python photo_to_svg.py --help
```

### Optional background removal (ML)

For complicated photos, optionally install `rembg` and an ONNX runtime. The first run might download model weights. Neither weights nor example images belong in this repository.

```bash
python -m pip install rembg onnxruntime
python photo_to_svg.py --input photos/object.jpg --segment rembg
```

Segmentation options: `auto` (alpha if present, otherwise uniform-white detection, otherwise GrabCut), `white`, `grabcut`, `rembg`, `alpha`, and `none`. For difficult scenes, the most reliable input is a manually prepared mask. Generic foreground extraction is not guaranteed to be accurate on every photograph.

## GUI

```bash
python photo_to_svg_gui.py
```

The GUI automatically reads **all available flags, defaults, types, groups, and choices** from the same `build_parser()` function used by the CLI. It provides sliders for commonly adjusted numeric settings, file pickers, a command preview, a Run button, log output, and the generated visual preview. New CLI options automatically appear in the GUI as fields; slider ranges are optional GUI-only metadata.

## Project layout

```text
stylus-effect/
├── photo_to_svg.py          # CLI, segmentation, engraving, SVG/PNG export
├── photo_to_svg_gui.py      # Tkinter GUI generated from the CLI parser
├── requirements.txt
├── .gitignore               # Excludes source photos and generated images
├── .github/workflows/tests.yml
└── tests/
    └── test_photo_to_svg.py # Synthetic, generated-in-memory test images
```

## How it works

1. **Segmentation:** alpha, white-background detection, GrabCut, optional rembg or manual mask.
2. **Region selection:** chooses the largest object, all objects, or an indexed connected region.
3. **Contours:** creates an editable silhouette with fragmented pencil-like strokes.
4. **Tonal analysis:** uses local contrast, brightness and gradients to select a limited number of accent-group positions.
5. **Engraving:** generates dense but localized short cuts along the object's curved outline; changing `--accent-local-density` affects marks inside chosen groups, not the number of group anchors.
6. **Output:** independent SVG groups and alpha-correct transparent PNG, plus optional preview and debug files.

This is a deterministic, style-driven renderer — not a generative illustration model. It works best on isolated product photos, ornaments, fruit, and simple forms. Complex textures or multiple overlapping objects may require a mask or manual review.

## Tests

```bash
python -m unittest discover -s tests -v
```