#!/usr/bin/env python3
"""Stylus Effect: deterministic photographic pencil engraving, exported as SVG and RGBA PNG.

The OpenCV engine works offline. For challenging backgrounds, use --segment rembg
(an optional dependency) or supply a hand-drawn --mask.
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Artwork:
    width: int
    height: int
    # layer name, polylines, width, opacity, closed
    layers: list[tuple[str, list[np.ndarray], float, float, bool]]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Photo to stylized engraving (SVG + transparent PNG)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("Input / output")
    io.add_argument("--input", "-i", type=Path, default=Path("input"), help="Photo or directory")
    io.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory")
    io.add_argument("--max-side", type=int, default=1400, help="Maximum processing size; 0 disables resizing")
    io.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True, help="Charcoal-background preview")
    io.add_argument("--debug", action="store_true", help="Save segmentation and tonal debug images")

    seg = p.add_argument_group("Segmentation")
    seg.add_argument("--segment", choices=["auto", "white", "grabcut", "rembg", "alpha", "none"],
                     default="auto", help="Auto: alpha -> white background -> GrabCut")
    seg.add_argument("--mask", type=Path, default=None, help="Optional explicit binary mask (white foreground)")
    seg.add_argument("--region", default="largest", help="largest, all or connected-component number (1,2,...)")
    seg.add_argument("--grabcut-margin", type=float, default=.07)

    style = p.add_argument_group("Engraving")
    style.add_argument("--style", choices=["accent", "classic"], default="accent")
    style.add_argument("--preset", choices=["minimal", "balanced", "detailed"], default="balanced")
    style.add_argument("--accent-clusters", type=int, default=6, help="Separate groups of accent cuts")
    style.add_argument("--accent-density", type=float, default=.35, help="How many groups are selected (0..2)")
    style.add_argument("--accent-local-density", type=float, default=4.2,
                       help="Number of cuts WITHIN each selected group (1..6), not more regions")
    style.add_argument("--accent-spacing", type=float, default=17., help="Minimum gap between group centers")
    style.add_argument("--accent-length", type=float, default=9., help="Average cut length in pixels")
    style.add_argument("--accent-jitter", type=float, default=.17, help="Natural variation of cut position (0..1)")
    style.add_argument("--accent-inner-lines", type=int, default=2, help="Maximum long internal form lines (0..4)")
    style.add_argument("--seed", type=int, default=42, help="Repeatable engraving layout")

    outline = p.add_argument_group("Pencil outline")
    outline.add_argument("--outline-style", choices=["cut", "smooth"], default="cut")
    outline.add_argument("--outline-cut-density", type=float, default=1.6)
    outline.add_argument("--outline-cut-length", type=float, default=11.)
    outline.add_argument("--outline-jitter", type=float, default=.18)
    outline.add_argument("--stroke", default="#798388", help="SVG ink color, #RRGGBB")
    outline.add_argument("--stroke-width", type=float, default=1.2)
    outline.add_argument("--background", default="#181818", help="Preview background only")
    outline.add_argument("--svg-background", action="store_true", help="Embed opaque background in the SVG")
    return p


def validate(ns: argparse.Namespace) -> None:
    import re
    for c in (ns.stroke, ns.background):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", c):
            raise ValueError("Colors must use #RRGGBB")
    if ns.max_side < 0 or ns.stroke_width <= 0:
        raise ValueError("--max-side must be >= 0 and --stroke-width > 0")
    if not .02 <= ns.grabcut_margin <= .4:
        raise ValueError("--grabcut-margin must be between .02 and .4")
    if not (1 <= ns.accent_clusters <= 20 and 0 <= ns.accent_density <= 2):
        raise ValueError("--accent-clusters: 1..20; --accent-density: 0..2")
    if not (1 <= ns.accent_local_density <= 6 and 5 <= ns.accent_spacing <= 100):
        raise ValueError("--accent-local-density: 1..6; --accent-spacing: 5..100")
    if not (3 <= ns.accent_length <= 70 and 0 <= ns.accent_jitter <= 1):
        raise ValueError("--accent-length: 3..70; --accent-jitter: 0..1")
    if not (0 <= ns.accent_inner_lines <= 4 and .2 <= ns.outline_cut_density <= 4):
        raise ValueError("--accent-inner-lines: 0..4; --outline-cut-density: .2..4")
    if not (3 <= ns.outline_cut_length <= 80 and 0 <= ns.outline_jitter <= 1):
        raise ValueError("--outline-cut-length: 3..80; --outline-jitter: 0..1")
    if ns.region not in ("largest", "all") and (not ns.region.isdigit() or int(ns.region) < 1):
        raise ValueError("--region must be largest, all or positive index")


def read_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def write_png(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Cannot save {path}")
    buffer.tofile(str(path))


def resize(image: np.ndarray, maximum: int) -> np.ndarray:
    h, w = image.shape[:2]
    if maximum == 0 or max(h, w) <= maximum:
        return image
    r = maximum / max(h, w)
    return cv2.resize(image, (max(1, round(w*r)), max(1, round(h*r))), interpolation=cv2.INTER_AREA)


def _white_mask(bgr: np.ndarray, force: bool) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    b = max(3, min(h, w) // 24)
    corners = np.concatenate([bgr[:b, :b].reshape(-1, 3), bgr[-b:, :b].reshape(-1, 3),
                              bgr[:b, -b:].reshape(-1, 3), bgr[-b:, -b:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    if not force and (np.min(bg) < 205 or np.max(np.std(corners, axis=0)) > 20):
        return None
    delta = np.max(np.abs(bgr.astype(np.float32) - bg), axis=2)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    binary = np.uint8((delta > 30) | ((hsv[:, :, 1] > 40) & (hsv[:, :, 2] < 245))) * 255
    return binary


def _grabcut_mask(bgr: np.ndarray, margin: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    mx, my = max(1, round(w*margin)), max(1, round(h*margin))
    if w - 2*mx < 2 or h - 2*my < 2:
        raise ValueError("Image too small for GrabCut")
    labels = np.zeros((h, w), np.uint8)
    cv2.grabCut(bgr, labels, (mx, my, w-2*mx, h-2*my),
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                5, cv2.GC_INIT_WITH_RECT)
    return np.uint8((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)) * 255


def _rembg_mask(bgr: np.ndarray) -> np.ndarray:
    try:
        from rembg import remove
    except ImportError as exc:
        raise ValueError("Install optional segmentation: pip install rembg onnxruntime") from exc
    png = cv2.imencode(".png", bgr)[1].tobytes()
    output = remove(png)
    rgba = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rembg did not return RGBA output")
    return np.uint8(rgba[:, :, 3] > 100) * 255


def select_region(binary: np.ndarray, region: str) -> np.ndarray:
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
    minimum = max(24, int(mask.size*.00015))
    ranked = sorted((i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum),
                    key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    if not ranked:
        raise ValueError("No foreground object detected. Try --segment rembg or --mask.")
    if region == "largest":
        ranked = ranked[:1]
    elif region != "all":
        index = int(region) - 1
        if index >= len(ranked):
            raise ValueError(f"Region {region} unavailable")
        ranked = [ranked[index]]
    return np.uint8(np.isin(labels, ranked)) * 255


def make_mask(img: np.ndarray, ns: argparse.Namespace) -> np.ndarray:
    bgr = img[:, :, :3]
    h, w = bgr.shape[:2]
    if ns.mask:
        if not ns.mask.is_file():
            raise ValueError(f"Mask does not exist: {ns.mask}")
        binary = cv2.imread(str(ns.mask), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            raise ValueError(f"Cannot read mask: {ns.mask}")
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = np.uint8(binary > 127) * 255
    elif ns.segment == "none":
        binary = np.full((h, w), 255, np.uint8)
    elif ns.segment in ("alpha", "auto") and img.shape[2] == 4 and np.any(img[:, :, 3] < 250):
        binary = np.uint8(img[:, :, 3] > 100) * 255
    elif ns.segment == "alpha":
        raise ValueError("No alpha channel detected")
    elif ns.segment == "rembg":
        binary = _rembg_mask(bgr)
    elif ns.segment == "grabcut":
        binary = _grabcut_mask(bgr, ns.grabcut_margin)
    else:
        binary = _white_mask(bgr, force=(ns.segment == "white"))
        if binary is None or cv2.countNonZero(binary) < max(25, int(h*w*.0003)):
            if ns.segment == "white":
                raise ValueError("Unable to isolate foreground on white background")
            binary = _grabcut_mask(bgr, ns.grabcut_margin)
    return select_region(binary, ns.region)


def _tone_maps(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    local = cv2.GaussianBlur(value, (0, 0), sigmaX=12)
    contrast = value - local
    inside = value[mask > 0]
    lo, hi = np.percentile(inside, [10, 90])
    span = max(18., float(hi-lo))
    dark = np.clip((hi-value)/span, 0, 1) *.7 + np.clip(-contrast/32., 0, 1)*.3
    light = np.clip(contrast/23., 0, 1)
    smooth = cv2.GaussianBlur(value, (0, 0), 1.2)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    strength = np.clip(np.hypot(gx, gy)/100., 0, 1)
    return dark, light, strength


def _interior_details(bgr: np.ndarray, mask: np.ndarray, count: int) -> list[np.ndarray]:
    if count <= 0:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(gray)
    e = cv2.Canny(cv2.GaussianBlur(enhanced, (5, 5), 0), 38, 105)
    safe = cv2.erode(mask, np.ones((9, 9), np.uint8))
    e = cv2.bitwise_and(e, safe)
    contours, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ranked = []
    for contour in contours:
        length = cv2.arcLength(contour, False)
        if length < 22:
            continue
        pts = cv2.approxPolyDP(contour, 1.2, False).reshape(-1, 2).astype(np.float32)
        if len(pts) > 2:
            ranked.append((length, pts))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    result = []
    for _, pts in ranked:
        if len(result) >= count:
            break
        if any(np.linalg.norm(pts.mean(axis=0)-line.mean(axis=0)) < 9 for line in result):
            continue
        result.append(pts)
    return result


def _inside(mask: np.ndarray, point: np.ndarray) -> bool:
    x, y = np.rint(point).astype(int)
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0


def _outline(mask: np.ndarray, ns: argparse.Namespace, rng: np.random.Generator):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    base, pencil = [], []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        if len(points) < 12:
            continue
        simple = cv2.approxPolyDP(contour, .75, True).reshape(-1, 2).astype(np.float32)
        base.append(simple)
        if ns.outline_style != "cut":
            continue
        gap = max(3.5, ns.outline_cut_length/ns.outline_cut_density)
        # CHAIN_APPROX_NONE follows the real curved silhouette at near-pixel steps.
        used = 0
        while used < len(points):
            size = max(3, int(ns.outline_cut_length * rng.uniform(.7, 1.3)))
            if rng.random() < .8:
                ids = np.arange(used, used+size, max(1, size//7)) % len(points)
                pts = points[ids].copy()
                normal = np.zeros_like(pts)
                normal[:, 0] = -(np.roll(pts[:, 1], -1)-np.roll(pts[:, 1], 1))
                normal[:, 1] = np.roll(pts[:, 0], -1)-np.roll(pts[:, 0], 1)
                normal /= np.maximum(1e-6, np.linalg.norm(normal, axis=1, keepdims=True))
                pts += normal * rng.uniform(-1, 1, (len(pts), 1)) * ns.outline_jitter * 2.3
                if len(pts) > 1:
                    pencil.append(pts)
            used += max(3, int(gap))
    return base, pencil


def _accent_marks(bgr: np.ndarray, mask: np.ndarray, ns: argparse.Namespace,
                  rng: np.random.Generator) -> list[np.ndarray]:
    if ns.style != "accent" or ns.accent_density == 0:
        return []
    dark, light, gradient = _tone_maps(bgr, mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    safe = cv2.erode(mask, np.ones((3, 3), np.uint8))
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 35:
            continue
        for i in range(0, len(pts), 4):
            tan = pts[(i+8) % len(pts)]-pts[(i-8) % len(pts)]
            tan /= max(1e-6, float(np.linalg.norm(tan)))
            n = np.array([-tan[1], tan[0]], np.float32)
            inward = n if _inside(mask, pts[i] + n*7) else -n
            center = pts[i] + inward*7
            if not _inside(safe, center):
                continue
            x, y = np.rint(center).astype(int)
            score = .15 + 1.2*dark[y, x] + .22*light[y, x] + .23*gradient[y, x]
            candidates.append((score, pts[i], tan, inward))
    if not candidates:
        return []
    n = min(20, max(1, round(ns.accent_clusters * ns.accent_density / .35)))
    weights = np.array([c[0]**2 for c in candidates], np.float64)
    selected = []
    anchors = []
    for _ in range(n):
        if not np.any(weights > 0):
            break
        ix = int(rng.choice(len(candidates), p=weights/weights.sum()))
        anchor = candidates[ix]
        anchors.append(anchor)
        weights[ix] = 0
        for j, other in enumerate(candidates):
            if np.linalg.norm(anchor[1]-other[1]) < ns.accent_spacing * 2.15:
                weights[j] = 0
    # No extra anchor positions are selected when changing --accent-local-density.
    for score, p, tan, inward in anchors:
        count = max(2, round(4 * ns.accent_local_density))
        for k in range(count):
            u = (k+.5)/count - .5
            spread = u * ns.accent_spacing * 1.55
            depth = 5 + ((k*7) % 9) + rng.uniform(-1.1, 1.1)
            center = p + tan*spread + inward*depth
            center += tan*rng.normal(0, ns.accent_jitter*1.1)
            if not _inside(safe, center):
                continue
            vec = tan*.67 + inward*.74
            vec /= max(1e-5, float(np.linalg.norm(vec)))
            length = ns.accent_length*rng.uniform(.72, 1.22)
            a, b = center - vec*length*.5, center + vec*length*.5
            if _inside(safe, a) and _inside(safe, b):
                selected.append(np.array([a, b], np.float32))
    return selected


def render_artwork(image: np.ndarray, ns: argparse.Namespace) -> tuple[Artwork, np.ndarray]:
    mask = make_mask(image, ns)
    bgr = image[:, :, :3]
    h, w = bgr.shape[:2]
    rng = np.random.default_rng(ns.seed)
    contour, pencil = _outline(mask, ns, rng)
    details = _interior_details(bgr, mask, ns.accent_inner_lines)
    accents = _accent_marks(bgr, mask, ns, rng)
    layers = [
        ("outline-base", contour, ns.stroke_width, .25 if pencil else .92, True),
        ("outline-pencil", pencil, ns.stroke_width*.9, .84, False),
        ("form-lines", details, .7, .48, False),
        ("accent-cuts", accents, .87, .9, False),
    ]
    if ns.style == "classic":
        layers[-1] = ("accent-cuts", [], .87, .9, False)
    return Artwork(w, h, layers), mask


def _path(points: np.ndarray, closed: bool) -> str:
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points) + (" Z" if closed else "")


def export_svg(path: Path, art: Artwork, ns: argparse.Namespace) -> None:
    ns_uri = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns_uri)
    root = ET.Element(f"{{{ns_uri}}}svg", {"viewBox": f"0 0 {art.width} {art.height}",
                                        "width": str(art.width), "height": str(art.height)})
    ET.SubElement(root, "title").text = "Stylus Effect — pencil engraving"
    if ns.svg_background:
        ET.SubElement(root, "rect", {"width": str(art.width), "height": str(art.height), "fill": ns.background})
    for label, lines, width, opacity, closed in art.layers:
        if not lines:
            continue
        group = ET.SubElement(root, "g", {"id": label, "fill": "none", "stroke": ns.stroke,
                                          "stroke-width": str(width), "stroke-opacity": str(opacity),
                                          "stroke-linecap": "round", "stroke-linejoin": "round"})
        for pts in lines:
            ET.SubElement(group, "path", {"d": _path(pts, closed)})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def transparent_png(art: Artwork, ns: argparse.Namespace) -> np.ndarray:
    # Rasterize paths on a 3x supersampled alpha channel for soft pencil edges.
    scale = 3
    h, w = art.height, art.width
    opacity_map = np.zeros((h*scale, w*scale), np.float32)
    for _, paths, width, opacity, closed in art.layers:
        if not paths:
            continue
        layer = np.zeros((h*scale, w*scale), np.uint8)
        for pts in paths:
            curve = np.rint(pts*scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(layer, [curve], closed, 255,
                          max(1, round(width*scale)), cv2.LINE_AA)
        a = layer.astype(np.float32) / 255. * opacity
        opacity_map = 1-(1-opacity_map)*(1-a)
    alpha = cv2.resize(opacity_map, (w, h), interpolation=cv2.INTER_AREA)
    rgb = bytes.fromhex(ns.stroke[1:])
    image = np.zeros((h, w, 4), np.uint8)
    image[:, :, :3] = (rgb[2], rgb[1], rgb[0])
    image[:, :, 3] = np.uint8(np.clip(np.round(alpha*255), 0, 255))
    return image


def process(path: Path, ns: argparse.Namespace) -> dict:
    image = resize(read_image(path), ns.max_side)
    art, mask = render_artwork(image, ns)
    ns.output.mkdir(parents=True, exist_ok=True)
    svg_path = ns.output / f"{path.stem}.svg"
    png_path = ns.output / f"{path.stem}_transparent.png"
    export_svg(svg_path, art, ns)
    rgba = transparent_png(art, ns)
    write_png(png_path, rgba)
    if ns.preview:
        color = bytes.fromhex(ns.background[1:])
        bg = np.full((art.height, art.width, 3), (color[2], color[1], color[0]), np.uint8)
        a = rgba[:, :, 3:4].astype(np.float32)/255.
        composite = np.uint8(np.round(rgba[:, :, :3]*a + bg*(1-a)))
        write_png(ns.output / f"{path.stem}_preview.png", composite)
    if ns.debug:
        write_png(ns.output / f"{path.stem}_mask.png", mask)
        dark, light, _ = _tone_maps(image[:, :, :3], mask)
        write_png(ns.output / f"{path.stem}_tone.png", np.uint8(np.clip(dark*.8+light*.2, 0, 1)*255))
    return {"svg": svg_path, "png": png_path,
            "cuts": len(art.layers[-1][1]), "outline": len(art.layers[0][1])}


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        validate(ns)
        if ns.input.is_file():
            files = [ns.input]
        elif ns.input.is_dir():
            files = sorted(p for p in ns.input.iterdir() if p.suffix.lower() in EXTENSIONS)
        else:
            raise ValueError(f"Input not found: {ns.input}. Create the folder or use --input FILE")
        if not files:
            raise ValueError(f"No supported images found in {ns.input}")
        if ns.mask and len(files) != 1:
            raise ValueError("--mask accepts one input image at a time")
        failed = 0
        for file in files:
            try:
                result = process(file, ns)
                print(f"[OK] {file.name}: {result['svg']} + {result['png']} "
                      f"(outline={result['outline']}, cuts={result['cuts']})")
            except (ValueError, cv2.error, OSError) as exc:
                failed += 1
                print(f"[ERROR] {file.name}: {exc}", file=sys.stderr)
        return 1 if failed else 0
    except (ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())