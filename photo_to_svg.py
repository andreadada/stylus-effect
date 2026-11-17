#!/usr/bin/env python3
"""O' Spuntin — fotografie -> contorni e riflessi interni -> SVG vettoriale.

Versione 7.1: stile accent del peperoncino selezionato come impostazione predefinita.
Solo OpenCV + NumPy; nessun servizio remoto, nessun modello ML richiesto.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PRESETS = {
    # Canny outer thresholds, Gaussian blur, min perimeter, Douglas-Peucker, max paths;
    # detail / highlight passes have lower thresholds to capture glossy reflections.
    "minimal": dict(low=75, high=160, blur=7, min_length=90., epsilon=2.3,
                    max_paths=110, detail_low=32, detail_high=95, detail_min_length=32.,
                    detail_epsilon=1.8, detail_max_paths=110, highlight_threshold=16),
    "balanced": dict(low=60, high=145, blur=5, min_length=55., epsilon=1.6,
                     max_paths=220, detail_low=18, detail_high=64, detail_min_length=15.,
                     detail_epsilon=0.9, detail_max_paths=230, highlight_threshold=8),
    "detailed": dict(low=40, high=115, blur=3, min_length=25., epsilon=0.9,
                     max_paths=400, detail_low=12, detail_high=42, detail_min_length=9.,
                     detail_epsilon=0.5, detail_max_paths=460, highlight_threshold=7),
}


@dataclass(frozen=True)
class Settings:
    low: int
    high: int
    blur: int
    min_length: float
    epsilon: float
    max_paths: int
    detail_low: int
    detail_high: int
    detail_min_length: float
    detail_epsilon: float
    detail_max_paths: int
    highlight_threshold: int


@dataclass
class Layers:
    outer: list[np.ndarray]
    details: list[np.ndarray]
    highlights: list[np.ndarray]
    hatch: list[np.ndarray]
    hatch_light: list[np.ndarray]
    hatch_region: np.ndarray | None
    mask: np.ndarray | None
    outer_closed: bool
    outer_edges: np.ndarray
    details_edges: np.ndarray
    highlights_edges: np.ndarray
    accent_marks: list[np.ndarray] = field(default_factory=list)
    accent_form: list[np.ndarray] = field(default_factory=list)
    outer_texture: list[np.ndarray] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="O' Spuntin: foto -> contorni / riflessi / incisione SVG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", "-i", type=Path, default=Path("input"), help="Immagine o cartella")
    ap.add_argument("--output", "-o", type=Path, default=Path("output"), help="Cartella di uscita")
    ap.add_argument("--preset", choices=PRESETS, default="balanced")
    ap.add_argument("--mode", choices=["engraving", "edges", "silhouette"], default="engraving",
                    help="engraving=contorni+riflessi, edges=solo Canny, silhouette=solo sagoma")
    ap.add_argument("--style", choices=["classic", "accent"], default="accent",
                    help="accent=pochi gruppi di incisioni vicino ai bordi e alle linee del volume")
    for name, typ, help_text in [
        ("low", int, "Soglia principale Canny bassa"),
        ("high", int, "Soglia principale Canny alta"),
        ("blur", int, "Kernel Gaussian dispari"),
        ("min-length", float, "Lunghezza minima contorni esterni"),
        ("epsilon", float, "Semplificazione tracciati esterni in px"),
        ("max-paths", int, "Numero massimo tracciati esterni"),
        ("detail-low", int, "Soglia dettagli interni bassa"),
        ("detail-high", int, "Soglia dettagli interni alta"),
        ("detail-min-length", float, "Lunghezza minima linee interne"),
        ("detail-epsilon", float, "Semplificazione linee interne in px"),
        ("detail-max-paths", int, "Numero massimo linee per ogni livello interno"),
        ("highlight-threshold", int, "Soglia riflessi in intensità locale, più bassa = più riflessi"),
    ]:
        ap.add_argument("--" + name, type=typ, default=None, help=help_text)
    ap.add_argument("--max-side", type=int, default=1400,
                    help="Massima dimensione lato; 0 non ridimensiona")
    ap.add_argument("--mask", type=Path, help="Maschera bianca sul soggetto, nera sullo sfondo")
    ap.add_argument("--grabcut", action="store_true", help="Isola il soggetto centrale con GrabCut")
    ap.add_argument("--grabcut-margin", type=float, default=.07)
    ap.add_argument("--auto-mask", choices=["auto", "white", "none"], default="auto",
                    help="auto identifica sfondo bianco dai bordi della foto")
    ap.add_argument("--stroke", default="#82939A", help="Colore delle linee SVG")
    ap.add_argument("--stroke-width", type=float, default=1.2,
                    help="Spessore principale SVG")
    ap.add_argument("--opacity", type=float, default=1., help="Opacità generale 0..1")
    ap.add_argument("--detail-width", type=float, default=.78)
    ap.add_argument("--detail-opacity", type=float, default=.67)
    ap.add_argument("--highlight-width", type=float, default=.75)
    ap.add_argument("--highlight-opacity", type=float, default=.88)
    ap.add_argument("--hatching", choices=["off", "light", "medium", "adaptive"], default="off",
                    help="adaptive=tratteggi curvi che seguono la forma e reagiscono a luci/ombre")
    ap.add_argument("--hatch-tone", choices=["shadows", "highlights", "both"], default="both",
                    help="Zone da incidere, solo in modalità adaptive")
    ap.add_argument("--hatch-region", default="largest",
                    help="Regione chiusa da riempire: largest (una sola), all, o indice 1,2,3... per area")
    ap.add_argument("--hatch-flow", choices=["contour", "diagonal"], default="contour",
                    help="Direzione incisioni: segue la forma o inclinazione costante")
    ap.add_argument("--hatch-angle", type=float, default=38.,
                    help="Inclinazione in gradi rispetto alla tangente alla forma")
    ap.add_argument("--hatch-spacing", type=float, default=12.,
                    help="Distanza base fra i centri dei taglietti (pixel)")
    ap.add_argument("--hatch-length", type=float, default=10.,
                    help="Lunghezza media di ogni tratteggio (pixel)")
    ap.add_argument("--hatch-density", type=float, default=.72,
                    help="Quantità incisioni, 0..3; maggiore = più fitto")
    ap.add_argument("--hatch-width", type=float, default=.82,
                    help="Spessore delle incisioni SVG")
    ap.add_argument("--hatch-opacity", type=float, default=.66,
                    help="Opacità incisioni ombra; luci calibrate separatamente")
    ap.add_argument("--hatch-seed", type=int, default=42,
                    help="Seed per una resa ripetibile")
    ap.add_argument("--accent-region", default="largest",
                    help="Oggetto chiuso da incidere: largest, all oppure numero 1,2,3...")
    ap.add_argument("--accent-follow", choices=["contour", "diagonal"], default="contour",
                    help="Orienta i tagli rispetto alle curve oppure in direzione fissa")
    ap.add_argument("--accent-density", type=float, default=.35,
                    help="Densita dei gruppi di incisioni (0..2, default 0.35)")
    ap.add_argument("--accent-spacing", type=float, default=17.,
                    help="Passo e ampiezza della zona di incisione (non determina nuovi gruppi)")
    ap.add_argument("--accent-local-density", type=float, default=4.2,
                    help="Fittezza dentro i gruppi: 1=leggero, 4.2=stile consigliato, 6=molto fitto")
    ap.add_argument("--accent-length", type=float, default=9.,
                    help="Lunghezza media di un taglio, in pixel")
    ap.add_argument("--accent-jitter", type=float, default=.15,
                    help="Irregolarita naturale dei tagli (0..1)")
    ap.add_argument("--accent-border-weight", type=float, default=.78,
                    help="Preferenza per i segni vicino al contorno (0..1)")
    ap.add_argument("--accent-inner-lines", type=int, default=2,
                    help="Numero massimo di linee di volume interne (0..4)")
    ap.add_argument("--accent-clusters", type=int, default=6,
                    help="Numero base di gruppi lungo il perimetro (1..12)")
    ap.add_argument("--accent-width", type=float, default=.9,
                    help="Spessore dei tagli vettoriali in pixel")
    ap.add_argument("--accent-seed", type=int, default=42,
                    help="Riproducibilita della disposizione")
    ap.add_argument("--outline-style", choices=["smooth", "cut"], default="smooth",
                    help="smooth=contorno uniforme, cut=contorno spezzato effetto matita/incisione")
    ap.add_argument("--outline-cut-density", type=float, default=1.4,
                    help="Quantita di segmenti sul bordo, senza cambiare la forma")
    ap.add_argument("--outline-cut-length", type=float, default=11.,
                    help="Lunghezza media dei segmenti del bordo")
    ap.add_argument("--outline-jitter", type=float, default=.18,
                    help="Irregolarita dei segmenti del bordo (0..1)")
    ap.add_argument("--export-transparent", action=argparse.BooleanOptionalAction, default=False,
                    help="Esporta anche un PNG trasparente con solo i tratti")
    ap.add_argument("--svg-background", action="store_true",
                    help="Inserisce nello SVG un rettangolo con il colore --background")
    ap.add_argument("--background", default="#101C24", help="Sfondo anteprima (non nell'SVG)")
    ap.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--debug", action="store_true", help="Esporta anche mappa Canny e maschera")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = build_parser()
    ns = ap.parse_args(argv)
    actual_args = list(sys.argv[1:] if argv is None else argv)
    def explicit_option(name: str) -> bool:
        return any(arg == "--" + name or arg.startswith("--" + name + "=")
                   or arg == "--no-" + name for arg in actual_args)
    # Modern charcoal engraving palette, without changing any old command's defaults.
    if ns.hatching == "adaptive" or ns.style == "accent":
        if not explicit_option("background"):
            ns.background = "#181818"
        if not explicit_option("stroke"):
            ns.stroke = "#798388"
    if ns.style == "accent":
        if not explicit_option("outline-style"):
            ns.outline_style = "cut"
        if not explicit_option("export-transparent"):
            ns.export_transparent = True
    return ns


def validated(ns: argparse.Namespace) -> Settings:
    defaults = PRESETS[ns.preset]
    s = Settings(**{name: getattr(ns, name) if getattr(ns, name) is not None else value
                    for name, value in defaults.items()})
    for a, b in [(s.low, s.high), (s.detail_low, s.detail_high)]:
        if not 0 <= a < b <= 255:
            raise ValueError("Soglie Canny: servono 0 <= low < high <= 255")
    if s.blur <= 0 or s.blur % 2 == 0:
        raise ValueError("--blur deve essere dispari e positivo")
    if min(s.min_length, s.detail_min_length) <= 0:
        raise ValueError("Le lunghezze minime devono essere positive")
    if min(s.epsilon, s.detail_epsilon) < 0:
        raise ValueError("Gli epsilon non possono essere negativi")
    if min(s.max_paths, s.detail_max_paths) < 1:
        raise ValueError("I massimi di tracciati devono essere >= 1")
    if not 0 <= s.highlight_threshold <= 255:
        raise ValueError("--highlight-threshold deve essere fra 0 e 255")
    if ns.max_side < 0:
        raise ValueError("--max-side deve essere >= 0")
    if ns.stroke_width <= 0 or ns.detail_width <= 0 or ns.highlight_width <= 0:
        raise ValueError("Gli spessori devono essere positivi")
    if any(not 0 <= a <= 1 for a in (ns.opacity, ns.detail_opacity, ns.highlight_opacity)):
        raise ValueError("Le opacità devono essere comprese tra 0 e 1")
    if not .02 <= ns.grabcut_margin <= .40:
        raise ValueError("--grabcut-margin deve essere fra 0.02 e 0.40")
    for color in [ns.stroke, ns.background]:
        if re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None:
            raise ValueError("Usa colori #RRGGBB")
    if not (3 <= ns.hatch_spacing <= 80):
        raise ValueError("--hatch-spacing deve essere fra 3 e 80")
    if not (2 <= ns.hatch_length <= 120):
        raise ValueError("--hatch-length deve essere fra 2 e 120")
    if not 0 <= ns.hatch_density <= 3:
        raise ValueError("--hatch-density deve essere fra 0 e 3")
    if ns.hatch_width <= 0 or not 0 <= ns.hatch_opacity <= 1:
        raise ValueError("--hatch-width deve essere positivo e --hatch-opacity 0..1")
    if not 0 <= ns.accent_density <= 2:
        raise ValueError("--accent-density deve essere fra 0 e 2")
    if not 5 <= ns.accent_spacing <= 100 or not 3 <= ns.accent_length <= 70:
        raise ValueError("--accent-spacing 5..100 e --accent-length 3..70")
    if not 1 <= ns.accent_local_density <= 6:
        raise ValueError("--accent-local-density deve essere fra 1 e 6: 1 mantiene la densita v4")
    if not (0 <= ns.accent_jitter <= 1 and 0 <= ns.accent_border_weight <= 1):
        raise ValueError("--accent-jitter e --accent-border-weight devono essere fra 0 e 1")
    if not 0 <= ns.accent_inner_lines <= 4 or not 1 <= ns.accent_clusters <= 12:
        raise ValueError("--accent-inner-lines 0..4 e --accent-clusters 1..12")
    if ns.accent_width <= 0:
        raise ValueError("--accent-width deve essere positivo")
    if not 0.2 <= ns.outline_cut_density <= 4:
        raise ValueError("--outline-cut-density deve essere fra 0.2 e 4")
    if not 3 <= ns.outline_cut_length <= 80:
        raise ValueError("--outline-cut-length deve essere fra 3 e 80")
    if not 0 <= ns.outline_jitter <= 1:
        raise ValueError("--outline-jitter deve essere fra 0 e 1")
    if ns.accent_region not in ("largest", "all"):
        if not ns.accent_region.isdigit() or int(ns.accent_region) < 1:
            raise ValueError("--accent-region: largest, all oppure indice 1,2,3...")
    if ns.style == "accent" and ns.mode != "engraving":
        raise ValueError("--style accent richiede --mode engraving")
    if ns.style == "accent" and ns.hatching != "off":
        raise ValueError("--style accent non si combina con --hatching; scegli uno dei due")
    if ns.hatch_region not in ("largest", "all"):
        if not ns.hatch_region.isdigit() or int(ns.hatch_region) < 1:
            raise ValueError("--hatch-region: largest, all, oppure indice 1,2,3...")
    if ns.grabcut and ns.mask:
        raise ValueError("Usa --grabcut oppure --mask, non entrambi")
    return s


def get_files(src: Path) -> list[Path]:
    if src.is_file():
        if src.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Estensione non supportata: {src}")
        return [src]
    if src.is_dir():
        files = [f for f in sorted(src.iterdir())
                 if f.is_file() and f.suffix.lower() in EXTENSIONS]
        if not files:
            raise ValueError(f"Nessuna foto nella cartella: {src}")
        return files
    raise FileNotFoundError(f"Input non trovato: {src}")


def read_image(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    # Unicode-friendly on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, mode)
    if img is None:
        raise ValueError(f"Immagine illeggibile: {path}")
    return img


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Errore esportazione: {path}")
    encoded.tofile(str(path))


def resize(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max_side == 0 or max(w, h) <= max_side:
        return image
    k = max_side / max(w, h)
    return cv2.resize(image, (max(1, round(w*k)), max(1, round(h*k))),
                      interpolation=cv2.INTER_AREA)


def white_background_mask(image: np.ndarray, force: bool = False) -> np.ndarray | None:
    """Segment a subject against a plain white/off-white photographic background.

    This is NOT a generic segmentation algorithm. For non-white scenes use a hand
    mask or --grabcut. All disconnected but meaningful object parts are kept.
    """
    h, w = image.shape[:2]
    b = max(4, min(h, w) // 35)
    corners = np.concatenate([image[:b, :b].reshape(-1, 3),
                              image[:b, -b:].reshape(-1, 3),
                              image[-b:, :b].reshape(-1, 3),
                              image[-b:, -b:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    saturation = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_BGR2HSV)[0, 0, 1]
    if not force and (np.min(bg) < 216 or saturation > 36):
        return None
    diff = np.max(np.abs(image.astype(np.int16) - bg.astype(np.int16)), axis=2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    foreground = np.uint8(((diff > 32) | ((hsv[:, :, 1] > 38) & (hsv[:, :, 2] < 245)))) * 255
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, small)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    keep = np.zeros((h, w), np.uint8)
    min_area = max(10, int(h*w * .00012))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    if cv2.countNonZero(keep) < max(15, h*w*.00015):
        return None
    # Fill interiors including dark cavities, retaining small holes only if
    # explicitly provided by a hand-drawn mask.
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(keep, contours, -1, 255, cv2.FILLED)
    return keep


def make_mask(image: np.ndarray, ns: argparse.Namespace) -> np.ndarray | None:
    h, w = image.shape[:2]
    if ns.mask:
        original = read_image(ns.mask, cv2.IMREAD_GRAYSCALE)
        original = cv2.resize(original, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.uint8(original > 127)*255
    if ns.grabcut:
        mx = max(1, round(w*ns.grabcut_margin))
        my = max(1, round(h*ns.grabcut_margin))
        if 2*mx >= w or 2*my >= h:
            raise ValueError("Immagine troppo piccola per GrabCut")
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(image, gc, (mx, my, w - 2*mx, h - 2*my),
                    np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                    5, cv2.GC_INIT_WITH_RECT)
        result = np.uint8((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))*255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(result)
        if n > 1:
            biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            result = np.uint8(labels == biggest)*255
        return result
    if ns.auto_mask != "none":
        return white_background_mask(image, force=ns.auto_mask == "white")
    return None


def as_paths(binary: np.ndarray, min_length: float, epsilon: float,
             max_paths: int, closed: bool = False) -> list[np.ndarray]:
    retrieval = cv2.RETR_EXTERNAL if closed else cv2.RETR_LIST
    contours, _ = cv2.findContours(binary, retrieval, cv2.CHAIN_APPROX_NONE)
    scored: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        length = cv2.arcLength(contour, closed)
        if length < min_length:
            continue
        simplified = cv2.approxPolyDP(contour, epsilon, closed=closed).reshape(-1, 2)
        if len(simplified) < (3 if closed else 2):
            continue
        if len(np.unique(simplified, axis=0)) < (3 if closed else 2):
            continue
        scored.append((length, simplified.astype(np.int32)))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [pts for _, pts in scored[:max_paths]]


def highlight_map(image: np.ndarray, mask: np.ndarray | None,
                  threshold: int) -> np.ndarray:
    # Positive local intensity changes catch narrow specular streaks on red,
    # black or metallic objects much better than the grayscale Canny alone.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    smooth = cv2.GaussianBlur(value, (0, 0), sigmaX=11)
    bright_delta = cv2.subtract(value, smooth)
    bright = np.uint8(bright_delta >= threshold)*255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if mask is not None:
        inside = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        bright = cv2.bitwise_and(bright, inside)
    return bright


def make_hatches(img: np.ndarray, mask: np.ndarray | None, level: str) -> list[np.ndarray]:
    if mask is None or level == "off":
        return []
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    values = gray[eroded > 0]
    if values.size < 30:
        return []
    q = 40 if level == "light" else 60
    cutoff = np.percentile(values, q)
    shadow = np.uint8((gray < cutoff) & (eroded > 0))*255
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    h, w = gray.shape
    spacing = 14 if level == "light" else 8
    lines: list[np.ndarray] = []
    # 45° segments on x - y = const, clipped to shadow regions.
    for d in range(-h, w, spacing):
        a, b = max(0, -d), min(h, w-d)
        if b-a < 12:
            continue
        yy = np.arange(a, b)
        xx = yy + d
        inside = shadow[yy, xx] > 0
        padded = np.pad(inside.astype(np.int8), (1, 1))
        starts = np.where(np.diff(padded) == 1)[0]
        ends = np.where(np.diff(padded) == -1)[0]
        for start, end in zip(starts, ends):
            if end-start < 12:
                continue
            lines.append(np.array([[xx[start], yy[start]],
                                   [xx[end-1], yy[end-1]]], np.int32))
    return lines[:550]



def select_closed_region(mask: np.ndarray, choice: str) -> tuple[np.ndarray, int]:
    """Select exactly one closed connected object by area, unless all is requested.

    Objects touching the border are allowed, but the background itself is excluded
    because selection operates only inside a precomputed foreground mask.
    """
    count, label, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), connectivity=8)
    if count < 2:
        raise ValueError("Nessuna regione chiusa trovata nella maschera")
    rank = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    # Ignore isolated segmentation dust in indexed selection.
    minimum = max(12, round(mask.size * .00006))
    rank = [i for i in rank if stats[i, cv2.CC_STAT_AREA] >= minimum]
    if not rank:
        raise ValueError("Nessuna regione sufficientemente grande")
    if choice == "all":
        selected = rank
    elif choice == "largest":
        selected = rank[:1]
    else:
        idx = int(choice) - 1
        if idx >= len(rank):
            raise ValueError(f"Regione {choice} inesistente: rilevate {len(rank)} regioni")
        selected = [rank[idx]]
    result = np.uint8(np.isin(label, selected)) * 255
    return result, len(rank)


def vector_field_for_hatches(mask: np.ndarray, ns: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Smooth tangent field from distance to the outer boundary of the object.

    The tangent is signed consistently along the object's principal axis; the
    desired diagonal engraving angle is then rotated relative to that tangent.
    """
    h, w = mask.shape
    if ns.hatch_flow == "diagonal":
        a = np.deg2rad(ns.hatch_angle)
        return (np.full((h, w), np.cos(a), np.float32),
                np.full((h, w), np.sin(a), np.float32))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sigma = max(2.5, ns.hatch_spacing * 1.0)
    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=sigma)
    dy, dx = np.gradient(dist)
    tx, ty = -dy, dx
    coords = np.column_stack(np.nonzero(mask > 0)).astype(np.float32)
    if len(coords) >= 3:
        covariance = np.cov(coords.T)
        evals, eigvecs = np.linalg.eigh(covariance)
        # coords are y,x; use x,y for the global reference vector.
        axis_y, axis_x = eigvecs[:, np.argmax(evals)]
        if axis_y < 0 or (abs(axis_y) < 1e-4 and axis_x < 0):
            axis_y, axis_x = -axis_y, -axis_x
    else:
        axis_x, axis_y = 0., 1.
    # Fallback for the medial axis, where the distance gradient is near zero.
    mag = np.hypot(tx, ty)
    tx = np.where(mag > 0.025, tx / np.maximum(mag, .001), axis_x)
    ty = np.where(mag > 0.025, ty / np.maximum(mag, .001), axis_y)
    reversal = (tx * axis_x + ty * axis_y) < 0
    tx = np.where(reversal, -tx, tx)
    ty = np.where(reversal, -ty, ty)
    # Smooth orientation without canceling opposite-pointing field vectors.
    tx = cv2.GaussianBlur(tx.astype(np.float32), (0, 0), sigmaX=max(1., sigma * .4))
    ty = cv2.GaussianBlur(ty.astype(np.float32), (0, 0), sigmaX=max(1., sigma * .4))
    mag = np.maximum(np.hypot(tx, ty), 1e-5)
    tx /= mag
    ty /= mag
    a = np.deg2rad(ns.hatch_angle)
    ca, sa = np.cos(a), np.sin(a)
    return ((tx * ca - ty * sa).astype(np.float32),
            (tx * sa + ty * ca).astype(np.float32))


def tonal_intensity(image: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized light and dark tonal maps, independent of object hue.

    Brightness is partly measured in HSV Value, so a glossy saturated-red object
    retains its shape even though a grayscale conversion would make it dark.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = .76 * hsv[:, :, 2].astype(np.float32) + .24 * gray.astype(np.float32)
    value = cv2.GaussianBlur(value, (0, 0), sigmaX=2.2)
    values = value[region > 0]
    if values.size < 20:
        return np.zeros_like(value), np.zeros_like(value)
    p10, p45, p68, p96 = np.percentile(values, [10, 45, 68, 96])
    shadow = np.clip((p68 - value) / max(8., p68 - p10), 0., 1.)
    light = np.clip((value - p45) / max(9., p96 - p45), 0., 1.)
    # Local gloss enhancement: light streaks against an otherwise uniform face.
    background = cv2.GaussianBlur(value, (0, 0), sigmaX=9.)
    bright_relief = np.clip((value - background - 3.) / 23., 0., 1.)
    light = np.maximum(light * .77, bright_relief)
    return shadow.astype(np.float32), light.astype(np.float32)


def make_adaptive_hatches(image: np.ndarray, mask: np.ndarray,
                          ns: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Short curved SVG cuts that follow geometry, clipped to one chosen region.

    Separate tone passes are used so brighter highlights are sparse and shadows
    visibly denser. No bitmap pattern is used in the generated vector SVG.
    """
    selected, _ = select_closed_region(mask, ns.hatch_region)
    shadow, light = tonal_intensity(image, selected)
    dx, dy = vector_field_for_hatches(selected, ns)
    h, w = selected.shape
    # Inset 2px keeps all vector strokes safely within the selected silhouette.
    safe = cv2.erode(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ys, xs = np.where(safe > 0)
    if len(xs) == 0:
        return [], [], selected
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    rng = np.random.default_rng(ns.hatch_seed)
    spacing = ns.hatch_spacing
    # Staggered centers, with jitter that keeps the marks hand-engraved but ordered.
    cuts_dark: list[np.ndarray] = []
    cuts_light: list[np.ndarray] = []
    max_marks = 18000  # protection for large images and extremely dense settings

    def field_at(x: float, y: float) -> tuple[float, float]:
        xi = int(np.clip(round(x), 0, w-1))
        yi = int(np.clip(round(y), 0, h-1))
        return float(dx[yi, xi]), float(dy[yi, xi])

    def trace(cx: float, cy: float, length: float) -> np.ndarray | None:
        # Integrate both directions to follow the curved tangent instead of
        # drawing a straight, global diagonal line.
        half = length * .5
        step = 1.25
        nsteps = max(2, int(np.ceil(half / step)))
        traces: list[list[tuple[float, float]]] = []
        for sign in (-1., 1.):
            x, y = cx, cy
            positions = []
            for _ in range(nsteps):
                vx, vy = field_at(x, y)
                nx, ny = x + sign * vx * step, y + sign * vy * step
                if not (1 <= nx < w - 1 and 1 <= ny < h - 1):
                    break
                if safe[int(round(ny)), int(round(nx))] == 0:
                    break
                positions.append((nx, ny))
                x, y = nx, ny
            traces.append(positions)
        points = traces[0][::-1] + [(cx, cy)] + traces[1]
        if len(points) < 3:
            return None
        pts = np.array(points, dtype=np.float32)
        if float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) < 3.:
            return None
        # Store fractional coordinates (to_path handles these precisely).
        return pts[::2] if len(pts) > 4 else pts

    rows = np.arange(min_y, max_y + 1, spacing)
    cols = np.arange(min_x, max_x + 1, spacing)
    for ri, y in enumerate(rows):
        for x in cols:
            if len(cuts_dark) + len(cuts_light) >= max_marks:
                return cuts_dark, cuts_light, selected
            cx = x + (spacing * .5 if ri % 2 else 0) + rng.uniform(-.32, .32) * spacing
            cy = y + rng.uniform(-.33, .33) * spacing
            xi, yi = round(cx), round(cy)
            if not (0 <= xi < w and 0 <= yi < h) or safe[yi, xi] == 0:
                continue
            s = float(shadow[yi, xi]); l = float(light[yi, xi])
            # Avoid filling neutral areas uniformly. Shadows get dense engraving;
            # highlights get occasional thinner accents on actual bright ridges.
            p_dark = min(.96, (0.04 + .95 * s ** 1.35) * ns.hatch_density)
            p_light = min(.7, (.03 + .33 * l ** 2.1) * ns.hatch_density)
            if ns.hatch_tone in ("shadows", "both") and rng.random() < p_dark:
                length = ns.hatch_length * (.67 + .76 * s) * rng.uniform(.76, 1.19)
                cut = trace(cx, cy, length)
                if cut is not None:
                    cuts_dark.append(cut)
            elif ns.hatch_tone in ("highlights", "both") and rng.random() < p_light:
                length = ns.hatch_length * (.52 + .33 * l) * rng.uniform(.76, 1.15)
                cut = trace(cx, cy, length)
                if cut is not None:
                    cuts_light.append(cut)
    return cuts_dark, cuts_light, selected




def _inside(region: np.ndarray, x: float, y: float) -> bool:
    h, w = region.shape
    ix, iy = int(round(x)), int(round(y))
    return 0 <= ix < w and 0 <= iy < h and bool(region[iy, ix])


def _clip_segment(center: np.ndarray, direction: np.ndarray, length: float,
                  safe: np.ndarray) -> np.ndarray | None:
    """Clips a small vector stroke to the selected interior; never draws outside."""
    n = max(8, int(np.ceil(length * 2)))
    ts = np.linspace(-length / 2, length / 2, n)
    pts = center[None, :] + ts[:, None] * direction[None, :]
    valid = np.array([_inside(safe, float(x), float(y)) for x, y in pts])
    change = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    if len(starts) == 0:
        return None
    start, end = max(zip(starts, ends), key=lambda x: x[1] - x[0])
    result = pts[start:end]
    if len(result) < 3 or float(np.linalg.norm(result[-1] - result[0])) < 3.:
        return None
    # Small hand-drawn bend; all samples remain inside the region.
    return np.asarray(result[::max(1, len(result) // 5)], np.float32) if len(result) > 2 else None


def _feature_spines(detail_paths: list[np.ndarray], region: np.ndarray,
                    max_lines: int) -> list[np.ndarray]:
    """Reduce noisy Canny contour loops to a few smooth medial visual axes.

    We bin long, narrow edge contours along their PCA major axis to get ONE
    centered guiding line, rather than tracing both sides of the same reflection.
    """
    if max_lines <= 0:
        return []
    distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    candidates: list[tuple[float, np.ndarray]] = []
    for path in detail_paths:
        xy = np.asarray(path, np.float32)
        if len(xy) < 8:
            continue
        origin = xy.mean(axis=0)
        axes = xy - origin
        try:
            _u, _s, vt = np.linalg.svd(axes, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        major = vt[0]
        t = axes @ major
        transverse = axes @ vt[1]
        extent = float(np.ptp(t))
        thickness = float(np.percentile(transverse, 90) - np.percentile(transverse, 10))
        if extent < 35 or extent < 2.2 * max(3, thickness):
            continue
        step = 3.5
        ts = np.arange(float(t.min()), float(t.max())+step, step)
        centers = []
        for start in ts:
            within = (t >= start) & (t < start+step)
            if within.sum() < 1:
                continue
            point = np.median(xy[within], axis=0)
            if _inside(region, float(point[0]), float(point[1])):
                centers.append(point)
        if len(centers) < 9:
            continue
        line = np.asarray(centers, np.float32)
        if len(line) >= 5:
            # Edge padding prevents line endpoints from shrinking.
            for i in (0, 1):
                v = np.pad(line[:, i], (2, 2), mode="edge")
                line[:, i] = np.convolve(v, np.array([1, 2, 3, 2, 1]) / 9., mode="valid")
        # Exclude visual contours right next to the outer silhouette.
        cx = np.clip(np.rint(line[:, 0]).astype(int), 0, region.shape[1]-1)
        cy = np.clip(np.rint(line[:, 1]).astype(int), 0, region.shape[0]-1)
        medial = float(np.median(distance[cy, cx]))
        if medial < 4.:
            continue
        coverage = np.mean([_inside(region, float(x), float(y)) for x, y in line])
        if coverage < .89:
            continue
        score = extent * min(3.5, extent / max(4., thickness))**.4 * min(1.5, medial/9.)
        candidates.append((score, line))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[np.ndarray] = []
    for _, line in candidates:
        if len(chosen) >= max_lines:
            break
        # Avoid returning both edges of the same specular streak.
        too_similar = False
        for existing in chosen:
            a = line[::max(1, len(line)//35)]
            b = existing[::max(1, len(existing)//35)]
            dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            if float(np.mean(dist.min(axis=1) < 10.)) > .42:
                too_similar = True
                break
        if not too_similar:
            chosen.append(line)
    return chosen


def make_accent_strokes(image: np.ndarray, mask: np.ndarray,
                        ns: argparse.Namespace,
                        detail_paths: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Sparse, clustered hand-engraved accents instead of texture fill.

    1) Use the closed foreground component(s) chosen by the user.
    2) Score positions near the boundary using local shadow and edge contrast.
    3) Pick only a handful of separated clusters and place short inward-facing cuts.
    4) Keep at most a few main internal volume lines, with sparse diagonal accents.
    Every generated mark is a genuine SVG polyline clipped within the region.
    """
    region, _ = select_closed_region(mask, ns.accent_region)
    if ns.accent_density == 0:
        return [], [], region
    rng = np.random.default_rng(ns.accent_seed)
    safe = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    shadow, light = tonal_intensity(image, region)
    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(v, (0, 0), 1.7)
    gy, gx = np.gradient(smooth)
    grad = np.clip(np.hypot(gx, gy) / 12.5, 0, 1)
    distance = cv2.distanceTransform(safe, cv2.DIST_L2, 5)
    form = _feature_spines(detail_paths, safe, ns.accent_inner_lines)
    marks: list[np.ndarray] = []
    used_centers: list[np.ndarray] = []

    def add_mark(center: np.ndarray, tangent: np.ndarray, inward: np.ndarray,
                 length: float, target: str) -> bool:
        # The cut is diagonal to the contour's tangent, but biased inward.
        if ns.accent_follow == "diagonal":
            vec = np.array([.81, -.59], np.float32)
        else:
            vec = .64*tangent + .77*inward
            vec /= max(1e-5, np.linalg.norm(vec))
        possible = _clip_segment(center, vec, length, safe)
        if possible is None:
            return False
        # La distanza minima segue la densita LOCALE, non la distanza tra zone.
        # Quindi aumentare la fittezza NON produce nuove zone o nuovi ancoraggi.
        local_min_distance = max(2.75, ns.accent_spacing * .41 / ns.accent_local_density)
        if any(np.linalg.norm(center - old) < local_min_distance for old in used_centers):
            return False
        marks.append(possible)
        used_centers.append(center)
        return True

    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_entries = []
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length < 45:
            continue
        points = contour.reshape(-1, 2).astype(np.float32)
        # CHAIN_APPROX_NONE stores approximately one point per pixel.
        step = max(1, int(round(ns.accent_spacing/4)))
        for i in range(0, len(points), step):
            before = points[(i - 7) % len(points)]
            after = points[(i + 7) % len(points)]
            tangent = after-before
            tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
            n1 = np.array([-tangent[1], tangent[0]], np.float32)
            n2 = -n1
            point = points[i]
            inward = n1 if _inside(region, *(point + n1*4)) else n2
            center = point + inward*max(3.8, ns.accent_length*.70)
            if not _inside(safe, *center):
                continue
            ix, iy = int(round(center[0])), int(round(center[1]))
            width = float(distance[iy, ix])
            if width < 1.8:
                continue
            dark, bright = float(shadow[iy, ix]), float(light[iy, ix])
            strength = float(grad[iy, ix])
            score = .12 + 1.18*dark**1.25 + .18*bright**1.5 + .16*strength
            # Prefer genuinely shaded/structured zones, not arbitrary uniform areas.
            if dark < .10 and strength < .24 and bright < .52:
                continue
            contour_entries.append((score, i, len(points), point, tangent, inward, center, width,
                                    dark, bright, points))

    # Choose anchor sites weighted by tone, then enforce generous separation.
    # This creates local bursts of marks with untouched spaces in between.
    nclusters = min(18, max(1, round(ns.accent_clusters * ns.accent_density / .35)))
    if ns.accent_border_weight < .08:
        nclusters = 0
    else:
        nclusters = max(1, round(nclusters * (ns.accent_border_weight/.78)**.65))
    anchors = []
    pool = list(range(len(contour_entries)))
    for _ in range(nclusters):
        if not pool:
            break
        weights = np.array([contour_entries[j][0]**2.5 for j in pool], dtype=np.float64)
        weights /= max(weights.sum(), 1e-10)
        selected_idx = pool[int(rng.choice(len(pool), p=weights))]
        entry = contour_entries[selected_idx]
        anchors.append(entry)
        p = entry[3]
        # Euclidean and arc-based separation avoid regular, even distributions.
        pool = [j for j in pool if np.linalg.norm(contour_entries[j][3]-p) >= ns.accent_spacing*2.1]

    for anchor in anchors:
        _, base_idx, npts, _p, _t, _n, _c, _w, _sd, _lt, boundary = anchor
        # La selezione dei gruppi (anchors) e' gia' conclusa a questo punto.
        # Il valore della densita' locale NON modifica nclusters, gli anchor,
        # l'ampiezza della zona o le soglie per la selezione dei gruppi.
        original_marks = max(2, min(9, round(5 * (ns.accent_density/.35)**.58)))
        original_marks = max(2, round(original_marks*(.65 + .38*anchor[8])))
        original_step = max(4, round(ns.accent_spacing*.53))
        span = (original_marks - 1) * original_step
        target_marks = max(original_marks, round(original_marks * ns.accent_local_density))
        # Con 1.0 si riproduce esattamente l'algoritmo v4; quando si aumenta,
        # si aggiungono segni nello stesso intervallo, senza allargarlo.
        if ns.accent_local_density == 1.:
            offsets = (np.arange(original_marks) - (original_marks - 1)/2) * original_step
            position_jitter = 2.
        else:
            offsets = np.linspace(-span / 2, span / 2, target_marks)
            position_jitter = .65
        rng.shuffle(offsets)
        created = 0
        for off in offsets:
            idx = int((base_idx + round(float(off) + rng.uniform(-position_jitter, position_jitter))) % npts)
            p = boundary[idx]
            tangent = boundary[(idx+6)%npts]-boundary[(idx-6)%npts]
            tangent = tangent/max(float(np.linalg.norm(tangent)), 1e-6)
            inward = np.array([-tangent[1], tangent[0]], np.float32)
            if not _inside(region, *(p + inward*4)):
                inward *= -1.
            x = p + inward * max(4., ns.accent_length*.67)
            x += tangent*rng.uniform(-ns.accent_jitter, ns.accent_jitter)*ns.accent_spacing*.38
            if not _inside(safe, *x):
                continue
            xi, yi = int(round(x[0])), int(round(x[1]))
            darkness = float(shadow[yi, xi])
            length = ns.accent_length*rng.uniform(.78, 1.14)*(.80+.33*darkness)
            if add_mark(x.astype(np.float32), tangent, inward, length, "edge"):
                created += 1

    # Accents along a visual fold: fewer than the contour marks and aligned with
    # the extracted feature's tangent; these are discrete dashes, not a hatch fill.
    for line in form:
        if len(line) < 10:
            continue
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0),axis=1))]
        total = cumulative[-1]
        if total < 25:
            continue
        interval = ns.accent_spacing*max(1.1, (.78 / max(.2, 1. - ns.accent_border_weight)))**.5
        for value in np.arange(interval*.6, total, interval):
            if rng.random() > min(1., .78 * ns.accent_density/.35):
                continue
            i = int(np.searchsorted(cumulative, value))
            i = max(2, min(len(line)-3,i))
            c = line[i].copy()
            tangent = line[i+2]-line[i-2]
            tangent /= max(np.linalg.norm(tangent), 1e-5)
            inward = np.array([-tangent[1], tangent[0]], np.float32)
            # Bias center slightly to one side of the crease.
            c += inward*rng.uniform(-1.3, 1.3)
            if not _inside(safe, *c):
                continue
            ix,iy=int(round(c[0])),int(round(c[1]))
            local = float(shadow[iy,ix])
            length=ns.accent_length*rng.uniform(.48,.91)*(.72+.30*local)
            made = add_mark(c.astype(np.float32), tangent, inward, length, "inside")
            if made and ns.accent_local_density > 1.:
                # Piccoli segni solo ACCANTO a un segno interno gia' esistente:
                # nessuna nuova linea guida e nessuna nuova zona di incisione.
                extra = max(0, round(ns.accent_local_density * 2) - 2)
                for k in range(extra):
                    side = -1 if k % 2 == 0 else 1
                    offset = side * (3.5 + 2.0 * (k // 2))
                    nearby = c + tangent * offset
                    if _inside(safe, *nearby):
                        add_mark(nearby.astype(np.float32), tangent, inward,
                                 length*rng.uniform(.69,.85), "inside")

    return marks, form, region


def detect_layers(img: np.ndarray, mask: np.ndarray | None, s: Settings,
                  ns: argparse.Namespace) -> Layers:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8)).apply(gray)
    processed = cv2.GaussianBlur(contrast, (s.blur, s.blur), 0)
    outer_edges = cv2.Canny(processed, s.low, s.high, L2gradient=True)
    if mask is not None:
        margin = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        outer_edges = cv2.bitwise_and(outer_edges, margin)

    if ns.mode == "silhouette":
        if mask is None:
            raise ValueError("--mode silhouette richiede una maschera: --mask, --grabcut o --auto-mask white")
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        outer = as_paths(smoothed, s.min_length, s.epsilon, s.max_paths, closed=True)
        empty = np.zeros_like(gray)
        return Layers(outer, [], [], [], [], None, mask, True, smoothed, empty, empty)

    # With segmentation, the true outside silhouette gets its own editable group.
    # Strong reflection edges should not be counted as thick external contours.
    outer_closed = ns.mode == "engraving" and mask is not None
    outer = (as_paths(mask, s.min_length, s.epsilon, s.max_paths, closed=True)
             if outer_closed else
             as_paths(outer_edges, s.min_length, s.epsilon, s.max_paths))
    details_edges = np.zeros_like(gray)
    highlights_edges = np.zeros_like(gray)
    details: list[np.ndarray] = []
    highlights: list[np.ndarray] = []
    hatch: list[np.ndarray] = []
    hatch_light: list[np.ndarray] = []
    hatch_region: np.ndarray | None = None
    accents: list[np.ndarray] = []
    form_lines: list[np.ndarray] = []

    if ns.mode == "engraving":
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        # Dual pass: HSV V recovers brightness changes which grayscale compresses
        # for saturated red objects; CLAHE additionally boosts faint gradients.
        v = cv2.bilateralFilter(v, 5, 38, 38)
        enhanced_v = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(v)
        details_edges = cv2.bitwise_or(
            cv2.Canny(enhanced_v, s.detail_low, s.detail_high, L2gradient=True),
            outer_edges if outer_closed else np.zeros_like(outer_edges)
        )
        highlight_binary = highlight_map(img, mask, s.highlight_threshold)
        highlights_edges = cv2.Canny(highlight_binary, 20, 65, L2gradient=True)

        if mask is not None:
            inside = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            details_edges = cv2.bitwise_and(details_edges, inside)
            highlights_edges = cv2.bitwise_and(highlights_edges, inside)

        # Avoid tracing the outer silhouette twice. When segmentation is
        # available, do not suppress genuine inner reflections from Canny.
        exclude_source = (cv2.Canny(mask, 30, 80) if outer_closed
                          else outer_edges)
        exclude = cv2.dilate(exclude_source,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        details_edges = cv2.bitwise_and(details_edges, cv2.bitwise_not(exclude))
        highlights_edges = cv2.bitwise_and(highlights_edges, cv2.bitwise_not(exclude))
        # Separate the specular ridges from other internal detail lines.
        highlights_buffer = cv2.dilate(highlights_edges,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        details_edges = cv2.bitwise_and(details_edges, cv2.bitwise_not(highlights_buffer))
        details = as_paths(details_edges, s.detail_min_length, s.detail_epsilon,
                           s.detail_max_paths)
        highlights = as_paths(highlights_edges, s.detail_min_length,
                              s.detail_epsilon, s.detail_max_paths)
        if ns.style == "accent":
            if mask is None:
                raise ValueError("--style accent richiede una maschera: usa --grabcut, --mask o uno sfondo uniforme")
            # In this style only the contour and 0..4 principal form lines survive.
            # Raw Canny internal details would make the drawing cluttered again.
            accents, form_lines, hatch_region = make_accent_strokes(img, mask, ns, details)
            details = []
            highlights = []
        elif ns.hatching == "adaptive":
            if mask is None:
                raise ValueError("--hatching adaptive richiede una maschera: usa una foto su sfondo uniforme, --grabcut o --mask")
            hatch, hatch_light, hatch_region = make_adaptive_hatches(img, mask, ns)
        else:
            hatch = make_hatches(img, mask, ns.hatching)

    return Layers(outer, details, highlights, hatch, hatch_light, hatch_region, mask, outer_closed,
                  outer_edges, details_edges, highlights_edges, accents, form_lines)


def to_path(points: np.ndarray, closed: bool = False) -> str:
    result = "M " + " L ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in points)
    return result + (" Z" if closed else "")


def svg_group(root: ET.Element, name: str, paths: list[np.ndarray],
              color: str, stroke_width: float, opacity: float,
              closed: bool = False) -> None:
    if not paths:
        return
    g = ET.SubElement(root, "g", {
        "id": name, "fill": "none", "stroke": color,
        "stroke-width": str(stroke_width), "stroke-opacity": f"{opacity:.4g}",
        "stroke-linecap": "round", "stroke-linejoin": "round",
    })
    for points in paths:
        ET.SubElement(g, "path", {"d": to_path(points, closed)})



def _sample_polyline(points: np.ndarray, distance: float, closed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, np.float32)
    if len(pts) < 2:
        return pts[0].astype(np.float32), np.array([1., 0.], np.float32)
    work = np.vstack([pts, pts[0]]) if closed else pts.copy()
    seg = np.diff(work, axis=0)
    lens = np.linalg.norm(seg, axis=1)
    valid = lens > 1e-6
    if not np.any(valid):
        return pts[0].astype(np.float32), np.array([1., 0.], np.float32)
    cum = np.concatenate([[0.], np.cumsum(lens)])
    total = float(cum[-1])
    if total <= 1e-6:
        return pts[0].astype(np.float32), np.array([1., 0.], np.float32)
    if closed:
        distance = distance % total
    else:
        distance = min(max(0., distance), total - 1e-6)
    idx = int(np.searchsorted(cum, distance, side='right') - 1)
    idx = min(idx, len(seg) - 1)
    local_len = float(lens[idx])
    if local_len <= 1e-6:
        point = work[idx].astype(np.float32)
        tangent = np.array([1., 0.], np.float32)
    else:
        t = (distance - cum[idx]) / local_len
        point = work[idx] + seg[idx] * t
        tangent = seg[idx] / local_len
    return point.astype(np.float32), tangent.astype(np.float32)


def make_outline_texture(paths: list[np.ndarray], ns: argparse.Namespace) -> list[np.ndarray]:
    if ns.outline_style != 'cut':
        return []
    rng = np.random.default_rng(ns.accent_seed + 701)
    textured: list[np.ndarray] = []
    for path in paths:
        pts = np.asarray(path, np.float32)
        if len(pts) < 3:
            continue
        closed = True
        work = np.vstack([pts, pts[0]])
        seg = np.diff(work, axis=0)
        lens = np.linalg.norm(seg, axis=1)
        total = float(lens.sum())
        if total < ns.outline_cut_length * 2:
            continue
        gap = max(4.0, ns.outline_cut_length * 1.08 / max(.25, ns.outline_cut_density))
        count = max(3, int(total / gap))
        base_shift = rng.uniform(0, gap)
        for i in range(count):
            if rng.random() > min(1.0, 0.68 + 0.18 * ns.outline_cut_density):
                continue
            start_d = base_shift + i * gap
            seg_len = ns.outline_cut_length * rng.uniform(.72, 1.28)
            k = int(max(3, round(seg_len / 3.2)))
            ds = np.linspace(start_d, start_d + seg_len, k)
            out = []
            for d in ds:
                p, t = _sample_polyline(pts, d, closed=closed)
                n = np.array([-t[1], t[0]], np.float32)
                p = p + n * rng.uniform(-1., 1.) * ns.outline_jitter * ns.outline_cut_length * .45
                p = p + t * rng.uniform(-1., 1.) * ns.outline_jitter * 1.8
                out.append(p)
            textured.append(np.array(out, np.float32))
    return textured


def _composite_rgba(base: np.ndarray, over: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    over_f = over.astype(np.float32) / 255.0
    oa = over_f[:, :, 3:4]
    ba = base_f[:, :, 3:4]
    out_a = oa + ba * (1.0 - oa)
    out_rgb = over_f[:, :, :3] * oa + base_f[:, :, :3] * ba * (1.0 - oa)
    out_rgb = np.where(out_a > 1e-6, out_rgb / np.maximum(out_a, 1e-6), 0)
    out = np.dstack([out_rgb, out_a])
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def draw_group_rgba(canvas: np.ndarray, paths: list[np.ndarray], color: tuple[int, int, int],
                    thickness: float, opacity: float, closed: bool = False) -> np.ndarray:
    if not paths:
        return canvas
    over = np.zeros_like(canvas)
    t = max(1, round(thickness))
    rgba = (color[0], color[1], color[2], max(1, min(255, round(opacity * 255))))
    for p in paths:
        cv2.polylines(over, [np.rint(p).astype(np.int32).reshape(-1, 1, 2)], closed, rgba, t, cv2.LINE_AA)
    return _composite_rgba(canvas, over)


def export_transparent_png(dest: Path, layers: Layers, w: int, h: int, ns: argparse.Namespace) -> None:
    ink = bgr(ns.stroke)
    canvas = np.zeros((h, w, 4), np.uint8)
    base_outer_opacity = ns.opacity if not layers.outer_texture else ns.opacity * .36
    canvas = draw_group_rgba(canvas, layers.outer, ink, ns.stroke_width, base_outer_opacity, layers.outer_closed)
    canvas = draw_group_rgba(canvas, layers.outer_texture, ink, max(.8, ns.stroke_width * .92), ns.opacity * .98, False)
    canvas = draw_group_rgba(canvas, layers.details, ink, ns.detail_width, ns.opacity * ns.detail_opacity, False)
    canvas = draw_group_rgba(canvas, layers.highlights, ink, ns.highlight_width, ns.opacity * ns.highlight_opacity, False)
    canvas = draw_group_rgba(canvas, layers.hatch, ink, ns.hatch_width if ns.hatching == 'adaptive' else ns.detail_width * .7,
                             ns.opacity * (ns.hatch_opacity if ns.hatching == 'adaptive' else .42), False)
    canvas = draw_group_rgba(canvas, layers.hatch_light, ink, ns.hatch_width * .72, ns.opacity * ns.hatch_opacity * .85, False)
    canvas = draw_group_rgba(canvas, layers.accent_form, ink, ns.accent_width * .65, ns.opacity * .54, False)
    canvas = draw_group_rgba(canvas, layers.accent_marks, ink, ns.accent_width, ns.opacity * .92, False)
    write_png(dest, canvas)


def export_svg(dest: Path, layers: Layers, w: int, h: int,
               ns: argparse.Namespace) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.Element("{http://www.w3.org/2000/svg}svg", {
        "viewBox": f"0 0 {w} {h}", "width": str(w), "height": str(h),
        "role": "img", "aria-label": "Incisione vettoriale da fotografia",
    })
    ET.SubElement(root, "title").text = "O’ Spuntin — Incisione fotografica"
    if ns.svg_background:
        ET.SubElement(root, "rect", {
            "width": str(w), "height": str(h), "fill": ns.background,
        })
    svg_group(root, "outer-contours", layers.outer, ns.stroke,
              ns.stroke_width, ns.opacity if not layers.outer_texture else ns.opacity*.36, layers.outer_closed)
    svg_group(root, "outer-cuts", layers.outer_texture, ns.stroke,
              max(.8, ns.stroke_width*.92), ns.opacity*.98, False)
    svg_group(root, "inner-details", layers.details, ns.stroke,
              ns.detail_width, ns.opacity*ns.detail_opacity)
    svg_group(root, "specular-highlights", layers.highlights, ns.stroke,
              ns.highlight_width, ns.opacity*ns.highlight_opacity)
    svg_group(root, "shadow-hatching", layers.hatch, ns.stroke,
              ns.hatch_width if ns.hatching == "adaptive" else ns.detail_width*.7,
              ns.opacity*(ns.hatch_opacity if ns.hatching == "adaptive" else .42))
    svg_group(root, "light-hatching", layers.hatch_light, ns.stroke,
              ns.hatch_width*.72, ns.opacity*ns.hatch_opacity*.85)
    svg_group(root, "accent-form-lines", layers.accent_form, ns.stroke,
              ns.accent_width*.65, ns.opacity*.54)
    svg_group(root, "accent-cuts", layers.accent_marks, ns.stroke,
              ns.accent_width, ns.opacity*.92)
    ET.ElementTree(root).write(dest, encoding="utf-8", xml_declaration=True)


def region_guide(mask: np.ndarray) -> np.ndarray:
    """Debug PNG showing 1-based regions in descending area order."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.uint8(mask > 0), connectivity=8)
    ranks = sorted(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    minimum = max(12, round(mask.size * .00006))
    guide = np.full((*mask.shape, 3), (24, 24, 24), np.uint8)
    for idx, label in enumerate(ranks, start=1):
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum:
            continue
        color = (80, 110, 135) if idx == 1 else (75, 80, 135)
        guide[labels == label] = color
        x, y = (int(round(c)) for c in centroids[label])
        cv2.putText(guide, str(idx), (x-7, y+5), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (245, 245, 245), 2, cv2.LINE_AA)
    return guide


def bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = bytes.fromhex(hex_color[1:])
    return b, g, r


def draw_group(canvas: np.ndarray, paths: list[np.ndarray], color: tuple,
               thickness: float, opacity: float, closed: bool = False) -> np.ndarray:
    if not paths:
        return canvas
    over = canvas.copy()
    t = max(1, round(thickness))
    for p in paths:
        cv2.polylines(over, [np.rint(p).astype(np.int32).reshape(-1, 1, 2)], closed, color, t, cv2.LINE_AA)
    return cv2.addWeighted(over, opacity, canvas, 1.-opacity, 0)


def preview_png(dest: Path, img: np.ndarray, layers: Layers,
                ns: argparse.Namespace) -> None:
    h, w = img.shape[:2]
    background = np.full((h, w, 3), bgr(ns.background), np.uint8)
    ink = bgr(ns.stroke)
    left = draw_group(background.copy(), layers.outer, ink, ns.stroke_width,
                      ns.opacity if not layers.outer_texture else ns.opacity*.36, layers.outer_closed)
    left = draw_group(left, layers.outer_texture, ink, max(.8, ns.stroke_width*.92),
                      ns.opacity*.98, False)
    final = draw_group(left.copy(), layers.details, ink,
                       ns.detail_width, ns.opacity*ns.detail_opacity)
    final = draw_group(final, layers.highlights, ink,
                       ns.highlight_width, ns.opacity*ns.highlight_opacity)
    final = draw_group(final, layers.hatch, ink,
                       ns.hatch_width if ns.hatching == "adaptive" else ns.detail_width*.7,
                       ns.opacity*(ns.hatch_opacity if ns.hatching == "adaptive" else .42))
    final = draw_group(final, layers.hatch_light, ink,
                       ns.hatch_width*.72, ns.opacity*ns.hatch_opacity*.85)
    final = draw_group(final, layers.accent_form, ink,
                       ns.accent_width*.65, ns.opacity*.54)
    final = draw_group(final, layers.accent_marks, ink,
                       ns.accent_width, ns.opacity*.92)
    if ns.hatching == "adaptive" or ns.style == "accent":
        write_png(dest.with_name(dest.name.replace("_preview.png", "_art.png")), final)
    bar_h = 34
    panes = []
    for frame, title in [(img, "ORIGINALE"), (left, "CONTORNI"),
                         (final, "RIFLESSI + INCISIONE")]:
        bar = np.full((bar_h, w, 3), bgr("#243540"), np.uint8)
        cv2.putText(bar, title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, bgr("#E2C48D"), 1, cv2.LINE_AA)
        panes.append(np.vstack((bar, frame)))
    separator = np.full((h+bar_h, 12, 3), bgr("#DBD6C9"), np.uint8)
    write_png(dest, np.hstack((panes[0], separator, panes[1], separator, panes[2])))


def process_one(path: Path, ns: argparse.Namespace, s: Settings) -> dict:
    image = resize(read_image(path), ns.max_side)
    mask = make_mask(image, ns)
    if mask is not None and cv2.countNonZero(mask) == 0:
        raise ValueError("Maschera vuota")
    # For generic non-white backgrounds, try central-object segmentation if
    # the new adaptive engraving was requested and no silhouette was found.
    if mask is None and (ns.hatching == "adaptive" or ns.style == "accent") and ns.mode == "engraving" and ns.auto_mask == "auto" and not ns.mask:
        fallback = argparse.Namespace(**vars(ns))
        fallback.grabcut = True
        try:
            mask = make_mask(image, fallback)
        except cv2.error:
            mask = None
    layers = detect_layers(image, mask, s, ns)
    layers.outer_texture = make_outline_texture(layers.outer, ns)
    if not (layers.outer or layers.details or layers.highlights or layers.hatch or layers.hatch_light or layers.accent_marks or layers.accent_form):
        raise ValueError("Nessun tracciato: usa --preset detailed oppure --min-length 15")
    ns.output.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    svg = ns.output / f"{path.stem}.svg"
    export_svg(svg, layers, w, h, ns)
    if ns.preview:
        preview_png(ns.output / f"{path.stem}_preview.png", image, layers, ns)
    if ns.export_transparent:
        export_transparent_png(ns.output / f"{path.stem}_art_transparent.png", layers, w, h, ns)
    if ns.debug:
        combined = np.zeros((h, w, 3), np.uint8)
        combined[:, :, 0] = layers.outer_edges
        combined[:, :, 1] = layers.details_edges
        combined[:, :, 2] = layers.highlights_edges
        write_png(ns.output / f"{path.stem}_debug.png", combined)
        if mask is not None:
            write_png(ns.output / f"{path.stem}_mask.png", mask)
        if layers.hatch_region is not None:
            write_png(ns.output / f"{path.stem}_hatch_region.png", layers.hatch_region)
            write_png(ns.output / f"{path.stem}_regions.png", region_guide(mask))
    return {"svg": svg, "outer": len(layers.outer), "details": len(layers.details),
            "highlights": len(layers.highlights), "hatches": len(layers.hatch) + len(layers.hatch_light),
            "accents": len(layers.accent_marks), "form_lines": len(layers.accent_form),
            "mask": mask is not None, "size": (w, h)}


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    try:
        s = validated(ns)
        files = get_files(ns.input)
        if ns.mask and len(files) != 1:
            raise ValueError("--mask richiede un solo file di input")
        if ns.mask and not ns.mask.is_file():
            raise FileNotFoundError(f"Maschera non trovata: {ns.mask}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERRORE] {exc}", file=sys.stderr)
        return 2
    failed = 0
    for file in files:
        try:
            result = process_one(file, ns, s)
            print(f"[OK] {file.name} -> {result['svg']} | "
                  f"contorni {result['outer']}, interni {result['details']}, "
                  f"riflessi {result['highlights']}, tratteggi {result['hatches']}, "
                  f"accenti {result['accents']}, linee forma {result['form_lines']} "
                  f"| maschera {'SI' if result['mask'] else 'NO'}")
            if ns.mode == "engraving" and not result["mask"]:
                print("     Nota: nessuna maschera. Per foto su sfondo complesso "
                      "prova --grabcut oppure --mask.")
        except (ValueError, OSError, cv2.error) as exc:
            failed += 1
            print(f"[ERRORE] {file.name}: {exc}", file=sys.stderr)
    print(f"\nElaborate {len(files) - failed}/{len(files)} fotografie. Output: {ns.output.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
