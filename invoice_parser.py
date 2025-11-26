#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
invoice_parser.py
Extractor pragmático para facturas (UY) desde PDFs o imágenes (OCR).

Características:
- Limitar recursos: threads + prioridad baja (Windows)
- QR primero (si opencv-python está instalado)
- OCR fast por recortes + fallback a OCR completo si faltan campos
- Extracción robusta por candidatos + scoring + validaciones
- Reporte contra ground-truth (gold.json) para medir aciertos

Dependencias recomendadas:
- pillow
- numpy
- easyocr (para OCR)
- opencv-python (para QR + preprocesado mejor)
- pypdf (para PDF texto) o pdfplumber (fallback)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------
# Resource limiting utilities
# ---------------------------

def set_low_priority_windows() -> bool:
    """Best-effort: set process priority below normal on Windows."""

    if os.name != "nt":
        return False

    try:
        import ctypes  # noqa: F401
        import ctypes.wintypes  # noqa: F401
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetCurrentProcess()
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        ok = kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        return bool(ok)
    except Exception:
        return False


def configure_cpu_threads(cpu_threads: int) -> None:
    """Set common env vars + torch threads (if available) to limit CPU usage."""
    cpu_threads = max(1, int(cpu_threads))

    # Env vars that many libs respect (OpenMP/MKL/NumExpr/BLAS).
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))

    # Torch threads (EasyOCR uses torch).
    try:
        import torch  # type: ignore
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, min(cpu_threads, 2)))
    except Exception:
        pass


# ---------------------------
# Data model
# ---------------------------

@dataclass
class InvoiceResult:
    fecha: Optional[str] = None
    serie: Optional[str] = None
    folio: Optional[str] = None
    serie_y_folio: Optional[str] = None
    razon_social: Optional[str] = None
    rut_emisor: Optional[str] = None
    es_nota_de_credito: bool = False
    importe_total_con_iva: Optional[str] = None
    importe_total_con_iva_num: Optional[float] = None
    importe_sin_iva: Optional[str] = None
    importe_sin_iva_num: Optional[float] = None
    importe_sin_iva_fuente: Optional[str] = None

    _archivo: Optional[str] = None
    _fuente: Optional[str] = None


# ---------------------------
# Text normalization + parsing
# ---------------------------

_GENERIC_STOPWORDS = {
    "RUC", "RUT", "NOMBRE", "DENOMINACION", "DOMICILIO", "FISCAL", "CIUDAD",
    "PAIS", "FECHA", "VENCIMIENTO", "MONEDA", "TOTAL", "SUBTOTAL", "IVA",
    "COMPRADOR", "RECEPTOR", "TIPO", "DOCUMENTO", "SERIE", "NUMERO",
    "PAGINA", "OBSERVACIONES", "CODIGO", "SECURIDAD", "CAE", "CAI",
    "VALIDEZ", "VERIFICAR", "COMPROBANTE", "DGI",
}

def normalize_ws(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def to_upper_safe(s: str) -> str:
    return normalize_ws(s).upper()

def clean_token(s: str) -> str:
    s = re.sub(r"[^A-Z0-9ÁÉÍÓÚÜÑ/.\- ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_all_ruts(text: str) -> List[str]:
    # UY RUT/RUC: 12 dígitos típico
    return re.findall(r"\b(\d{12})\b", text)

def parse_date_candidates(text: str) -> List[Tuple[str, int, int]]:
    """
    Return list of (date_str, start_idx, end_idx)
    Accept dd/mm/yyyy or d/m/yyyy (also with spaces).
    """
    candidates = []
    for m in re.finditer(r"\b(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})\b", text):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        ds = f"{int(d):02d}/{int(mo):02d}/{y}"
        candidates.append((ds, m.start(), m.end()))
    return candidates

def _amount_str_to_float(s: str) -> Optional[float]:
    """Parse amounts like '4.018,01', '4 018,01', '4018,01', '1.540,00'."""
    s0 = s.strip()
    if not s0:
        return None

    # keep digits, comma, dot, space
    s0 = re.sub(r"[^\d,.\s\-]", "", s0)

    # Handle negative
    neg = False
    if "-" in s0:
        neg = True
        s0 = s0.replace("-", " ")

    s0 = re.sub(r"\s+", " ", s0).strip()

    # If both '.' and ',' exist, assume ',' decimal and '.' thousand (common UY)
    # If only ',' exists: comma decimal
    # If only '.' exists: ambiguous; treat '.' as decimal if looks like xx.xx else thousand
    if "," in s0:
        # remove thousand separators: dots/spaces
        s1 = s0.replace(".", "").replace(" ", "")
        # keep comma as decimal point
        s1 = s1.replace(",", ".")
    else:
        # no comma
        # remove spaces
        s1 = s0.replace(" ", "")
        # if more than one dot -> thousand separators
        if s1.count(".") > 1:
            s1 = s1.replace(".", "")
        else:
            # single dot: decide if it's decimal or thousand
            # if exactly 2 digits after dot -> decimal else thousand
            if re.search(r"\.\d{2}\b", s1):
                pass
            else:
                s1 = s1.replace(".", "")

    try:
        val = float(s1)
        return -val if neg else val
    except Exception:
        return None

def parse_amount_candidates(text: str) -> List[Tuple[str, float, int, int]]:
    """
    Find amount candidates and return list (raw, value, start, end).
    Tries to catch '4 018,01' and '4.018,01' and '4018,01'.
    """
    cands: List[Tuple[str, float, int, int]] = []

    # Pattern allowing spaces/dots as thousand separators, comma as decimal
    # Also catches plain "4018,01"
    pattern = r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})\b|\b\d+(?:,\d{2})\b"
    for m in re.finditer(pattern, text):
        raw = m.group(0)
        v = _amount_str_to_float(raw)
        if v is None:
            continue
        cands.append((raw, v, m.start(), m.end()))

    # Also catch cases like "4 018,01" OCR split into "4 018,01" is handled above,
    # but sometimes it becomes "4 018, 01" with spaces around comma.
    pattern2 = r"\b\d{1,3}(?:[.\s]\d{3})*\s*,\s*\d{2}\b"
    for m in re.finditer(pattern2, text):
        raw = re.sub(r"\s+", "", m.group(0).replace(" ", ""))
        raw = raw.replace(",,", ",")
        v = _amount_str_to_float(raw)
        if v is None:
            continue
        cands.append((m.group(0), v, m.start(), m.end()))

    # de-dup by (value, start)
    uniq = []
    seen = set()
    for raw, v, a, b in cands:
        k = (round(v, 2), a)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((raw, v, a, b))
    return uniq

def parse_iva_rate(text: str) -> Optional[float]:
    """
    Heurística: detecta si aparece 22% o 10% cerca de 'IVA'.
    Devuelve 0.22 u 0.10 si está claro, sino None.
    """
    t = to_upper_safe(text)
    has22 = bool(re.search(r"IVA[^0-9]{0,20}22\s*[%*]?", t)) or ("(22%)" in t) or (" 22% " in t) or ("22*" in t)
    has10 = bool(re.search(r"IVA[^0-9]{0,20}10\s*[%*]?", t)) or ("(10%)" in t) or (" 10% " in t) or ("10*" in t)
    if has22 and not has10:
        return 0.22
    if has10 and not has22:
        return 0.10
    return None

def score_date_candidate(text_upper: str, date_str: str, start: int, end: int) -> float:
    """
    Score dates: prefer 'FECHA' context, penalize CAE/CAI/VENCIMIENTO.
    """
    ctx_left = text_upper[max(0, start - 50):start]
    ctx_right = text_upper[end:min(len(text_upper), end + 50)]
    ctx = ctx_left + " " + ctx_right

    score = 0.0

    # Prefer "FECHA" nearby
    if "FECHA" in ctx:
        score += 3.0

    # Penalize "VENC" nearby (often CAE/CAI expiration)
    if "VENC" in ctx or "VENCE" in ctx:
        score -= 2.5

    # Penalize CAE/CAI context strongly
    if "CAE" in ctx or "CAI" in ctx:
        score -= 4.0

    # Plausibility by year (not too far future)
    try:
        d = _dt.datetime.strptime(date_str, "%d/%m/%Y").date()
        # Date far in future? penalize hard
        today = _dt.date.today()
        if d > today + _dt.timedelta(days=365):
            score -= 3.0
        # Date far in past (still possible), mild penalty
        if d < today - _dt.timedelta(days=3650):
            score -= 1.0
    except Exception:
        score -= 1.0

    return score

def pick_best_date(text: str) -> Optional[str]:
    t_up = to_upper_safe(text)
    cands = parse_date_candidates(t_up)
    scored = []
    for ds, a, b in cands:
        sc = score_date_candidate(t_up, ds, a, b)
        scored.append((sc, ds, a))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], -x[2]), reverse=True)
    # require some minimum signal
    best = scored[0]
    return best[1]

def score_amount_candidate(text_upper: str, raw: str, value: float, start: int, end: int) -> float:
    """
    Score amounts: prefer 'TOTAL'/'TOTAL A PAGAR' context, penalize small values.
    """
    ctx_left = text_upper[max(0, start - 60):start]
    ctx_right = text_upper[end:min(len(text_upper), end + 60)]
    ctx = ctx_left + " " + ctx_right

    score = 0.0

    # keywords
    if "TOTAL A PAGAR" in ctx:
        score += 5.0
    if re.search(r"\bTOTAL\b", ctx):
        score += 3.0
    if "SUBTOTAL" in ctx:
        score -= 0.5
    if "IVA" in ctx and "TOTAL" not in ctx:
        score -= 0.5

    # Penalize tiny totals (OCR often finds 12,47 / 18,01 fragmentos)
    if value < 50:
        score -= 5.0
    elif value < 200:
        score -= 2.0

    # Preference for amounts with decimals (most totals have cents)
    if re.search(r",\d{2}\b", raw):
        score += 0.3

    # If looks like "4 018,01" splitted is fine; no penalty.
    return score

def pick_best_total(text: str) -> Optional[Tuple[str, float, str]]:
    """
    Return (formatted_str, float_value, source_tag)
    - tries to pick total amount with best context. If not found, uses largest plausible.
    """
    t_up = to_upper_safe(text)
    cands = parse_amount_candidates(t_up)
    if not cands:
        return None

    scored: List[Tuple[float, str, float]] = []
    for raw, val, a, b in cands:
        sc = score_amount_candidate(t_up, raw, val, a, b)
        scored.append((sc, raw, val))

    # Prefer best scoring above threshold
    scored.sort(key=lambda x: (x[0], x[2]), reverse=True)

    best_sc, best_raw, best_val = scored[0]
    if best_sc >= 1.0:
        return (format_uy_amount(best_val), round(best_val, 2), "scored_context")

    # fallback: choose largest "reasonable" amount (avoid picking e.g., RUT-like numbers)
    plausible = [(raw, val) for _, raw, val in scored if 50 <= val <= 10_000_000]
    if plausible:
        raw, val = max(plausible, key=lambda t: t[1])
        return (format_uy_amount(val), round(val, 2), "max_plausible")

    return None

def format_uy_amount(v: float) -> str:
    """
    Format as UY style: thousands '.' and decimal ',' with 2 decimals.
    """
    v = round(float(v), 2)
    sign = "-" if v < 0 else ""
    v = abs(v)
    whole = int(v)
    frac = int(round((v - whole) * 100))
    whole_str = f"{whole:,}".replace(",", ".")
    return f"{sign}{whole_str},{frac:02d}"

def pick_rut_emisor(text: str) -> Optional[str]:
    t = to_upper_safe(text)
    ruts = find_all_ruts(t)
    if not ruts:
        return None

    # Prefer near "RUT EMISOR" or first after "RUC/RUT"
    best = None
    best_score = -1e9

    for rut in ruts:
        idx = t.find(rut)
        ctx = t[max(0, idx - 40): idx + 40]
        sc = 0.0
        if "RUT EMISOR" in ctx or "RUC EMISOR" in ctx:
            sc += 5.0
        if re.search(r"\b(RUC|RUT)\b", ctx):
            sc += 2.0
        if "RECEPTOR" in ctx or "COMPRADOR" in ctx:
            sc -= 1.5

        # Penalize if equals common receptor in tus ejemplos (pero sin hardcode fuerte)
        if rut == "218849400010":
            sc -= 3.0

        if sc > best_score:
            best_score = sc
            best = rut

    return best

def pick_razon_social(text: str) -> Optional[str]:
    """
    Heurística simple: toma la mejor frase "tipo empresa" arriba,
    evitando palabras genéricas.
    """
    t = clean_token(text)
    lines = t.split("\n")
    head = " ".join(lines[:12])  # la cabecera suele estar al principio
    head = re.sub(r"\s+", " ", head).strip()

    # Candidatos: secuencias de palabras MAYUSC >= 3 letras
    # Ej: "PURA PALTA SAS", "BIMBO", "MODELO NATURAL SRL"
    candidates = []

    # 1) Si hay "SAS" o "SRL" cerca, buena señal
    for m in re.finditer(r"\b([A-ZÁÉÍÓÚÜÑ]{3,}(?:\s+[A-ZÁÉÍÓÚÜÑ]{2,}){0,6})\s+(SAS|SRL|SA|S\.A\.|LTDA)\b", head):
        txt = (m.group(1) + " " + m.group(2)).strip()
        candidates.append((5.0, txt))

    # 2) Palabra fuerte conocida (bimbo) aparece fácil
    for kw in ["BIMBO", "PURA PALTA"]:
        if kw in head:
            candidates.append((4.0, kw))

    # 3) Fallback: frase más "limpia" (evita stopwords)
    chunks = re.findall(r"\b[A-ZÁÉÍÓÚÜÑ]{3,}\b(?:\s+\b[A-ZÁÉÍÓÚÜÑ]{3,}\b){0,3}", head)
    for ch in chunks:
        toks = ch.split()
        if any(tok in _GENERIC_STOPWORDS for tok in toks):
            continue
        if len("".join(toks)) < 4:
            continue
        score = 1.0 + 0.2 * len(toks)
        candidates.append((score, ch))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    # pick first not too generic
    return candidates[0][1].strip()

def parse_serie_folio(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Intentar extraer SERIE y NUMERO/FOLIO.
    - Primero por patrones explícitos "SERIE <X>" + "NUMERO <digits>"
    - Luego por "SERIE NUMERO" grid (PDFs suelen)
    - Luego por patrón "A 049009" o "A-3519972"
    """
    t = to_upper_safe(text)

    serie = None
    folio = None

    # Explicit "SERIE <letter>"
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,10}([A-Z])\b", t)
    if m:
        serie = m.group(1)

    # Explicit "NUMERO <digits>"
    m = re.search(r"\bNUMER[O0]\b[^0-9]{0,10}(\d{4,9})\b", t)
    if m:
        folio = m.group(1)

    # Sometimes "SERIE NUMERO" then "A 049009"
    if (serie is None or folio is None):
        m2 = re.search(r"\bSERIE\b.*?\bNUMER[O0]\b.*?\b([A-Z])\s+0?(\d{4,9})\b", t, flags=re.DOTALL)
        if m2:
            serie = serie or m2.group(1)
            folio = folio or m2.group(2).lstrip("0") or m2.group(2)

    # General pattern A-3519972 or A 3519972
    if (serie is None or folio is None):
        m3 = re.search(r"\b([A-Z])\s*[-]?\s*0?(\d{4,9})\b", t)
        if m3:
            # Score with context for safety
            idx = m3.start()
            ctx = t[max(0, idx - 30): idx + 30]
            if ("SERIE" in ctx) or ("NUMER" in ctx) or ("DOCUMENTO" in ctx):
                serie = serie or m3.group(1)
                folio = folio or (m3.group(2).lstrip("0") or m3.group(2))

    # Validate folio not something silly like "60" or "218849400010"
    if folio is not None and len(folio) < 4:
        folio = None
    if folio is not None and len(folio) > 9:
        folio = None

    return serie, folio

def detect_nota_credito(text: str) -> bool:
    t = to_upper_safe(text)
    # Nota de crédito real suele decirlo; "Credito" como forma de pago NO.
    if "NOTA DE CREDITO" in t or "NOTA CREDITO" in t or re.search(r"\bN\.?\s*C\.?\b", t):
        return True
    return False


# ---------------------------
# PDF text extraction
# ---------------------------

def extract_pdf_text(path: Path) -> str:
    # Try pypdf first
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
        return normalize_ws("\n".join(parts))
    except Exception:
        pass

    # Fallback: pdfplumber
    try:
        import pdfplumber  # type: ignore
        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                parts.append(p.extract_text() or "")
        return normalize_ws("\n".join(parts))
    except Exception:
        return ""


# ---------------------------
# Image loading + preprocessing
# ---------------------------

def load_image_pil(path: Path):
    from PIL import Image, ImageOps  # type: ignore
    img = Image.open(str(path))
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img

def limit_image_size(img, max_dim: int, max_pixels: int):
    from PIL import Image  # type: ignore
    w, h = img.size
    if max_dim and max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        w2, h2 = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((w2, h2), Image.BICUBIC)

    if max_pixels and (img.size[0] * img.size[1]) > max_pixels:
        w, h = img.size
        scale = math.sqrt(max_pixels / float(w * h))
        w2, h2 = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((w2, h2), Image.BICUBIC)

    return img

def pil_to_numpy_rgb(img):
    import numpy as np  # type: ignore
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)

def preprocess_for_ocr(img, aggressive: bool = False):
    """
    Preprocess PIL image; aggressive mode uses thresholding.
    """
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter  # type: ignore

    if img.mode != "RGB":
        img = img.convert("RGB")

    # to grayscale
    g = ImageOps.grayscale(img)

    # contrast
    g = ImageOps.autocontrast(g)

    # sharpen a bit
    g = g.filter(ImageFilter.SHARPEN)

    if aggressive:
        # increase contrast and apply threshold
        g2 = ImageEnhance.Contrast(g).enhance(2.0)
        # simple threshold
        g2 = g2.point(lambda p: 255 if p > 160 else 0)
        return g2

    return g

def crop_regions(img):
    """
    Return (header_img, totals_img) crops for fast mode.
    """
    w, h = img.size
    header_h = int(h * 0.45)
    totals_y = int(h * 0.65)
    header = img.crop((0, 0, w, header_h))
    totals = img.crop((0, totals_y, w, h))
    return header, totals


# ---------------------------
# QR decode (optional)
# ---------------------------

def try_decode_qr(img_rgb_np) -> List[str]:
    """
    Return decoded QR texts (if opencv installed).
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return []

    try:
        detector = cv2.QRCodeDetector()
        # detectAndDecodeMulti may not exist in old builds; try both
        texts: List[str] = []
        if hasattr(detector, "detectAndDecodeMulti"):
            ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img_rgb_np)
            if ok and decoded_info:
                for s in decoded_info:
                    if s:
                        texts.append(s)
        else:
            s, pts, _ = detector.detectAndDecode(img_rgb_np)
            if s:
                texts.append(s)
        return [normalize_ws(t) for t in texts if t and t.strip()]
    except Exception:
        return []


# ---------------------------
# OCR (EasyOCR optional)
# ---------------------------

class EasyOcrEngine:
    def __init__(self, languages: Sequence[str], gpu: bool = False, verbose: bool = False):
        self.languages = list(languages)
        self.gpu = bool(gpu)
        self.verbose = bool(verbose)
        self._reader = None

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        import easyocr  # type: ignore
        # verbose False reduces console spam; gpu False to avoid weirdness
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=self.verbose)
        return self._reader

    def read_text(self, img_np_rgb) -> List[str]:
        reader = self._get_reader()
        # detail=0 returns string list, much cheaper
        out = reader.readtext(img_np_rgb, detail=0, paragraph=False)
        # normalize
        out2 = []
        for s in out:
            if not s:
                continue
            s2 = normalize_ws(str(s))
            if s2:
                out2.append(s2)
        return out2


# ---------------------------
# Field extraction pipeline
# ---------------------------

def build_invoice_from_text(
    text: str,
    source: str,
    file_path: Path,
    debug: bool = False,
) -> InvoiceResult:
    res = InvoiceResult()
    res._archivo = str(file_path)
    res._fuente = source

    text_norm = normalize_ws(text)
    text_up = to_upper_safe(text_norm)

    res.es_nota_de_credito = detect_nota_credito(text_up)

    res.rut_emisor = pick_rut_emisor(text_up)
    res.razon_social = pick_razon_social(text_norm)

    serie, folio = parse_serie_folio(text_up)
    res.serie = serie
    res.folio = folio
    res.serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    res.fecha = pick_best_date(text_up)

    total_pick = pick_best_total(text_up)
    if total_pick:
        res.importe_total_con_iva, res.importe_total_con_iva_num, _src = total_pick
    else:
        res.importe_total_con_iva = None
        res.importe_total_con_iva_num = None

    # infer net amount (sin IVA) if IVA rate is detectable AND we have total
    iva_rate = parse_iva_rate(text_up)
    if iva_rate is not None and res.importe_total_con_iva_num is not None:
        net = round(res.importe_total_con_iva_num / (1.0 + iva_rate), 2)
        res.importe_sin_iva_num = net
        res.importe_sin_iva = format_uy_amount(net)
        res.importe_sin_iva_fuente = f"total_div_{int(iva_rate*100)}"
    else:
        res.importe_sin_iva = None
        res.importe_sin_iva_num = None
        res.importe_sin_iva_fuente = None

    if debug:
        # Fix obvious gotchas: if picked date is very future and there's another
        pass

    return res

def minimal_quality_ok(res: InvoiceResult) -> bool:
    """
    If core fields are missing, OCR fast likely failed: allow fallback.
    """
    if res.rut_emisor is None:
        return False
    if res.importe_total_con_iva_num is None:
        return False
    # Serie/folio can be missing in some layouts, but try.
    if res.fecha is None:
        return False
    return True


def parse_image_file(
    path: Path,
    ocr_engine: Optional[EasyOcrEngine],
    ocr_mode: str,
    max_dim: int,
    max_pixels: int,
    debug: bool = False,
    allow_fallback_full: bool = True,
) -> Tuple[Optional[InvoiceResult], str, Dict[str, str]]:
    """
    Returns (InvoiceResult or None, source_label, debug_blobs)
    debug_blobs may include header/totals text.
    """
    debug_blobs: Dict[str, str] = {}

    try:
        img = load_image_pil(path)
        img = limit_image_size(img, max_dim=max_dim, max_pixels=max_pixels)
    except Exception as e:
        return None, "error", {"error": f"no se pudo abrir imagen: {e}"}

    # Try QR first
    qr_texts = []
    try:
        np_rgb = pil_to_numpy_rgb(img)
        qr_texts = try_decode_qr(np_rgb)
    except Exception:
        qr_texts = []

    if qr_texts:
        # If QR exists, use it as primary text to parse fields.
        # Still can fall back to OCR if QR doesn’t contain enough.
        qr_join = "\n".join(qr_texts)
        debug_blobs["qr"] = qr_join
        res_qr = build_invoice_from_text(qr_join, source="image_qr", file_path=path, debug=debug)
        # If QR yielded enough, return
        if minimal_quality_ok(res_qr):
            return res_qr, "image_qr", debug_blobs
        # else keep going with OCR (QR maybe partial)

    if ocr_engine is None:
        return None, "error", {"error": "No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract."}

    def run_ocr_on_pil(pil_img, aggressive: bool) -> str:
        pil_img2 = preprocess_for_ocr(pil_img, aggressive=aggressive)
        np_rgb2 = pil_to_numpy_rgb(pil_img2.convert("RGB"))
        lines = ocr_engine.read_text(np_rgb2)
        return normalize_ws("\n".join(lines))

    # OCR by mode
    text_header = ""
    text_totals = ""
    text_full = ""

    if ocr_mode in ("fast", "balanced"):
        header, totals = crop_regions(img)
        text_header = run_ocr_on_pil(header, aggressive=False)
        text_totals = run_ocr_on_pil(totals, aggressive=True if ocr_mode == "balanced" else False)
        debug_blobs["ocr_header"] = text_header
        debug_blobs["ocr_totals"] = text_totals
        combined = normalize_ws(text_header + "\n" + text_totals)

        res = build_invoice_from_text(combined, source="image_ocr_easyocr", file_path=path, debug=debug)

        # fallback to full OCR if low quality
        if allow_fallback_full and not minimal_quality_ok(res):
            text_full = run_ocr_on_pil(img, aggressive=True)
            debug_blobs["ocr_full"] = text_full
            res2 = build_invoice_from_text(text_full, source="image_ocr_easyocr_full", file_path=path, debug=debug)
            if minimal_quality_ok(res2):
                return res2, "image_ocr_easyocr_full", debug_blobs
        return res, "image_ocr_easyocr", debug_blobs

    # accurate/full
    text_full = run_ocr_on_pil(img, aggressive=True)
    debug_blobs["ocr_full"] = text_full
    res = build_invoice_from_text(text_full, source="image_ocr_easyocr_full", file_path=path, debug=debug)
    return res, "image_ocr_easyocr_full", debug_blobs


def parse_pdf_file(path: Path, debug: bool = False) -> Tuple[Optional[InvoiceResult], str, Dict[str, str]]:
    debug_blobs: Dict[str, str] = {}

    text = extract_pdf_text(path)
    if not text.strip():
        return None, "error", {"error": "PDF sin texto extraíble (probablemente escaneado). Implementar OCR de PDF si lo necesitás."}

    debug_blobs["pdf_text"] = text
    res = build_invoice_from_text(text, source="pdf_text", file_path=path, debug=debug)
    return res, "pdf_text", debug_blobs


# ---------------------------
# Gold report (measurement)
# ---------------------------

def load_gold(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError("gold.json debe ser un objeto { 'relative/path': {campos...} }")

def compare_invoice(gold: Dict[str, Any], got: InvoiceResult) -> Dict[str, Any]:
    """
    Compare fields: fecha, rut_emisor, serie, folio, total (num)
    """
    def norm(s):
        if s is None:
            return None
        return str(s).strip()

    out = {"ok": True, "fields": {}}
    checks = [
        ("fecha", norm(gold.get("fecha")), got.fecha),
        ("rut_emisor", norm(gold.get("rut_emisor")), got.rut_emisor),
        ("serie", norm(gold.get("serie")), got.serie),
        ("folio", norm(gold.get("folio")), got.folio),
    ]

    # total: compare floats with tolerance
    gold_total = gold.get("importe_total_con_iva_num", gold.get("total_num", None))
    got_total = got.importe_total_con_iva_num
    total_ok = None
    if gold_total is None or got_total is None:
        total_ok = (gold_total is None and got_total is None)
    else:
        try:
            total_ok = abs(float(gold_total) - float(got_total)) <= 0.02
        except Exception:
            total_ok = False

    for name, g, v in checks:
        ok = (g == v) if g is not None else (v is None)
        out["fields"][name] = {"gold": g, "got": v, "ok": ok}
        if not ok:
            out["ok"] = False

    out["fields"]["importe_total_con_iva_num"] = {"gold": gold_total, "got": got_total, "ok": bool(total_ok)}
    if not total_ok:
        out["ok"] = False

    return out


# ---------------------------
# CLI + main
# ---------------------------

def iter_files(folder: Path) -> Iterable[Path]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def print_debug_block(title: str, text: str) -> None:
    print(f"\n---[{title}]---")
    print(text.strip() if text else "(vacío)")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Carpeta o archivo a procesar")
    ap.add_argument("--json", action="store_true", help="Imprimir salida en JSON (lista)")
    ap.add_argument("--out", default=None, help="Guardar JSON a un archivo (ej: out.json)")
    ap.add_argument("--debug", action="store_true", help="Imprimir debug (OCR chunks, etc.)")
    ap.add_argument("--ocr-mode", default="fast", choices=["fast", "balanced", "full"], help="Modo OCR para imágenes")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Cantidad de threads CPU a usar (recomendado 1-2)")
    ap.add_argument("--max-dim", type=int, default=1600, help="Máximo de ancho/alto (resize) para imágenes")
    ap.add_argument("--max-pixels", type=int, default=2_000_000, help="Máximo total de pixels (resize) para imágenes")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (Windows best-effort)")

    ap.add_argument("--report", action="store_true", help="Comparar contra gold.json y mostrar métricas")
    ap.add_argument("--gold", default="gold.json", help="Ruta a gold.json (para --report)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    configure_cpu_threads(args.cpu_threads)
    if args.low_priority:
        set_low_priority_windows()

    input_path = Path(args.input)

    # OCR engine init (optional)
    ocr_engine = None
    try:
        # Spanish + English improves OCR for mixed tokens
        ocr_engine = EasyOcrEngine(languages=["es", "en"], gpu=False, verbose=False)
    except Exception:
        ocr_engine = None

    results: List[InvoiceResult] = []
    errors: List[Dict[str, Any]] = []

    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(iter_files(input_path))

    if args.report:
        gold_map = load_gold(Path(args.gold))
        total_docs = 0
        ok_docs = 0
        field_ok = {"fecha": 0, "rut_emisor": 0, "serie": 0, "folio": 0, "importe_total_con_iva_num": 0}
        failures = []

        for f in files:
            rel = str(f.relative_to(input_path)) if input_path.is_dir() else f.name
            gold = gold_map.get(rel)
            if gold is None:
                continue

            total_docs += 1
            inv, source, dbg = None, None, {}
            try:
                if f.suffix.lower() == ".pdf":
                    inv, source, dbg = parse_pdf_file(f, debug=args.debug)
                else:
                    inv, source, dbg = parse_image_file(
                        f,
                        ocr_engine=ocr_engine,
                        ocr_mode=args.ocr_mode,
                        max_dim=args.max_dim,
                        max_pixels=args.max_pixels,
                        debug=args.debug,
                        allow_fallback_full=True,
                    )
            except Exception as e:
                inv = None
                errors.append({"file": str(f), "error": str(e)})

            if inv is None:
                failures.append({"file": rel, "ok": False, "reason": "no_result"})
                continue

            comp = compare_invoice(gold, inv)
            if comp["ok"]:
                ok_docs += 1
            else:
                failures.append({"file": rel, "ok": False, "fields": comp["fields"]})

            for k in field_ok.keys():
                if comp["fields"].get(k, {}).get("ok"):
                    field_ok[k] += 1

        # print summary
        print("\n=== REPORT ===")
        print(f"Docs con gold: {total_docs}")
        if total_docs:
            print(f"Docs OK completos: {ok_docs} ({ok_docs/total_docs*100:.1f}%)")
            for k, v in field_ok.items():
                print(f"Campo OK {k}: {v}/{total_docs} ({v/total_docs*100:.1f}%)")
        if failures:
            print("\nFallos (primeros 10):")
            for item in failures[:10]:
                print(json.dumps(item, ensure_ascii=False, indent=2))
        return 0

    # Normal parse
    for f in files:
        try:
            if f.suffix.lower() == ".pdf":
                inv, source, dbg = parse_pdf_file(f, debug=args.debug)
                src_label = "pdf_text"
            else:
                inv, src_label, dbg = parse_image_file(
                    f,
                    ocr_engine=ocr_engine,
                    ocr_mode=args.ocr_mode,
                    max_dim=args.max_dim,
                    max_pixels=args.max_pixels,
                    debug=args.debug,
                    allow_fallback_full=True,
                )

            if args.debug:
                print(f"\n=== {f.name} ({src_label}) ===")
                if "qr" in dbg:
                    print_debug_block("QR", dbg["qr"])
                if "ocr_header" in dbg:
                    print_debug_block("OCR HEADER", dbg["ocr_header"])
                if "ocr_totals" in dbg:
                    print_debug_block("OCR TOTALS", dbg["ocr_totals"])
                if "ocr_full" in dbg:
                    print_debug_block("OCR FULL", dbg["ocr_full"])
                if "pdf_text" in dbg:
                    print_debug_block("PDF TEXT", dbg["pdf_text"])
                print("=== FIN ===\n")

            if inv is None:
                # propagate error details
                err = dbg.get("error") if isinstance(dbg, dict) else "error desconocido"
                errors.append({"file": str(f), "error": err})
                continue

            results.append(inv)

        except Exception as e:
            msg = str(e)
            if args.debug:
                msg += "\n" + traceback.format_exc()
            errors.append({"file": str(f), "error": msg})

    # Attach errors as synthetic results? Keep separate; but for JSON output include them as null objects.
    if errors and not args.debug:
        # keep console cleaner unless debug
        pass
    elif errors and args.debug:
        for er in errors:
            print(f"[ERROR] {er['file']}: {er['error']}")

    if args.json or args.out:
        payload = [dataclasses.asdict(r) for r in results]
        if errors:
            # include errors as separate list at end? Better: print to stderr only.
            pass

        out_str = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(out_str, encoding="utf-8")
        print(out_str)
    else:
        # human-friendly
        for r in results:
            print(f"{Path(r._archivo or '').name}: {r.serie_y_folio} {r.fecha} {r.rut_emisor} {r.importe_total_con_iva}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
