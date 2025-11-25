#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

# Optional deps (OCR / PDF)
# EasyOCR is required for images (and for PDFs without embedded text).
try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:
    fitz = None

try:
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore
except Exception as e:
    raise RuntimeError("Pillow es requerido. Instalá con: pip install pillow") from e


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}


# -----------------------------
# Helpers: text normalization
# -----------------------------
def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def norm(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def only_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


# -----------------------------
# Money parsing/formatting (UY)
# -----------------------------
AMOUNT_RE = re.compile(r"(?<!\d)(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)(?!\d)")

def parse_amount_uy(raw: str) -> Optional[Decimal]:
    """
    Parses amounts like:
      9.261,00  -> 9261.00
      3,129,00  -> try to recover (OCR glitch)
      10742.00  -> 10742.00
      9820      -> 9820.00
    """
    if not raw:
        return None

    s = raw.strip()
    s = s.replace("UYU", "").replace("$", "").replace(" ", "")

    # Fix common OCR weirdness: "3,129,00" -> "3.129,00"
    if re.fullmatch(r"\d{1,3},\d{3},\d{2}", s):
        s = s.replace(",", ".", 1)  # first comma -> dot
        # now pattern is 3.129,00

    # If both separators exist, assume dot thousands, comma decimals
    if "." in s and "," in s:
        s2 = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # Assume comma decimals if last group has 2 digits
        if re.search(r",\d{2}$", s):
            s2 = s.replace(".", "").replace(",", ".")
        else:
            # Maybe comma thousands: 1,234 -> 1234
            s2 = s.replace(",", "")
    elif "." in s:
        # If ends with .dd => decimal point, else thousands
        if re.search(r"\.\d{2}$", s):
            s2 = s.replace(",", "")
        else:
            s2 = s.replace(".", "")
    else:
        s2 = s

    try:
        d = Decimal(s2)
    except InvalidOperation:
        return None

    # Normalize to 2 decimals
    return d.quantize(Decimal("0.01"))


def format_amount_uy(d: Optional[Decimal]) -> Optional[str]:
    if d is None:
        return None
    sign = "-" if d < 0 else ""
    d = abs(d).quantize(Decimal("0.01"))
    s = f"{d:.2f}"  # 9261.00
    int_part, dec_part = s.split(".")
    # thousands with dots
    chunks = []
    while int_part:
        chunks.append(int_part[-3:])
        int_part = int_part[:-3]
    int_fmt = ".".join(reversed(chunks)) if chunks else "0"
    return f"{sign}{int_fmt},{dec_part}"


# -----------------------------
# OCR / PDF reading
# -----------------------------
def preprocess_image(img: Image.Image, max_width: int = 2000) -> Image.Image:
    # Convert to grayscale, increase contrast, slight sharpening-like effect.
    img = ImageOps.exif_transpose(img)  # fix orientation if EXIF exists
    img = img.convert("L")

    # Resize up to a sane width to help OCR, but don't explode RAM.
    if img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, int(img.height * ratio)))

    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.7)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def ocr_image_lines(reader, img: Image.Image) -> List[str]:
    if np is None:
        raise RuntimeError("numpy es requerido para EasyOCR. Instalá con: pip install numpy")
    img2 = preprocess_image(img)
    arr = np.array(img2)
    # detail=0 returns list[str]
    lines = reader.readtext(arr, detail=0, paragraph=False)
    # Normalize whitespace
    return [re.sub(r"\s+", " ", (ln or "")).strip() for ln in lines if (ln or "").strip()]


def read_pdf_text_pages(pdf_path: Path) -> Tuple[str, List[Image.Image]]:
    """
    Returns (extracted_text, rendered_images).
    If PyMuPDF not available -> returns ("", []).
    """
    if fitz is None:
        return ("", [])
    doc = fitz.open(str(pdf_path))
    texts: List[str] = []
    images: List[Image.Image] = []
    for page in doc:
        t = page.get_text("text") or ""
        if t.strip():
            texts.append(t)
        # Always render page as fallback OCR source
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return ("\n".join(texts).strip(), images)


# -----------------------------
# Extraction heuristics
# -----------------------------
RUT_12_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")

def is_credit_note(lines: List[str]) -> bool:
    txt = norm("\n".join(lines))
    return ("nota de credito" in txt) or ("nota de cr" in txt) or ("credit note" in txt)


def extract_date(lines: List[str]) -> Optional[str]:
    # dd/mm/yyyy or dd/mm/yy or yyyy-mm-dd
    date_pat = re.compile(r"\b(\d{2})[/-](\d{2})[/-](\d{2,4})\b")
    iso_pat = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

    candidates: List[Tuple[int, str]] = []

    for i, ln in enumerate(lines):
        ln_norm = norm(ln)
        for m in date_pat.finditer(ln):
            dd, mm, yy = m.groups()
            if len(yy) == 2:
                yy = "20" + yy
            val = f"{dd}/{mm}/{yy}"
            score = 0
            if "fecha" in ln_norm:
                score += 10
            # boost top section a bit
            if i < 25:
                score += 2
            candidates.append((score, val))

        for m in iso_pat.finditer(ln):
            yyyy, mm, dd = m.groups()
            val = f"{dd}/{mm}/{yyyy}"
            score = 0
            if "fecha" in ln_norm:
                score += 10
            if i < 25:
                score += 2
            candidates.append((score, val))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def extract_series_folio(lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (serie, folio).
    Attempts multiple Uruguay-ish formats.
    """
    joined = "\n".join(lines)
    jn = strip_accents(joined)

    # Pattern: "Serie A ... Numero 110897"
    p1 = re.compile(r"\bSerie\b\s*[:\-]?\s*([A-Z])\b[\s\S]{0,80}?\bN[úu]mero\b\s*[:\-]?\s*([0-9]{1,10})\b", re.IGNORECASE)
    m = p1.search(jn)
    if m:
        return (m.group(1).upper(), m.group(2).lstrip("0") or m.group(2))

    # Pattern: "Serie/Numero A,133203" or "A 133203"
    p2 = re.compile(r"\bSerie\s*/\s*N[úu]mero\b\s*[:\-]?\s*([A-Z])\s*[, ]\s*([0-9]{1,10})\b", re.IGNORECASE)
    m = p2.search(jn)
    if m:
        return (m.group(1).upper(), m.group(2).lstrip("0") or m.group(2))

    # Try line-by-line with lookahead
    for i, ln in enumerate(lines):
        ln2 = strip_accents(ln)
        if re.search(r"\bSerie\b", ln2, re.IGNORECASE):
            mm = re.search(r"\bSerie\b\s*[:\-]?\s*([A-Z])\b", ln2, re.IGNORECASE)
            if mm:
                serie = mm.group(1).upper()
                # look for Numero nearby
                for j in range(i, min(i + 6, len(lines))):
                    ln3 = strip_accents(lines[j])
                    mm2 = re.search(r"\bN[úu]mero\b\s*[:\-]?\s*([0-9]{1,10})\b", ln3, re.IGNORECASE)
                    if mm2:
                        folio = mm2.group(1).lstrip("0") or mm2.group(1)
                        return (serie, folio)

    # Fallback: lines containing e-Factura + (Serie letter + digits) or standalone "A 1710173"
    for i, ln in enumerate(lines[:40]):  # top area only
        ln2 = strip_accents(ln)
        if re.search(r"e\s*-?\s*factura|cfe|comprobante", ln2, re.IGNORECASE):
            # e-Factura ... A 1710173
            m = re.search(r"\b([A-Z])\s*[, ]\s*([0-9]{3,10})\b", ln2)
            if m:
                return (m.group(1).upper(), m.group(2).lstrip("0") or m.group(2))
            m = re.search(r"\b([A-Z])\s*[-]\s*([0-9]{3,10})\b", ln2)
            if m:
                return (m.group(1).upper(), m.group(2).lstrip("0") or m.group(2))

    # Another fallback: detect lines that are just: "A" in one line and digits in next
    for i in range(min(60, len(lines) - 1)):
        a = lines[i].strip()
        b = lines[i + 1].strip()
        if len(a) == 1 and a.isalpha() and b.isdigit() and 3 <= len(b) <= 10:
            # only trust if around it appears "serie" / "numero"
            window = " ".join(lines[max(0, i - 2):min(len(lines), i + 4)])
            if re.search(r"serie|numero|n[uú]mero", strip_accents(window), re.IGNORECASE):
                return (a.upper(), b.lstrip("0") or b)

    return (None, None)


def extract_rut_emisor(lines: List[str]) -> Optional[str]:
    """
    Picks the best 12-digit candidate based on context:
    - prefer lines mentioning emisor / rut emisor / ruc emisor
    - avoid comprador / receptor
    """
    candidates: List[Tuple[int, str]] = []

    for i, ln in enumerate(lines[:80]):  # emisor info is usually near top
        ln_raw = ln
        ln2 = strip_accents(ln_raw)
        lnN = norm(ln_raw)

        for m in RUT_12_RE.finditer(only_digits(ln_raw) if re.search(r"\d", ln_raw) else ln_raw):
            rut = m.group(1)
            score = 0
            if "emisor" in lnN or "rut emisor" in lnN or "ruc emisor" in lnN:
                score += 20
            if "rut" in lnN or "ruc" in lnN:
                score += 5
            if i < 20:
                score += 3
            if "comprador" in lnN or "receptor" in lnN or "cliente" in lnN:
                score -= 15
            candidates.append((score, rut))

        # Sometimes OCR splits digits with spaces, so scan whole line digits
        dig = only_digits(ln_raw)
        if len(dig) == 12:
            score = 0
            if "emisor" in lnN:
                score += 20
            if "rut" in lnN or "ruc" in lnN:
                score += 6
            if "comprador" in lnN or "receptor" in lnN:
                score -= 15
            if i < 20:
                score += 3
            candidates.append((score, dig))

    if not candidates:
        return None

    # choose best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    # final sanity: avoid buyer rut that's repeated a lot (if we can detect it)
    # If best appears many times with "comprador", try next.
    def appears_as_comprador(rut: str) -> bool:
        for ln in lines:
            lnN = norm(ln)
            if rut in only_digits(ln) and ("comprador" in lnN or "receptor" in lnN):
                return True
        return False

    if appears_as_comprador(best) and len(candidates) > 1:
        for _, rut in candidates[1:]:
            if not appears_as_comprador(rut):
                return rut
    return best


CORP_HINT_RE = re.compile(r"\b(S\.?A\.?|S\.?R\.?L\.?|SAS|LTDA|SOCIEDAD|TRADING)\b", re.IGNORECASE)

def extract_razon_social(lines: List[str]) -> Optional[str]:
    """
    Best-effort emitter "razón social" from top section.
    Avoids buyer ("MODELO NATURAL", "HOSPITAL", etc.)
    """
    bad_tokens = [
        "modelo natural", "hospital", "rut comprador", "ruc comprador",
        "cliente", "receptor", "direccion", "domicilio", "montevideo",
        "uruguay", "telefono", "tel", "web", "www", "@", "comprobante",
        "e-factura", "cfe", "serie", "numero", "fecha", "moneda", "forma de pago"
    ]

    candidates: List[Tuple[int, str]] = []
    top = lines[:40]

    for i, ln in enumerate(top):
        s = ln.strip()
        if not s:
            continue
        sN = norm(s)

        # Skip obvious noise
        if any(tok in sN for tok in bad_tokens):
            continue
        if len(s) < 3:
            continue
        if sum(ch.isdigit() for ch in s) >= max(3, len(s) // 3):
            continue

        score = 0
        if i < 12:
            score += 4
        if s.isupper():
            score += 2
        if CORP_HINT_RE.search(s):
            score += 8
        # penalize lines that look like a person address fragment (weak)
        if any(w in sN for w in ["av", "calle", "cno", "ruta", "km"]):
            score -= 2

        candidates.append((score, s))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _amount_candidates(lines: List[str]) -> List[Tuple[int, Decimal, str]]:
    """
    Returns list of (score, amount, source_line).
    """
    cands: List[Tuple[int, Decimal, str]] = []
    for idx, ln in enumerate(lines):
        lnN = norm(ln)

        # Extract all amounts in the line, keep last (usually the rightmost is what we want)
        hits = AMOUNT_RE.findall(ln.replace(" ", ""))
        if not hits:
            continue

        # Try parse each, but prefer the last as the "main"
        parsed: List[Tuple[Decimal, str]] = []
        for h in hits:
            d = parse_amount_uy(h)
            if d is not None:
                parsed.append((d, h))
        if not parsed:
            continue

        amount, raw_amount = parsed[-1]

        score = 0
        # Position boost: totals usually near bottom
        if idx > int(len(lines) * 0.60):
            score += 2

        # Keyword scoring
        if "total a pagar" in lnN or "totalapagar" in lnN:
            score += 15
        elif re.search(r"\btotal\b", lnN):
            score += 6

        if "subtotal" in lnN:
            score -= 4
        if "total iva" in lnN or re.search(r"\biva\b", lnN) and "total" in lnN and "pagar" not in lnN:
            score -= 3

        # Currency cue
        if "uyu" in lnN or "$" in ln:
            score += 1

        cands.append((score, amount, ln))
    return cands


def extract_total_con_iva(lines: List[str]) -> Optional[Decimal]:
    cands = _amount_candidates(lines)

    if not cands:
        return None

    # Prefer explicit TOTAL / TOTAL A PAGAR
    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_amt, best_line = cands[0]

    # Safety: if best line looks like "TOTAL IVA", look for next better "TOTAL"
    if "total iva" in norm(best_line) and len(cands) > 1:
        for sc, am, li in cands[1:]:
            if re.search(r"\btotal\b", norm(li)) and "total iva" not in norm(li):
                return am
    return best_amt


def extract_iva_total(lines: List[str]) -> Optional[Decimal]:
    """
    Try to sum Total IVA (10%) + Total IVA (22%), or grab a single IVA total if present.
    """
    iva10 = None
    iva22 = None
    iva_any = None

    for ln in lines:
        lnN = norm(ln)
        if "total iva" in lnN:
            # e.g. "Total IVA (10%): 56,19"
            hits = AMOUNT_RE.findall(ln.replace(" ", ""))
            for h in reversed(hits):
                d = parse_amount_uy(h)
                if d is None:
                    continue
                if "10" in lnN:
                    iva10 = d
                elif "22" in lnN:
                    iva22 = d
                else:
                    iva_any = d

    if iva10 is not None or iva22 is not None:
        return (iva10 or Decimal("0.00")) + (iva22 or Decimal("0.00"))
    return iva_any


def extract_importe_sin_iva(lines: List[str], total_con_iva: Optional[Decimal]) -> Tuple[Optional[Decimal], Optional[str]]:
    """
    Tries:
      - "subtotal gravado (xx%)" / "imponible" / "sub total" / "subtotal"
    If not found but IVA total exists:
      - base ~= total - iva_total
    Returns (amount, source) where source is 'subtotal'|'calculado'|None.
    """
    patterns = [
        r"subtotal\s+gravado",
        r"imponible",
        r"\bsub\s*total\b",
        r"\bsubtotal\b",
    ]

    best: Optional[Tuple[int, Decimal]] = None
    for idx, ln in enumerate(lines):
        lnN = norm(ln)
        if "subtotal no gravado" in lnN:
            continue
        if "subtotal gravado" in lnN or re.search(r"\bsub\s*total\b", lnN) or ("subtotal" in lnN and "total" not in lnN):
            hits = AMOUNT_RE.findall(ln.replace(" ", ""))
            for h in reversed(hits):
                d = parse_amount_uy(h)
                if d is None:
                    continue
                score = 0
                if "subtotal gravado" in lnN:
                    score += 6
                if "imponible" in lnN:
                    score += 5
                if idx > int(len(lines) * 0.50):
                    score += 1
                if best is None or score > best[0]:
                    best = (score, d)

    if best is not None:
        return (best[1], "subtotal")

    iva_total = extract_iva_total(lines)
    if total_con_iva is not None and iva_total is not None:
        base = (total_con_iva - iva_total).quantize(Decimal("0.01"))
        return (base, "calculado")

    return (None, None)


# -----------------------------
# Main processing
# -----------------------------
@dataclass
class Extracted:
    fecha: Optional[str]
    serie: Optional[str]
    folio: Optional[str]
    razon_social: Optional[str]
    rut_emisor: Optional[str]
    es_nota_de_credito: bool
    importe_total_con_iva: Optional[Decimal]
    importe_sin_iva: Optional[Decimal]
    importe_sin_iva_fuente: Optional[str]


def extract_from_lines(lines: List[str]) -> Extracted:
    credit = is_credit_note(lines)
    fecha = extract_date(lines)
    serie, folio = extract_series_folio(lines)
    razon = extract_razon_social(lines)
    rut = extract_rut_emisor(lines)

    total = extract_total_con_iva(lines)
    neto, neto_src = extract_importe_sin_iva(lines, total)

    # Apply credit-note sign convention
    if credit:
        if total is not None and total > 0:
            total = -total
        if neto is not None and neto > 0:
            neto = -neto

    return Extracted(
        fecha=fecha,
        serie=serie,
        folio=folio,
        razon_social=razon,
        rut_emisor=rut,
        es_nota_de_credito=credit,
        importe_total_con_iva=total,
        importe_sin_iva=neto,
        importe_sin_iva_fuente=neto_src,
    )


def build_output(ex: Extracted) -> Dict[str, object]:
    serie_y_folio = None
    if ex.serie and ex.folio:
        serie_y_folio = f"{ex.serie}-{ex.folio}"

    out: Dict[str, object] = {
        "fecha": ex.fecha,
        "serie": ex.serie,
        "folio": ex.folio,
        "serie_y_folio": serie_y_folio,
        "razon_social": ex.razon_social,
        "rut_emisor": ex.rut_emisor,
        "es_nota_de_credito": ex.es_nota_de_credito,
        "importe_total_con_iva": format_amount_uy(ex.importe_total_con_iva),
        "importe_total_con_iva_num": float(ex.importe_total_con_iva) if ex.importe_total_con_iva is not None else None,
        "importe_sin_iva": format_amount_uy(ex.importe_sin_iva),
        "importe_sin_iva_num": float(ex.importe_sin_iva) if ex.importe_sin_iva is not None else None,
        "importe_sin_iva_fuente": ex.importe_sin_iva_fuente,
    }
    return out


def iter_inputs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for child in sorted(pp.rglob("*")):
                if child.suffix.lower() in SUPPORTED_EXTS:
                    out.append(child)
        else:
            out.append(pp)
    # de-dup
    seen = set()
    uniq = []
    for x in out:
        k = str(x.resolve()) if x.exists() else str(x)
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae campos clave de facturas (UY) desde imágenes o PDFs.")
    ap.add_argument("inputs", nargs="+", help="Rutas a imágenes/PDFs o carpetas.")
    ap.add_argument("--json", action="store_true", help="Imprime JSON (una salida por archivo).")
    ap.add_argument("--debug", action="store_true", help="Imprime líneas OCR/texto para inspección.")
    ap.add_argument("--no-ocr", action="store_true", help="Para PDFs: no usar OCR aunque el texto sea pobre (debug).")
    args = ap.parse_args()

    files = iter_inputs(args.inputs)
    if not files:
        print("No encontré archivos para procesar.", file=sys.stderr)
        return 2

    reader = None
    if not args.no_ocr:
        if easyocr is None:
            # OCR only required if we actually need it
            pass
        else:
            # Spanish + English usually works better on URY invoices
            reader = easyocr.Reader(["es", "en"], gpu=False)

    results = []

    for f in files:
        if not f.exists():
            print(f"[WARN] No existe: {f}", file=sys.stderr)
            continue

        lines: List[str] = []
        source = "unknown"

        if f.suffix.lower() == ".pdf":
            pdf_text, rendered = read_pdf_text_pages(f)
            if pdf_text and len(pdf_text.strip()) > 80:
                # Use embedded text
                source = "pdf_text"
                lines = [ln.strip() for ln in pdf_text.splitlines() if ln.strip()]
            else:
                source = "pdf_ocr"
                if args.no_ocr:
                    lines = []
                else:
                    if reader is None:
                        raise RuntimeError("Necesitás OCR para este PDF, pero EasyOCR no está instalado.")
                    for img in rendered:
                        lines.extend(ocr_image_lines(reader, img))
        else:
            source = "image_ocr"
            if reader is None:
                raise RuntimeError("EasyOCR no está instalado. Instalá requirements.txt y reintentá.")
            img = Image.open(f)
            lines = ocr_image_lines(reader, img)

        if args.debug:
            print(f"\n=== {f.name} ({source}) ===")
            for ln in lines[:250]:
                print(ln)
            print("=== FIN ===\n")

        ex = extract_from_lines(lines)
        out = build_output(ex)
        out["_archivo"] = str(f)
        out["_fuente"] = source
        results.append(out)

        if not args.json:
            print(f"\nArchivo: {f}")
            print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.json:
        if len(results) == 1:
            print(json.dumps(results[0], ensure_ascii=False, indent=2))
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
