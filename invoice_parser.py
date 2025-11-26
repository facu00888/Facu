# ruff: noqa: E501
r"""
invoice_parser.py

Extractor simple de metadatos de e-Facturas (UY) desde:
- PDFs con texto (pdfplumber/PyPDF2)
- Imágenes (OCR con EasyOCR si está disponible; sino intenta pytesseract si hay tesseract instalado)

Uso (Windows):
  python invoice_parser.py "C:\Proyectos\Facu\Facturas" --json --debug --backend auto

Salida:
  Lista JSON con campos:
    fecha, serie, folio, serie_y_folio, razon_social, rut_emisor,
    es_nota_de_credito, importe_total_con_iva (+ _num),
    importe_sin_iva (+ _num + _fuente),
    _archivo, _fuente
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------
# Dependencias opcionales
# ---------------------------

pdfplumber_spec = importlib.util.find_spec("pdfplumber")
if pdfplumber_spec:
    import pdfplumber  # type: ignore
else:  # pragma: no cover
    pdfplumber = None  # type: ignore

pypdf2_spec = importlib.util.find_spec("PyPDF2")
if pypdf2_spec:
    from PyPDF2 import PdfReader  # type: ignore
else:  # pragma: no cover
    PdfReader = None  # type: ignore

numpy_spec = importlib.util.find_spec("numpy")
if numpy_spec:
    import numpy as np  # type: ignore
else:  # pragma: no cover
    np = None  # type: ignore

pil_spec = importlib.util.find_spec("PIL.Image")
if pil_spec:
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore
else:  # pragma: no cover
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore

easyocr_spec = importlib.util.find_spec("easyocr")
if easyocr_spec:
    import easyocr  # type: ignore
else:  # pragma: no cover
    easyocr = None  # type: ignore

pytesseract_spec = importlib.util.find_spec("pytesseract")
if pytesseract_spec:
    import pytesseract  # type: ignore
else:  # pragma: no cover
    pytesseract = None  # type: ignore


# ---------------------------
# Modelo de salida
# ---------------------------


@dataclass
class InvoiceResult:
    fecha: Optional[str]
    serie: Optional[str]
    folio: Optional[str]
    serie_y_folio: Optional[str]
    razon_social: Optional[str]
    rut_emisor: Optional[str]
    es_nota_de_credito: bool
    importe_total_con_iva: Optional[str]
    importe_total_con_iva_num: Optional[float]
    importe_sin_iva: Optional[str]
    importe_sin_iva_num: Optional[float]
    importe_sin_iva_fuente: Optional[str]
    _archivo: str
    _fuente: str


# ---------------------------
# Utilidades
# ---------------------------

DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
MONEY_RE = re.compile(r"\b(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))\b")
MONEY_RE_LOOSE = re.compile(r"\b(-?\d{3,})\b")
RUT_RE = re.compile(r"\b(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})\b")


def _safe_upper(s: str | None) -> str:
    return (s or "").upper()


def _collapse_spaces(s: str | None) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())


def normalize_text_block(text: str) -> str:
    """Normaliza saltos de línea y espacios para mejorar las búsquedas."""
    lines = [_collapse_spaces(line) for line in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def parse_money_uy(s: str) -> Optional[float]:
    """
    Convierte:
      "4.919,61" -> 4919.61
      "701,00"   -> 701.0
    """
    if not s:
        return None
    s = s.strip()
    if not re.search(r"\d", s):
        return None
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\.,-]", "", s)
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None


def parse_money_uy_loose(s: str, *, allow_short: bool = False) -> Optional[float]:
    """
    Interpreta montos sin separadores (fallback OCR sucio):
      "401247" -> 4012.47
      "-11813" -> -118.13

    Por defecto ignora cadenas de menos de 4 dígitos para no capturar años.
    Si allow_short=True, acepta 3 dígitos (p.ej. IVA leido como 224).
    """
    if not s:
        return None
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if len(digits) < 4 and not allow_short:
        return None
    if len(digits) < 3:
        return None
    whole = digits[:-2] or "0"
    frac = digits[-2:]
    try:
        v = float(f"{int(whole)}.{int(frac):02d}")
        return -v if neg else v
    except Exception:
        return None


def parse_money_with_fallback(s: str, *, allow_short: bool = False) -> Optional[float]:
    """Primero intenta el formato UY estándar, luego sin separadores."""
    return parse_money_uy(s) or parse_money_uy_loose(s, allow_short=allow_short)


def format_money_uy(v: Optional[float]) -> Optional[str]:
    """
    4919.61 -> "4.919,61"
    """
    if v is None:
        return None
    v = round(float(v), 2)
    neg = v < 0
    v = abs(v)
    whole = int(v)
    frac = int(round((v - whole) * 100))
    whole_str = f"{whole:,}".replace(",", ".")
    out = f"{whole_str},{frac:02d}"
    return f"-{out}" if neg else out


def derive_razon_social_from_filename(path: Path) -> Optional[str]:
    """
    Usa el nombre del archivo como fuente "fuerte" para razón social:
      "BIMBO A3519972 CREDITO.jpeg" -> "BIMBO"
      "DEL SUR NOTA DE CREDITO A18911 CREDITO.jpeg" -> "DEL SUR"
    """
    stem = _safe_upper(path.stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = _collapse_spaces(stem)
    stem = re.split(r"\s+A\d+\b", stem)[0]
    stem = stem.replace("NOTA DE CREDITO", "").replace("NOTA DE CRÉDITO", "").strip()
    stem = re.sub(r"\s+CREDITO\b", "", stem).strip()
    stem = _collapse_spaces(stem)
    return stem or None


def is_credit_note(text: str, path: Path) -> bool:
    x = _safe_upper(text) + " " + _safe_upper(path.name)
    return ("NOTA DE CREDITO" in x) or ("NOTA DE CRÉDITO" in x)


def pick_best_date(text: str) -> Optional[str]:
    if not text:
        return None

    best = None
    best_score = -10_000.0
    for m in DATE_RE.finditer(text):
        d = m.group(1)
        i = m.start()
        ctx = _safe_upper(text[max(0, i - 40) : i + 40])

        score = 0.0
        if "FECHA DE DOCUMENTO" in ctx:
            score += 12.0
        elif re.search(r"\bFECHA\b", ctx):
            score += 6.0

        # Castigo MUCHO mayor a fechas asociadas a vencimiento
        if "VENC" in ctx or "VTO" in ctx:
            score -= 25.0
        if "CAE" in ctx:
            score -= 8.0

        # Preferir fechas al inicio del documento
        score += max(0, 2500 - i) / 2500.0

        if score > best_score:
            best_score = score
            best = d

    return best


def _clean_rut(raw: str) -> str:
    return re.sub(r"[^0-9]", "", raw)


def extract_rut_emisor(text: str) -> Optional[str]:
    if not text:
        return None

    up = _safe_upper(text)

    m = re.search(r"\bRU[TC]\s*EMISOR\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = re.search(r"\bRUC\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = RUT_RE.search(up)
    if m:
        return _clean_rut(m.group(1))

    return None


def _looks_like_rut(num: str) -> bool:
    digits = re.sub(r"\D", "", num)
    return len(digits) >= 9  # RUT / cédulas / cosas largas; no usar como folio


def extract_serie_folio_from_filename(path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Extrae serie/folio desde el nombre del archivo:
      "BIMBO A3519972 CREDITO.jpeg" -> ("A", "3519972")
    """
    stem = _safe_upper(path.stem)
    m = re.search(r"\b([A-Z])\s*0*([0-9]{3,8})\b", stem)
    if not m:
        return None, None
    serie = m.group(1).upper()
    folio_digits = re.sub(r"\D", "", m.group(2))
    if not (3 <= len(folio_digits) <= 8):
        return None, None
    folio = folio_digits.lstrip("0") or "0"
    return serie, folio


def extract_serie_folio(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Intenta detectar SERIE / NUMERO evitando confundir RUTs, CAE, etc.
    Solo acepta folios de 3 a 8 dígitos.
    """
    if not text:
        return None, None

    patterns = [
        # SERIE <A> ... NUMERO <1234567>
        r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMERO\b[^0-9]{0,20}(\d{3,})",
        # SERIE <A> <1234567>
        r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\s*([0-9]{3,})",
        # NUMERO <A> <1234567>
        r"\bNUMERO\b[^A-Z0-9]{0,20}([A-Z])\s*([0-9]{3,})",
        # fallback genérico: A-1234567
        r"\b([A-Z])\s*-?\s*(\d{3,})\b",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if not m:
            continue
        serie = m.group(1).upper()
        folio_raw = m.group(2)
        digits = re.sub(r"\D", "", folio_raw)
        # Evitar folios absurdos (RUTs, CAE largos, etc.)
        if not (3 <= len(digits) <= 8):
            continue
        folio = digits.lstrip("0") or "0"
        return serie, folio

    return None, None


def extract_iva_total(text: str) -> Optional[float]:
    """
    Busca IVA total o suma IVA 10% + IVA 22%.
    Tolera OCR sucio (valores sin separadores y "22%" leídos como "224").
    """
    if not text:
        return None
    up = _safe_upper(text)

    money_pat = MONEY_RE.pattern + "|" + MONEY_RE_LOOSE.pattern

    m = re.search(r"\bTOTAL\s*IVA\b[^0-9\-]{0,20}((?:" + money_pat + r"))", up)
    if m:
        return parse_money_with_fallback(m.group(1), allow_short=True)

    iva10 = None
    iva22 = None

    def _money_after(pattern: str) -> Optional[float]:
        m_local = re.search(pattern, up)
        if m_local:
            return parse_money_with_fallback(m_local.group(1), allow_short=True)
        return None

    iva10 = _money_after(r"\bTOTAL\s*IVA\b[^0-9]{0,40}10\D{0,5}(" + money_pat + r")")
    iva22 = _money_after(r"\bTOTAL\s*IVA\b[^0-9]{0,40}22\d?\D{0,5}(" + money_pat + r")")

    if iva10 is None:
        iva10 = _money_after(r"\bI\.?V\.?A\.?\s*10\d?\D{0,15}(" + money_pat + r")")

    if iva22 is None:
        iva22 = _money_after(r"\bI\.?V\.?A\.?\s*22\d?\D{0,15}(" + money_pat + r")")

    if iva10 is None and iva22 is None:
        return None

    return (iva10 or 0.0) + (iva22 or 0.0)


def extract_total(text: str) -> Optional[float]:
    if not text:
        return None

    candidates: list[tuple[float, int]] = []

    money_pat = MONEY_RE.pattern + "|" + MONEY_RE_LOOSE.pattern

    def _collect(pattern: str, score: int) -> None:
        for m in re.finditer(pattern, text, flags=re.I):
            v = parse_money_with_fallback(m.group(1))
            if v is not None:
                candidates.append((v, score))

    # Total a pagar / Total:
    _collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,25}((?:" + money_pat + r"))", 100)
    _collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,25}((?:" + money_pat + r"))", 70)
    _collect(r"\bTOTAL\s*:\s*\$?\s*((?:" + money_pat + r"))", 75)

    if candidates:
        candidates.sort(key=lambda t: (t[1], t[0]), reverse=True)
        return candidates[0][0]

    vals = [parse_money_with_fallback(m.group(1)) for m in MONEY_RE.finditer(text)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals)


def extract_subtotal(text: str) -> Optional[float]:
    """
    Detecta importe sin IVA a partir de etiquetas comunes.
    Retorna el candidato POSITIVO más grande (es el que suele representar
    el subtotal gravado principal).
    """
    if not text:
        return None

    money_pat = MONEY_RE.pattern + "|" + MONEY_RE_LOOSE.pattern
    labels = [
        r"SUBTOTAL",
        r"TOTAL\s+SIN\s+IVA",
        r"IMPORTE\s+NETO",
        r"NETO\s+GRAVADO",
        r"TOTAL\s+EXENTO",
    ]

    candidates: list[float] = []

    for label in labels:
        pattern = rf"\b{label}\b[^0-9\-]{{0,40}}((?:{money_pat}))"
        for m in re.finditer(pattern, text, flags=re.I):
            v = parse_money_with_fallback(m.group(1))
            if v is not None and v > 0:
                candidates.append(v)

    if not candidates:
        return None
    return max(candidates)


# ---------------------------
# OCR / Extractores de texto
# ---------------------------


class OCRBackend:
    def __init__(self, preferred_backend: str = "auto") -> None:
        self._easy_reader = None
        self.preferred_backend = preferred_backend

    def has_easyocr(self) -> bool:
        return easyocr is not None and np is not None

    def has_tesseract(self) -> bool:
        return pytesseract is not None and shutil.which("tesseract") is not None and Image is not None

    def _get_easy_reader(self):
        if self._easy_reader is None:
            self._easy_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)  # type: ignore
        return self._easy_reader

    def _preprocess_image(self, img):
        img = ImageOps.exif_transpose(img)  # type: ignore
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
        return img

    def ocr_image_easy(self, img) -> str:
        reader = self._get_easy_reader()
        arr = np.array(img.convert("RGB"))  # type: ignore
        lines = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)

    def ocr_image_tess(self, img) -> str:
        cfg = "--oem 3 --psm 6"
        lang = "spa"
        try:
            return pytesseract.image_to_string(img, lang=lang, config=cfg)  # type: ignore
        except Exception:
            return pytesseract.image_to_string(img, lang="eng", config=cfg)  # type: ignore

    def _iter_backends(self) -> list[str]:
        available: list[str] = []
        if self.has_easyocr():
            available.append("easyocr")
        if self.has_tesseract():
            available.append("tesseract")

        if self.preferred_backend in available:
            return [self.preferred_backend] + [b for b in available if b != self.preferred_backend]
        return available

    def ocr_image(self, img) -> tuple[str, str]:
        if Image is None:
            raise RuntimeError("Falta pillow (PIL) para OCR de imágenes.")

        img = self._preprocess_image(img)
        errors: list[str] = []

        for backend in self._iter_backends():
            try:
                if backend == "easyocr":
                    txt = self.ocr_image_easy(img)
                else:
                    txt = self.ocr_image_tess(img)
                normalized = normalize_text_block(txt)
                if normalized:
                    return normalized, f"image_ocr_{backend}"
            except Exception as exc:  # pragma: no cover
                errors.append(f"{backend}: {exc}")

        if errors:
            raise RuntimeError("; ".join(errors))

        raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract.")


def crop_rel(img, l: float, t: float, r: float, b: float):
    w, h = img.size
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def extract_text_from_pdf(path: Path) -> tuple[str, str]:
    if pdfplumber is not None:
        try:
            chunks: list[str] = []
            with pdfplumber.open(str(path)) as pdf:  # type: ignore
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        chunks.append(txt)
            out = normalize_text_block("\n".join(chunks).strip())
            if out:
                return out, "pdf_text"
        except Exception:
            pass

    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))  # type: ignore
            chunks = []
            for p in reader.pages:
                txt = (p.extract_text() or "").strip()
                if txt:
                    chunks.append(txt)
            out = normalize_text_block("\n".join(chunks).strip())
            if out:
                return out, "pdf_text"
        except Exception:
            pass

    return "", "pdf_text"


# ---------------------------
# Parse documento
# ---------------------------


def parse_invoice_from_text(
    text_full: str,
    text_header: str,
    text_totals: str,
    path: Path,
    fuente: str,
) -> InvoiceResult:
    razon = derive_razon_social_from_filename(path)
    es_nc = is_credit_note(text_full, path)

    # Serie / folio desde texto
    serie, folio = extract_serie_folio(text_header or text_full)

    # Fallback: si quedó vacío o folio parece un RUT, probar nombre de archivo
    if not serie or not folio or _looks_like_rut(folio):
        serie2, folio2 = extract_serie_folio_from_filename(path)
        if serie2 and folio2:
            serie, folio = serie2, folio2

    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_emisor = extract_rut_emisor(text_header or text_full)
    fecha = pick_best_date(text_header or text_full)

    subtotal_label = extract_subtotal(text_totals) or extract_subtotal(text_full)
    total_num = extract_total(text_totals) or extract_total(text_full)
    iva_total_raw = extract_iva_total(text_totals) or extract_iva_total(text_full)

    # Lógica para confiar (o no) en los IVA/subtotales
    iva_total: Optional[float] = None
    sin_iva_num: Optional[float] = None
    sin_iva_fuente: Optional[str] = None

    if total_num is not None:
        # Caso 1: tenemos subtotal etiquetado + IVA
        if subtotal_label is not None and iva_total_raw is not None:
            if 0 < subtotal_label < total_num and 0 < iva_total_raw < total_num:
                if abs((subtotal_label + iva_total_raw) - total_num) <= max(0.05, total_num * 0.01):
                    # Cierra razonablemente -> confiable
                    iva_total = round(iva_total_raw, 2)
                    sin_iva_num = round(subtotal_label, 2)
                    sin_iva_fuente = "subtotal_label"

        # Caso 2: no confiamos en subtotal, pero el IVA aislado parece coherente
        if iva_total is None and iva_total_raw is not None:
            if 0 < iva_total_raw < total_num:
                base = total_num - iva_total_raw
                if base > 0:
                    ratio = iva_total_raw / base
                    # Rango razonable de IVA efectivo (5% a 30%)
                    if 0.05 <= ratio <= 0.30:
                        iva_total = round(iva_total_raw, 2)
                        sin_iva_num = round(base, 2)
                        sin_iva_fuente = "total_menos_iva"

    # Si nada fue confiable, dejamos sin_iva como None
    total_str = format_money_uy(total_num)
    sin_iva_str = format_money_uy(sin_iva_num)

    return InvoiceResult(
        fecha=fecha,
        serie=serie,
        folio=folio,
        serie_y_folio=serie_y_folio,
        razon_social=razon,
        rut_emisor=rut_emisor,
        es_nota_de_credito=es_nc,
        importe_total_con_iva=total_str,
        importe_total_con_iva_num=total_num,
        importe_sin_iva=sin_iva_str,
        importe_sin_iva_num=sin_iva_num,
        importe_sin_iva_fuente=sin_iva_fuente,
        _archivo=str(path),
        _fuente=fuente,
    )


def process_image(path: Path, ocr: OCRBackend, debug: bool) -> tuple[str, str, str, str]:
    if Image is None:
        raise RuntimeError("Falta pillow (PIL) para leer imágenes.")
    img = Image.open(path)

    # Header: parte superior; totales: esquina inferior derecha (heurístico)
    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.38)
    totals_img = crop_rel(img, 0.50, 0.60, 1.00, 0.95)

    text_full, fuente = ocr.ocr_image(img)
    text_header, _ = ocr.ocr_image(header_img)
    text_totals, _ = ocr.ocr_image(totals_img)

    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print(_collapse_spaces(text_full))
        print("\n---[OCR HEADER]---")
        print(_collapse_spaces(text_header))
        print("\n---[OCR TOTALS]---")
        print(_collapse_spaces(text_totals))
        print("=== FIN ===\n")

    return text_full, text_header, text_totals, fuente


def process_pdf(path: Path, debug: bool) -> tuple[str, str, str, str]:
    text_full, fuente = extract_text_from_pdf(path)
    text_header = text_full
    text_totals = text_full

    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print(_collapse_spaces(text_full))
        print("=== FIN ===\n")

    return text_full, text_header, text_totals, fuente


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return

    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def parse_path(root: Path, as_json: bool, debug: bool, preferred_backend: str) -> list[dict[str, Any]]:
    if not root.exists():
        print(f"[WARN] No existe: {root}")
        return []

    ocr = OCRBackend(preferred_backend=preferred_backend)
    results: list[InvoiceResult] = []

    for path in sorted(iter_files(root)):
        try:
            if path.suffix.lower() == ".pdf":
                text_full, text_header, text_totals, fuente = process_pdf(path, debug=debug)
            else:
                text_full, text_header, text_totals, fuente = process_image(path, ocr=ocr, debug=debug)

            res = parse_invoice_from_text(
                text_full=text_full,
                text_header=text_header,
                text_totals=text_totals,
                path=path,
                fuente=fuente,
            )
            results.append(res)
        except Exception as e:
            if debug:
                print(f"[ERROR] {path.name}: {e}")
            # Devolvemos al menos razón social y archivo
            results.append(
                InvoiceResult(
                    fecha=None,
                    serie=None,
                    folio=None,
                    serie_y_folio=None,
                    razon_social=derive_razon_social_from_filename(path),
                    rut_emisor=None,
                    es_nota_de_credito=is_credit_note("", path),
                    importe_total_con_iva=None,
                    importe_total_con_iva_num=None,
                    importe_sin_iva=None,
                    importe_sin_iva_num=None,
                    importe_sin_iva_fuente=None,
                    _archivo=str(path),
                    _fuente="error",
                )
            )

    out = [asdict(r) for r in results]

    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for r in out:
            print(
                f"{Path(r['_archivo']).name} | {r.get('fecha')} | {r.get('serie_y_folio')} | "
                f"{r.get('razon_social')} | Total: {r.get('importe_total_con_iva')}"
            )

    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Archivo o carpeta con facturas (pdf/jpg/png/etc.)")
    ap.add_argument("--json", action="store_true", help="Imprimir salida JSON (lista)")
    ap.add_argument("--debug", action="store_true", help="Imprimir texto extraído por OCR/PDF")
    ap.add_argument(
        "--backend",
        choices=["auto", "easyocr", "tesseract"],
        default="auto",
        help="Forzar backend OCR (por defecto el primero disponible)",
    )
    args = ap.parse_args(argv)

    parse_path(Path(args.path), as_json=args.json, debug=args.debug, preferred_backend=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
