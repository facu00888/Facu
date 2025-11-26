# ruff: noqa: E501
"""
invoice_parser.py

Extractor simple de metadatos de e-Facturas (UY) desde:
- PDFs con texto (pdfplumber/PyPDF2)
- Imágenes (OCR: EasyOCR si está disponible; sino pytesseract si hay tesseract instalado)

Uso (Windows):
  python invoice_parser.py "C:\\Proyectos\\Facu\\Facturas" --json --debug --backend auto

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
# Regex / Constantes
# ---------------------------

DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

# Montos UY con separadores "." o espacios, y decimal con ","
# Ej: "4.919,61" | "4 018,01" | "701,00"
MONEY_RE = re.compile(r"\b(-?\d{1,3}(?:[.\s]\d{3})*(?:,\d{2,})|-?\d+(?:,\d{2,}))\b")

# Dígitos sueltos (OCR pegado). Ojo: se usa SOLO en contextos con etiqueta (TOTAL/IVA/etc.)
MONEY_RE_LOOSE_ANY = re.compile(r"\b(-?\d{4,})\b")

# RUT / RUC: en UY suele ser 12 dígitos (a veces 11 en OCR), con puntos/guiones opcionales
RUT_RE = re.compile(r"\b(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})\b")

FOLIO_MIN_LEN = 3
FOLIO_MAX_LEN = 9  # >9 normalmente ya es un RUT receptor o un código raro

TOL_SUBTOTAL = 0.10  # tolerancia en pesos para validar subtotal + iva = total


# ---------------------------
# Utilidades
# ---------------------------

def _safe_upper(s: str) -> str:
    return (s or "").upper()


def _collapse_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())


def normalize_text_block(text: str) -> str:
    """Normaliza saltos de línea y espacios para mejorar las búsquedas."""
    lines = [_collapse_spaces(line) for line in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def normalize_ocr_dates(text: str) -> str:
    """
    Corrige formatos comunes de OCR:
      - 12/112025  -> 12/11/2025
      - 110/2025   -> 1/10/2025
      - 1210/2025  -> 12/10/2025
    """
    if not text:
        return text

    # dd/mmYYYY -> dd/mm/YYYY
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})(\d{4})\b", r"\1/\2/\3", text)

    # ddmm/YYYY -> dd/mm/YYYY
    text = re.sub(r"\b(\d{1,2})(\d{2})/(\d{4})\b", r"\1/\2/\3", text)

    return text


def _date_key(d: str) -> tuple[int, int, int]:
    dd, mm, yyyy = d.split("/")
    return (int(yyyy), int(mm), int(dd))


def parse_money_uy(s: str) -> Optional[float]:
    """
    Convierte:
      "4.919,61" -> 4919.61
      "4 018,01" -> 4018.01
      "701,00"   -> 701.0
    Tolera decimales OCR con 3+ dígitos: usa los últimos 2 como centavos.
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
        # quitar miles
        s = s.replace(".", "")
        whole, frac = s.split(",", 1)
        digits = re.sub(r"\D", "", whole + frac)
        if len(digits) >= 3:
            # interpretá últimos 2 como centavos
            whole_d = digits[:-2] or "0"
            frac_d = digits[-2:]
            try:
                v = float(f"{int(whole_d)}.{int(frac_d):02d}")
                return -v if neg else v
            except Exception:
                return None
        return None

    # No coma: probablemente no es dinero (o es OCR sucio). Lo dejamos a parse_money_uy_loose.
    return None


def parse_money_uy_loose(s: str, *, allow_short: bool = False) -> Optional[float]:
    """
    Fallback para OCR sucio sin separadores:
      "401247" -> 4012.47
      "-11813" -> -118.13

    Por defecto se ignoran cadenas cortas para no capturar años.
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


def parse_money_context(raw: str, *, allow_loose: bool) -> Optional[float]:
    v = parse_money_uy(raw)
    if v is not None:
        return v
    if allow_loose:
        return parse_money_uy_loose(raw)
    return None


def format_money_uy(v: Optional[float]) -> Optional[str]:
    """4919.61 -> '4.919,61'"""
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
    Usa el nombre del archivo como señal fuerte:
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


def _clean_rut(raw: str) -> str:
    return re.sub(r"[^0-9]", "", raw)


# ---------------------------
# Extractores
# ---------------------------

def extract_serie_folio(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    # Normalizar OCR raro
    t = normalize_ocr_dates(text)

    # Caso PDF típico: "SERIE NUMERO ... A 049009"
    m = re.search(r"\bSERIE\b.{0,30}\bNUMERO\b.{0,40}\b([A-Z])\s*0*([0-9]{3,})\b", t, flags=re.I | re.S)
    if m:
        serie = m.group(1).upper()
        folio_raw = m.group(2)
        if FOLIO_MIN_LEN <= len(folio_raw) <= FOLIO_MAX_LEN:
            folio = folio_raw.lstrip("0") or "0"
            return serie, folio

    # Caso OCR: "SERIE ... A 3519972" o "NUMERO ... A 3519972"
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,25}([A-Z])\b.{0,40}\b(?:NUMERO|N[ÚU]MERO)\b[^0-9]{0,25}0*([0-9]{3,})\b", t, flags=re.I | re.S)
    if m:
        serie = m.group(1).upper()
        folio_raw = m.group(2)
        if FOLIO_MIN_LEN <= len(folio_raw) <= FOLIO_MAX_LEN:
            folio = folio_raw.lstrip("0") or "0"
            return serie, folio

    # Caso simple "A-3519972" / "A 3519972"
    for m in re.finditer(r"\b([A-Z])\s*[- ]\s*0*([0-9]{3,})\b", t, flags=re.I):
        serie = m.group(1).upper()
        folio_raw = m.group(2)
        # Evitá capturar RUT receptor tipo "A 218849400010"
        if len(folio_raw) > FOLIO_MAX_LEN:
            continue
        folio = folio_raw.lstrip("0") or "0"
        return serie, folio

    return None, None


def extract_rut_emisor(text: str) -> Optional[str]:
    if not text:
        return None

    up = _safe_upper(text)

    # Preferencias explícitas
    m = re.search(r"\bRUT\s*EMISOR\b[^0-9]{0,40}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = re.search(r"\bRUC\s*EMISOR\b[^0-9]{0,40}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    # Muchos OCR ponen "RUC <emisor>" arriba y luego "RUC COMPRADOR <receptor>"
    # Tomamos el PRIMER "RUC <numero>" que aparezca.
    m = re.search(r"\bRUC\b[^0-9]{0,20}(\d{11,12})\b", up)
    if m:
        return _clean_rut(m.group(1))

    # Último recurso: primer RUT/RUC que aparezca
    m = RUT_RE.search(up)
    if m:
        return _clean_rut(m.group(1))

    return None


def extract_fecha_documento(text: str) -> Optional[str]:
    if not text:
        return None

    up = _safe_upper(normalize_ocr_dates(text))

    # Intento directo: "FECHA DE DOCUMENTO"
    m = re.search(r"\bFECHA\s+DE\s+DOCUMENTO\b[^0-9]{0,40}(\d{1,2}/\d{1,2}/\d{4})", up)
    if m:
        return m.group(1)

    # Caso común: "FECHA ... VENCIMIENTO ... <fecha_doc> <fecha_vto>"
    m = re.search(
        r"\bFECHA\b.{0,80}\bVENC(?:IMIENTO)?\b.{0,120}(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})",
        up,
        flags=re.S,
    )
    if m:
        d1, d2 = m.group(1), m.group(2)
        return d1 if _date_key(d1) <= _date_key(d2) else d2

    # Heurística general por puntaje, castigando VENC/CAE fuerte
    best = None
    best_score = -10_000.0
    for dm in DATE_RE.finditer(up):
        d = dm.group(1)
        i = dm.start()
        ctx = up[max(0, i - 70) : i + 70]

        score = 0.0
        if "FECHA DE DOCUMENTO" in ctx:
            score += 12.0
        elif re.search(r"\bFECHA\b", ctx):
            score += 6.0

        if "VENC" in ctx or "VTO" in ctx:
            score -= 30.0
        if "CAE" in ctx:
            score -= 14.0

        # preferir lo que aparece arriba
        score += max(0, 2500 - i) / 2500.0

        if score > best_score:
            best_score = score
            best = d

    return best


def extract_total(text: str) -> Optional[float]:
    if not text:
        return None

    t = normalize_ocr_dates(text)

    # En patrones con etiqueta, permitimos "loose"
    def _collect(pattern: str, score: int, allow_loose: bool) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        for m in re.finditer(pattern, t, flags=re.I | re.S):
            raw = m.group(1)
            v = parse_money_context(raw, allow_loose=allow_loose)
            if v is not None:
                out.append((v, score))
        return out

    candidates: list[tuple[float, int]] = []
    money_pat = r"(" + MONEY_RE.pattern.strip(r"\b") + r"|" + MONEY_RE_LOOSE_ANY.pattern.strip(r"\b") + r")"

    candidates += _collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,40}" + money_pat, 120, True)
    candidates += _collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,40}" + money_pat, 90, True)
    candidates += _collect(r"\bTOTAL\s*:\s*\$?\s*" + money_pat, 95, True)

    if candidates:
        # mayor score, y como desempate el valor (en general el total es alto)
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return candidates[0][0]

    # Fallback MUY conservador: máximo de montos "bien formados", sin loose
    vals = []
    for m in MONEY_RE.finditer(t):
        v = parse_money_context(m.group(1), allow_loose=False)
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def _extract_iva_rate(text: str, rate: int) -> Optional[float]:
    """
    Busca "Total iva (22%) 231,18" o variantes OCR.
    """
    if not text:
        return None

    up = _safe_upper(normalize_ocr_dates(text))
    money_pat = r"(" + MONEY_RE.pattern.strip(r"\b") + r"|" + MONEY_RE_LOOSE_ANY.pattern.strip(r"\b") + r")"

    if rate == 10:
        rate_pat = r"(?:10|10%|10\*|101|104)"
    else:  # 22
        rate_pat = r"(?:22|22%|22\*|221|224)"

    # Total iva (22%) <monto>
    patterns = [
        rf"\bTOTAL\s*IVA\b.{0,80}\({0,1}\s*{rate_pat}\s*%{0,1}\){0,1}.{0,30}{money_pat}",
        rf"\bTOTAL\s*IVA\b.{0,80}\b{rate_pat}\b.{0,30}{money_pat}",
        rf"\bI\.?V\.?A\.?\b.{0,60}\b{rate_pat}\b.{0,30}{money_pat}",
    ]

    best: Optional[float] = None
    for pat in patterns:
        for m in re.finditer(pat, up, flags=re.S):
            v = parse_money_context(m.group(1), allow_loose=True)
            if v is None:
                continue
            # normalmente IVA no es enorme; si viene un disparate, ignorar
            if v < 0:
                continue
            best = v if best is None else max(best, v)
    return best


def extract_iva_total(text: str) -> Optional[float]:
    """
    Busca IVA total o suma IVA 10% + IVA 22%.
    Importante: NO se queda con el primer "Total iva (10%) 0,00" si existe el 22%.
    """
    if not text:
        return None

    iva10 = _extract_iva_rate(text, 10)
    iva22 = _extract_iva_rate(text, 22)

    if iva10 is not None or iva22 is not None:
        return (iva10 or 0.0) + (iva22 or 0.0)

    # Último recurso: "TOTAL IVA <monto>"
    up = _safe_upper(normalize_ocr_dates(text))
    money_pat = r"(" + MONEY_RE.pattern.strip(r"\b") + r"|" + MONEY_RE_LOOSE_ANY.pattern.strip(r"\b") + r")"
    m = re.search(rf"\bTOTAL\s*IVA\b[^0-9\-]{{0,40}}{money_pat}", up)
    if m:
        return parse_money_context(m.group(1), allow_loose=True)

    return None


def extract_subtotal_candidates(text: str) -> list[float]:
    """
    Detecta posibles "importe sin IVA" a partir de etiquetas comunes.
    OJO: después se validan contra total/iva, porque OCR/PDF a veces miente.
    """
    if not text:
        return []

    t = normalize_ocr_dates(text)
    money_pat = r"(" + MONEY_RE.pattern.strip(r"\b") + r"|" + MONEY_RE_LOOSE_ANY.pattern.strip(r"\b") + r")"
    labels = [
        r"SUBTOTAL",
        r"TOTAL\s+SIN\s+IVA",
        r"IMPORTE\s+NETO",
        r"NETO\s+GRAVADO",
        r"SUBTOTAL\s+GRAVADO",
    ]

    out: list[float] = []
    for label in labels:
        pattern = rf"\b{label}\b[^0-9\-]{{0,60}}{money_pat}"
        for m in re.finditer(pattern, t, flags=re.I | re.S):
            v = parse_money_context(m.group(1), allow_loose=True)
            if v is not None and v >= 0:
                out.append(v)

    return out


def _validate_subtotal(sub: float, total: Optional[float], iva: Optional[float]) -> bool:
    if total is None or iva is None:
        return False
    if sub < 0 or iva < 0 or total <= 0:
        return False
    return abs((sub + iva) - total) <= TOL_SUBTOTAL


def compute_importe_sin_iva(total: Optional[float], iva_total: Optional[float], text_hint: str) -> tuple[Optional[float], Optional[str]]:
    """
    Prioridad:
      1) total - iva_total (si iva parece razonable)
      2) si no hay iva, inferir por tasa dominante (1.22 o 1.10)
    """
    if total is None:
        return None, None

    up = _safe_upper(text_hint or "")

    if iva_total is not None and 0 <= iva_total < total:
        base = total - iva_total
        # relación IVA/base razonable (acepta 0 por exento)
        if base >= 0 and (base == 0 or (iva_total / max(base, 0.01)) <= 0.45):
            return round(base, 2), "total_menos_iva"

    # Inferir tasa (si aparece 22% o 10% por texto)
    if re.search(r"\b(22%|IVA.{0,15}22|SUBTOTAL.{0,30}22)\b", up):
        return round(total / 1.22, 2), "total_div_22"
    if re.search(r"\b(10%|IVA.{0,15}10|SUBTOTAL.{0,30}10)\b", up):
        return round(total / 1.10, 2), "total_div_10"

    # Default razonable en UY (muchas ventas van a 22)
    return round(total / 1.22, 2), "total_div_22"


# ---------------------------
# OCR / Extractores de texto
# ---------------------------

class OCRBackend:
    def __init__(self, preferred_backend: str = "auto") -> None:
        self._easy_reader = None
        self.preferred_backend = preferred_backend

    def has_easyocr(self) -> bool:
        return easyocr is not None and np is not None and Image is not None

    def has_tesseract(self) -> bool:
        return pytesseract is not None and shutil.which("tesseract") is not None and Image is not None

    def _get_easy_reader(self):
        if self._easy_reader is None:
            self._easy_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)  # type: ignore
        return self._easy_reader

    def _preprocess_image(self, img):
        img = ImageOps.exif_transpose(img)  # type: ignore
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.2)
        img = ImageEnhance.Sharpness(img).enhance(1.6)
        # Upscale suave para OCR
        w, h = img.size
        img = img.resize((int(w * 1.6), int(h * 1.6)))
        return img

    def ocr_image_easy(self, img) -> str:
        reader = self._get_easy_reader()
        arr = np.array(img.convert("RGB"))  # type: ignore
        lines = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)

    def ocr_image_tess(self, img) -> str:
        cfg = "--oem 3 --psm 6"
        # spa suele andar; si falla, cae a eng
        try:
            return pytesseract.image_to_string(img, lang="spa", config=cfg)  # type: ignore
        except Exception:
            return pytesseract.image_to_string(img, lang="eng", config=cfg)  # type: ignore

    def _iter_backends(self) -> list[str]:
        available = []
        if self.has_easyocr():
            available.append("easyocr")
        if self.has_tesseract():
            available.append("tesseract")

        if self.preferred_backend in available:
            return [self.preferred_backend] + [b for b in available if b != self.preferred_backend]
        if self.preferred_backend == "auto":
            return available

        # si pidió uno que no está, igual probá los disponibles
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

                normalized = normalize_text_block(normalize_ocr_dates(txt))
                if normalized:
                    return normalized, f"image_ocr_{backend}"
            except Exception as exc:  # pragma: no cover
                errors.append(f"{backend}: {exc}")

        if not self._iter_backends():
            raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract.")
        if errors:
            raise RuntimeError("OCR falló: " + "; ".join(errors))

        raise RuntimeError("OCR falló: no se obtuvo texto útil.")


def crop_rel(img, l: float, t: float, r: float, b: float):
    w, h = img.size
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def extract_text_from_pdf(path: Path) -> tuple[str, str]:
    # 1) pdfplumber
    if pdfplumber is not None:
        try:
            chunks: list[str] = []
            with pdfplumber.open(str(path)) as pdf:  # type: ignore
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        chunks.append(txt)
            out = normalize_text_block(normalize_ocr_dates("\n".join(chunks).strip()))
            if out:
                return out, "pdf_text"
        except Exception:
            pass

    # 2) PyPDF2
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))  # type: ignore
            chunks = []
            for p in reader.pages:
                txt = (p.extract_text() or "").strip()
                if txt:
                    chunks.append(txt)
            out = normalize_text_block(normalize_ocr_dates("\n".join(chunks).strip()))
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
    # Normalizaciones clave
    text_full = normalize_text_block(normalize_ocr_dates(text_full or ""))
    text_header = normalize_text_block(normalize_ocr_dates(text_header or ""))
    text_totals = normalize_text_block(normalize_ocr_dates(text_totals or ""))

    razon = derive_razon_social_from_filename(path)
    es_nc = is_credit_note(text_full, path)

    # Serie/folio: probá con full primero (evita crops OCR erróneos)
    serie, folio = extract_serie_folio(text_full)
    if not serie or not folio:
        s2, f2 = extract_serie_folio(text_header)
        serie, folio = serie or s2, folio or f2

    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    # RUT emisor: full primero; header puede estar malocrificado
    rut_full = extract_rut_emisor(text_full)
    rut_head = extract_rut_emisor(text_header)
    rut_emisor = rut_full or rut_head

    # Fecha: header suele tener más contexto, pero castigamos venc/CAE
    fecha = extract_fecha_documento(text_header) or extract_fecha_documento(text_full)

    # Total: preferir totals, caer a full
    total_num = extract_total(text_totals) or extract_total(text_full)
    total_str = format_money_uy(total_num)

    # IVA: preferir totals, caer a full
    iva_total = extract_iva_total(text_totals) or extract_iva_total(text_full)

    # Subtotal candidates: tomar el que valide contra total/iva; si no valida, ignorar
    sin_iva_num: Optional[float] = None
    sin_iva_fuente: Optional[str] = None

    if total_num is not None and iva_total is not None:
        # validación con candidates
        subs = extract_subtotal_candidates(text_totals) + extract_subtotal_candidates(text_full)
        for sub in sorted(set(subs), reverse=True):
            if _validate_subtotal(sub, total_num, iva_total):
                sin_iva_num = round(sub, 2)
                sin_iva_fuente = "subtotal_label_validado"
                break

    # si no hay subtotal validado: calcula
    if sin_iva_num is None:
        sin_iva_num, sin_iva_fuente = compute_importe_sin_iva(total_num, iva_total, text_full + "\n" + text_totals)

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

    # Crops razonables (fallan menos que 0.50/0.60 fijo)
    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.42)
    totals_img = crop_rel(img, 0.00, 0.55, 1.00, 1.00)

    text_full, fuente = ocr.ocr_image(img)
    text_header, _ = ocr.ocr_image(header_img)
    text_totals, _ = ocr.ocr_image(totals_img)

    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print(text_full)
        print("\n---[OCR HEADER]---")
        print(text_header)
        print("\n---[OCR TOTALS]---")
        print(text_totals)
        print("=== FIN ===\n")

    return text_full, text_header, text_totals, fuente


def process_pdf(path: Path, debug: bool) -> tuple[str, str, str, str]:
    text_full, fuente = extract_text_from_pdf(path)
    text_header = text_full
    text_totals = text_full

    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print(text_full)
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
