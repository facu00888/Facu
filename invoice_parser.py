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
import json
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------
# Silenciar warnings ruidosos (EasyOCR/Torch)
# ---------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*pin_memory.*no accelerator.*",
    category=UserWarning,
)

# ---------------------------
# Dependencias opcionales
# ---------------------------
try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover
    easyocr = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
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
# Regex / patrones
# ---------------------------
DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
# Monto tipo UY: 1.234,56 o 701,00
MONEY_RE = re.compile(r"\b(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))\b")
# OCR sucio: enteros largos (pero OJO con IDs)
MONEY_RE_LOOSE = re.compile(r"\b(-?\d{4,})\b")

RUT_RE = re.compile(r"\b(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})\b")

# Para extraer folios que vienen con espacios: "3 519 972"
DIGITS_WITH_SPACES_RE = re.compile(r"[0-9][0-9\s]{2,}[0-9]")


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


def _only_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _digits_compact(s: str) -> str:
    """Convierte '3 519 972' -> '3519972'."""
    return _only_digits(s)


def _clean_rut(raw: str) -> str:
    return _only_digits(raw)


def _parse_ddmmyyyy(s: str) -> Optional[date]:
    try:
        d, m, y = s.split("/")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def parse_money_uy(s: str) -> Optional[float]:
    """
    Convierte:
      "4.919,61" -> 4919.61
      "701,00"   -> 701.0
      "4 018,01" -> 4018.01  (OCR)
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

    Por defecto:
      - Rechaza <4 dígitos (para no agarrar años/porcentajes)
      - Rechaza demasiado largo (para no agarrar IDs tipo CAE/RUT)
    """
    if not s:
        return None
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    digits = _only_digits(s)
    if not digits:
        return None

    # Evitar IDs (ej: CAE, RUT, etc.)
    if len(digits) > 9:
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


def parse_money_token(s: str, *, allow_loose: bool = True, allow_short_loose: bool = False) -> Optional[float]:
    """Parse robusto: UY primero; luego loose pero con límites para no comer IDs."""
    v = parse_money_uy(s)
    if v is not None:
        return v
    if allow_loose:
        return parse_money_uy_loose(s, allow_short=allow_short_loose)
    return None


def format_money_uy(v: Optional[float]) -> Optional[str]:
    """4919.61 -> "4.919,61"."""
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

    # cortar antes de algo tipo "A3519972" o "A 3519972"
    stem = re.split(r"\s+[A-Z]\s*0*\d{3,}\b", stem)[0]

    stem = stem.replace("NOTA DE CREDITO", "").replace("NOTA DE CRÉDITO", "").strip()
    stem = re.sub(r"\s+CREDITO\b", "", stem).strip()
    stem = _collapse_spaces(stem)
    return stem or None


def derive_serie_folio_from_filename(path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Fallback MUY útil para imágenes con OCR sucio:
      "BIMBO A3519972 CREDITO.jpeg" -> ("A", "3519972")
      "PURA PALTA A49404 CREDITO.jpeg" -> ("A", "49404")
    """
    name = _safe_upper(path.stem)
    # buscar letra + número pegado o con espacios
    m = re.search(r"\b([A-Z])\s*0*([0-9][0-9\s]{2,})\b", name)
    if not m:
        return None, None
    serie = m.group(1).upper()
    folio_digits = _digits_compact(m.group(2))
    folio_digits = folio_digits.lstrip("0") or "0"

    # evitar capturar RUT/RUC por accidente
    if len(folio_digits) > 9:
        return serie, None
    return serie, folio_digits


def is_credit_note(text: str, path: Path) -> bool:
    x = _safe_upper(text) + " " + _safe_upper(path.name)
    return ("NOTA DE CREDITO" in x) or ("NOTA DE CRÉDITO" in x)


def _near(text: str, idx: int, pattern: str, *, left: int = 0, right: int = 0) -> bool:
    a = max(0, idx - left)
    b = min(len(text), idx + right)
    return re.search(pattern, text[a:b], flags=re.I) is not None


def pick_best_date(text: str) -> Optional[str]:
    """
    Heurística:
    - Prioriza FECHA / FECHA DE DOCUMENTO
    - Penaliza VENC/VTO y CAE
    - Descarta fechas demasiado futuras (típico "vencimiento CAE 2027")
    """
    if not text:
        return None

    today = date.today()
    min_ok = today - timedelta(days=365 * 10)
    max_ok = today + timedelta(days=370)  # tolera vencimientos razonables

    best: Optional[str] = None
    best_score = -10_000.0

    for m in DATE_RE.finditer(text):
        dstr = m.group(1)
        idx = m.start()
        dobj = _parse_ddmmyyyy(dstr)
        if dobj is None:
            continue

        # Ventana temporal: mata CAE en 2027 cuando estás en 2025
        if dobj < min_ok or dobj > max_ok:
            continue

        score = 0.0

        # "FECHA DE DOCUMENTO" (fuerte)
        if _near(text, idx, r"FECHA\s+DE\s+DOCUMENTO", left=30, right=30):
            score += 80
        # "FECHA" (moderado)
        if _near(text, idx, r"\bFECHA\b", left=18, right=18):
            score += 25

        # penalizaciones cerca del match (no en todo el párrafo)
        if _near(text, idx, r"\bVENC(?:IMIENTO)?\b|\bVTO\b", left=25, right=10):
            score -= 70
        if _near(text, idx, r"\bCAE\b|\bRANGO\s+DE\s+CAE\b|\bFECHA\s+EMISOR\b", left=35, right=15):
            score -= 80

        # Bonus si aparece justo tras la moneda (caso OCR: "UYU 13/10/2025")
        if _near(text, idx, r"\bUYU\b|\bPESO\b", left=12, right=0):
            score += 10

        # Bonus por “cercanía” al inicio (suele estar arriba)
        score += max(0, 2500 - idx) / 2500.0

        # Preferir fechas más cercanas al presente (pero leve)
        days_from_today = abs((today - dobj).days)
        score += max(0.0, 10.0 - (days_from_today / 30.0))

        if score > best_score:
            best_score = score
            best = dstr

    return best


def extract_rut_emisor(text: str) -> Optional[str]:
    if not text:
        return None

    up = _safe_upper(text)

    m = re.search(r"\bRU[TC]\s*EMISOR\b[^0-9]{0,40}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = re.search(r"\bRUC\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = re.search(r"\bRUT\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = RUT_RE.search(up)
    if m:
        return _clean_rut(m.group(1))

    return None


def _valid_folio_digits(digits: str) -> bool:
    # Folios típicos: 3 a 8 dígitos. Si es 11/12 ya es RUT/RUC casi seguro.
    return 3 <= len(digits) <= 9


def extract_serie_folio(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    # 1) SERIE X ... NUMERO 3 519 972
    m = re.search(
        r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMER[O0]\b[^0-9]{0,30}([0-9][0-9\s]{2,})",
        text,
        flags=re.I | re.S,
    )
    if m:
        serie = m.group(1).upper()
        folio = _digits_compact(m.group(2)).lstrip("0") or "0"
        if _valid_folio_digits(folio):
            return serie, folio

    # 2) SERIE X 3519972
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\s*0*([0-9][0-9\s]{2,})\b", text, flags=re.I)
    if m:
        serie = m.group(1).upper()
        folio = _digits_compact(m.group(2)).lstrip("0") or "0"
        if _valid_folio_digits(folio):
            return serie, folio

    # 3) "DOCUMENTO serie NUMERO ... 3 519 972" (OCR típico)
    m = re.search(
        r"\bDOCUMENTO\b.*?\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMER[O0]\b[^0-9]{0,30}([0-9][0-9\s]{2,})",
        text,
        flags=re.I | re.S,
    )
    if m:
        serie = m.group(1).upper()
        folio = _digits_compact(m.group(2)).lstrip("0") or "0"
        if _valid_folio_digits(folio):
            return serie, folio

    # 4) patrón rápido: "A 3519972" (pero filtrar RUT/RUC)
    m = re.search(r"\b([A-Z])\s*-?\s*0*([0-9][0-9\s]{2,})\b", text, flags=re.I)
    if m:
        serie = m.group(1).upper()
        folio = _digits_compact(m.group(2)).lstrip("0") or "0"
        if _valid_folio_digits(folio):
            return serie, folio

    return None, None


def detect_vat_rates(text: str) -> set[float]:
    """Detecta si aparecen referencias a IVA 22% y/o 10%."""
    up = _safe_upper(text or "")
    rates: set[float] = set()
    if re.search(r"\b22\s*%|\bIVA\s*22|\b22\)", up):
        rates.add(0.22)
    if re.search(r"\b10\s*%|\bIVA\s*10|\b10\)", up):
        rates.add(0.10)
    return rates


def extract_iva_total(text: str) -> Optional[float]:
    """
    Busca IVA total o suma IVA 10% + IVA 22%.
    Tolera OCR sucio.
    """
    if not text:
        return None
    up = _safe_upper(text)

    # "TOTAL IVA: 1.234,56"
    m = re.search(r"\bTOTAL\s*IVA\b[^0-9\-]{0,25}(" + MONEY_RE.pattern + r")", up)
    if m:
        return parse_money_token(m.group(1), allow_loose=True, allow_short_loose=True)

    # "Total iva (22%) 231,18" etc
    iva10 = None
    iva22 = None

    def _money_after(pattern: str) -> Optional[float]:
        mm = re.search(pattern, up)
        if mm:
            return parse_money_token(mm.group(1), allow_loose=True, allow_short_loose=True)
        return None

    iva10 = _money_after(r"\bTOTAL\s*IVA\b[^0-9]{0,60}10\D{0,10}(" + MONEY_RE.pattern + r")")
    iva22 = _money_after(r"\bTOTAL\s*IVA\b[^0-9]{0,60}22\D{0,10}(" + MONEY_RE.pattern + r")")

    if iva10 is None:
        iva10 = _money_after(r"\bI\.?V\.?A\.?\s*10\D{0,20}(" + MONEY_RE.pattern + r")")
    if iva22 is None:
        iva22 = _money_after(r"\bI\.?V\.?A\.?\s*22\D{0,20}(" + MONEY_RE.pattern + r")")

    if iva10 is None and iva22 is None:
        return None
    return round((iva10 or 0.0) + (iva22 or 0.0), 2)


def extract_total(text: str) -> Optional[float]:
    if not text:
        return None

    candidates: list[tuple[float, int]] = []

    def _collect(pattern: str, score: int) -> None:
        for m in re.finditer(pattern, text, flags=re.I | re.S):
            tok = m.group(1)
            v = parse_money_token(tok, allow_loose=True)
            if v is not None:
                candidates.append((v, score))

    # Prioritarios
    _collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,40}(" + MONEY_RE.pattern + r")", 120)
    _collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,40}(" + MONEY_RE_LOOSE.pattern + r")", 110)

    # "TOTAL: 4.018,01" (pero no "TOTAL IVA")
    _collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,40}(" + MONEY_RE.pattern + r")", 90)
    _collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,40}(" + MONEY_RE_LOOSE.pattern + r")", 70)

    if candidates:
        # Preferir los más “etiquetados” y, a igualdad, el mayor
        candidates.sort(key=lambda t: (t[1], t[0]), reverse=True)
        # Filtro anti-locura: si el mejor es descomunal y el segundo es razonable, agarrar el segundo.
        best_v, best_s = candidates[0]
        if best_v > 5_000_000 and len(candidates) > 1 and candidates[1][0] < 500_000:
            return candidates[1][0]
        return best_v

    # fallback: mayor monto con formato UY
    vals = [parse_money_token(m.group(1), allow_loose=False) for m in MONEY_RE.finditer(text)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals)


def extract_subtotal(text: str) -> Optional[float]:
    """Detecta importe sin IVA a partir de etiquetas comunes."""
    if not text:
        return None

    labels = [
        r"SUBTOTAL",
        r"TOTAL\s+SIN\s+IVA",
        r"IMPORTE\s+NETO",
        r"NETO\s+GRAVADO",
        r"SUBTOTAL\s+GRAVADO",
    ]
    for label in labels:
        pattern = rf"\b{label}\b[^0-9\-]{{0,40}}({MONEY_RE.pattern})"
        m = re.search(pattern, text, flags=re.I)
        if m:
            v = parse_money_token(m.group(1), allow_loose=False)
            if v is not None:
                return v
    return None


def compute_importe_sin_iva(
    total: Optional[float],
    iva_total: Optional[float],
    vat_rates: set[float],
    text_full: str,
) -> tuple[Optional[float], Optional[str]]:
    """
    Orden:
    1) total - iva_total (solo si la tasa resultante es razonable)
    2) si parece SOLO IVA 22 -> total / 1.22
       si parece SOLO IVA 10 -> total / 1.10
    """
    if total is None:
        return None, None

    # 1) total - iva_total (validando “tasa razonable”)
    if iva_total is not None and iva_total >= 0 and iva_total < total:
        net = round(total - iva_total, 2)
        # Validación: iva/net cerca de 0.22 o 0.10 (si se detectaron) o al menos “posible”.
        rate = iva_total / max(net, 0.01)
        if vat_rates:
            if any(abs(rate - r) <= 0.03 for r in vat_rates):
                return net, "total_menos_iva"
        else:
            if 0.07 <= rate <= 0.25:
                return net, "total_menos_iva"

    # 2) dividir por tasa (solo si NO parece mixto)
    up = _safe_upper(text_full)
    has_22 = (0.22 in vat_rates) or (re.search(r"\bIVA\s*22|\b22\s*%", up) is not None)
    has_10 = (0.10 in vat_rates) or (re.search(r"\bIVA\s*10|\b10\s*%", up) is not None)

    # Si detecta ambos, evitar inventar
    if has_22 and has_10:
        return None, None

    if has_22:
        return round(total / 1.22, 2), "total_div_22"
    if has_10:
        return round(total / 1.10, 2), "total_div_10"

    return None, None


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
            # idiomas: español+inglés, GPU off (Windows común)
            self._easy_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)  # type: ignore
        return self._easy_reader

    def _preprocess_image(self, img):
        img = ImageOps.exif_transpose(img)  # type: ignore
        img = img.convert("L")
        img = ImageOps.autocontrast(img)  # type: ignore
        img = ImageEnhance.Contrast(img).enhance(2.0)  # type: ignore
        img = ImageEnhance.Sharpness(img).enhance(1.6)  # type: ignore
        # Upscale suave (ayuda a OCR en facturas chicas)
        w, h = img.size
        if max(w, h) < 2000:
            img = img.resize((int(w * 1.5), int(h * 1.5)))
        return img

    def ocr_image_easy(self, img) -> str:
        reader = self._get_easy_reader()
        arr = np.array(img.convert("RGB"))  # type: ignore

        # Pass 1: paragraph=True
        lines = reader.readtext(arr, detail=0, paragraph=True)
        if lines:
            return "\n".join(lines)

        # Pass 2: paragraph=False (a veces paragraph colapsa todo a vacío)
        lines = reader.readtext(arr, detail=0, paragraph=False)
        return "\n".join(lines) if lines else ""

    def ocr_image_tess(self, img) -> str:
        cfg = "--oem 3 --psm 6"
        try:
            return pytesseract.image_to_string(img, lang="spa", config=cfg)  # type: ignore
        except Exception:
            return pytesseract.image_to_string(img, lang="eng", config=cfg)  # type: ignore

    def _iter_backends(self) -> list[str]:
        available: list[str] = []
        if self.has_easyocr():
            available.append("easyocr")
        if self.has_tesseract():
            available.append("tesseract")

        if self.preferred_backend != "auto" and self.preferred_backend in available:
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
            raise RuntimeError("OCR falló; " + "; ".join(errors))
        raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract.")


def crop_rel(img, l: float, t: float, r: float, b: float):
    w, h = img.size
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def extract_text_from_pdf(path: Path) -> tuple[str, str]:
    # pdfplumber primero
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

    # fallback PyPDF2
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
    text_full = text_full or ""
    text_header = text_header or text_full
    text_totals = text_totals or text_full

    razon = derive_razon_social_from_filename(path)
    es_nc = is_credit_note(text_full, path)

    # serie/folio (primero OCR; si falla, filename)
    serie, folio = extract_serie_folio(text_header)
    if not serie or not folio:
        s2, f2 = derive_serie_folio_from_filename(path)
        serie = serie or s2
        folio = folio or f2

    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_emisor = extract_rut_emisor(text_header) or extract_rut_emisor(text_full)
    fecha = pick_best_date(text_header) or pick_best_date(text_full)

    total_num = extract_total(text_totals) or extract_total(text_full)
    total_str = format_money_uy(total_num)

    vat_rates = detect_vat_rates(text_full + "\n" + text_totals)
    iva_total = extract_iva_total(text_totals) or extract_iva_total(text_full)

    sin_iva_num, sin_iva_fuente = compute_importe_sin_iva(
        total=total_num,
        iva_total=iva_total,
        vat_rates=vat_rates,
        text_full=text_full + "\n" + text_totals,
    )
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

    try:
        img = Image.open(path)
    except Exception as e:
        raise RuntimeError(f"No pude abrir la imagen: {e}") from e

    # Zonas típicas
    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.45)
    totals_img = crop_rel(img, 0.40, 0.55, 1.00, 0.98)

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

            # Fallback mínimo para no romper el batch
            s, f = derive_serie_folio_from_filename(path)
            results.append(
                InvoiceResult(
                    fecha=None,
                    serie=s,
                    folio=f,
                    serie_y_folio=f"{s}-{f}" if s and f else None,
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
