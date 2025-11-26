# ruff: noqa: E501
r"""
invoice_parser.py

Extractor de metadatos de e-Facturas (UY) desde:
- PDFs con texto (pdfplumber/PyPDF2)
- Imágenes (OCR con EasyOCR si está disponible; si falla, intenta pytesseract si hay tesseract instalado)

Uso (Windows / PowerShell):
  python invoice_parser.py "C:\Proyectos\Facu\Facturas" --json --debug

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
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

# Silencia el warning repetitivo de torch cuando EasyOCR corre en CPU.
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*")


# ---------------------------
# Dependencias opcionales
# ---------------------------

def _try_import(module: str):
    try:
        return __import__(module)
    except Exception:
        return None


pdfplumber = _try_import("pdfplumber")
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore

easyocr = _try_import("easyocr")
pytesseract = _try_import("pytesseract")


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
# Utilidades: texto
# ---------------------------

def _safe_str(s: object) -> str:
    return "" if s is None else str(s)


def _collapse_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", _safe_str(s)).strip()


def normalize_text_block(text: str) -> str:
    """Normaliza saltos de línea y espacios para mejorar las búsquedas."""
    lines = [_collapse_spaces(line) for line in _safe_str(text).splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def norm_key(s: str) -> str:
    """Normalización agresiva para matching: mayúsculas, sin tildes, espacios colapsados."""
    s = _safe_str(s).replace("\u00a0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# ---------------------------
# Utilidades: razón social / tipo doc
# ---------------------------

def derive_razon_social_from_filename(path: Path) -> Optional[str]:
    """
    Usa el nombre del archivo como fuente "fuerte" para razón social:
      "BIMBO A3519972 CREDITO.jpeg" -> "BIMBO"
      "DEL SUR NOTA DE CREDITO A18911 CREDITO.jpeg" -> "DEL SUR"
    """
    stem = norm_key(path.stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = _collapse_spaces(stem)
    stem = re.split(r"\s+A\d+\b", stem)[0]
    stem = stem.replace("NOTA DE CREDITO", "").replace("NOTA DE CRÉDITO", "").strip()
    stem = re.sub(r"\s+CREDITO\b", "", stem).strip()
    stem = _collapse_spaces(stem)
    return stem or None


def is_credit_note(text: str, path: Path) -> bool:
    x = norm_key(text) + " " + norm_key(path.name)
    return ("NOTA DE CREDITO" in x) or ("NOTA DE CRÉDITO" in x)


# ---------------------------
# Fechas (incluye OCR "12/112025")
# ---------------------------

DATE_RE_STRICT = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
# OCR común: 12/112025 (falta la barra entre mes y año)
DATE_RE_MISSING_SLASH = re.compile(r"\b(\d{1,2})/(\d{2})(\d{4})\b")
DATE_RE_DASH = re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b")


def _valid_date(d: str) -> bool:
    try:
        dd, mm, yy = d.split("/")
        date(int(yy), int(mm), int(dd))
        return True
    except Exception:
        return False


def _normalize_date_token(raw: str) -> Optional[str]:
    raw = _safe_str(raw).strip()
    if not raw:
        return None

    m = DATE_RE_STRICT.search(raw)
    if m:
        d = m.group(1)
        if _valid_date(d):
            return d

    m = DATE_RE_MISSING_SLASH.search(raw)
    if m:
        d = f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}"
        if _valid_date(d):
            return d

    m = DATE_RE_DASH.search(raw)
    if m:
        d = f"{int(m.group(1))}/{int(m.group(2))}/{m.group(3)}"
        if _valid_date(d):
            return d

    return None


def iter_dates_with_context(text: str):
    """Yield (date_str, position, ctx_window)."""
    for m in DATE_RE_STRICT.finditer(text):
        d = _normalize_date_token(m.group(1))
        if not d:
            continue
        i = m.start()
        ctx = text[max(0, i - 80) : i + 80]
        yield d, i, ctx

    for m in DATE_RE_MISSING_SLASH.finditer(text):
        d = _normalize_date_token(m.group(0))
        if not d:
            continue
        i = m.start()
        ctx = text[max(0, i - 80) : i + 80]
        yield d, i, ctx

    for m in DATE_RE_DASH.finditer(text):
        d = _normalize_date_token(m.group(0))
        if not d:
            continue
        i = m.start()
        ctx = text[max(0, i - 80) : i + 80]
        yield d, i, ctx


def pick_best_fecha_documento(text: str) -> Optional[str]:
    """
    Intenta elegir "fecha de documento" (no vencimiento).
    Heurísticas:
    - Preferir "FECHA DE DOCUMENTO"
    - Preferir cercanía a "MONEDA" / "UYU" / "USD"
    - Penalizar cercanía a "VENC" / "VTO" y "CAE"
    """
    if not text:
        return None

    best: Optional[str] = None
    best_score = -1_000_000.0

    for d, pos, ctx in iter_dates_with_context(text):
        up = norm_key(ctx)

        score = 0.0

        if "FECHA DE DOCUMENTO" in up:
            score += 50
        elif "FECHA" in up:
            score += 8

        if "MONEDA" in up:
            score += 10
        if any(cc in up for cc in (" UYU", " USD", " US$", " U$S", " UI")):
            score += 8

        if "VENC" in up or "VTO" in up:
            score -= 25
        if "CAE" in up or "RANGO DE CAE" in up:
            score -= 20

        score += max(0.0, 3000 - pos) / 3000.0

        if score > best_score:
            best_score = score
            best = d

    return best


# ---------------------------
# Serie / folio / RUT emisor
# ---------------------------

RUT_RE = re.compile(r"\b(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})\b")


def extract_serie_folio(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMERO\b[^0-9]{0,20}(\d{3,})", text, flags=re.I | re.S)
    if m:
        return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\s*([0-9]{3,})\b", text, flags=re.I)
    if m:
        return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

    m = re.search(r"\bNUMERO\b[^A-Z0-9]{0,20}([A-Z])\s*([0-9]{3,})\b", text, flags=re.I)
    if m:
        return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

    m = re.search(r"\b([A-Z])\s*-?\s*(\d{3,})\b", text, flags=re.I)
    if m:
        return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

    return None, None


def _clean_digits(raw: str) -> str:
    return re.sub(r"[^0-9]", "", raw)


def extract_rut_emisor(text: str) -> Optional[str]:
    if not text:
        return None

    up = norm_key(text)

    m = re.search(r"\bRU[TC]\s*EMISOR\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_digits(m.group(1))

    m = re.search(r"\bRUC\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_digits(m.group(1))

    m = RUT_RE.search(up)
    if m:
        return _clean_digits(m.group(1))

    return None


# ---------------------------
# Dinero: parsing robusto (UY) + OCR sucio
# ---------------------------

AMOUNT_TOKEN_RE = re.compile(r"(?<!\w)-?[0-9OoIlISsBbJjAaGg][0-9OoIlISsBbJjAaGg\.,\s]{1,22}[0-9OoIlISsBbJjAaGg](?!\w)")

BAD_MONEY_CONTEXT = ("CAE", "RANGO", "SEGURIDAD", "CODIGO", "CÓDIGO", "CAI", "RUT ", "RUC ", "DOCUMENTO", "SERIE", "NUMERO")


def _expand_ambiguous(token: str) -> set[str]:
    """
    Genera variantes de token reemplazando letras OCR ambiguas dentro de números.
    """
    token = _safe_str(token)
    variants = {token}

    repl_map = {
        "O": ["0"], "o": ["0"],
        "I": ["1"], "l": ["1"], "|": ["1"],
        "S": ["5"], "s": ["5"],
        "B": ["8"],
        "G": ["6"], "g": ["6"],
        "j": ["3", "5"], "J": ["3", "5"],
        "a": ["3", "8"], "A": ["3", "8"],
    }

    for ch, options in repl_map.items():
        new_vars = set()
        for v in variants:
            if ch in v:
                for opt in options:
                    new_vars.add(v.replace(ch, opt))
            else:
                new_vars.add(v)
        variants = new_vars

    return variants


def _strip_money_junk(s: str) -> str:
    s = _safe_str(s)
    s = s.replace("$", "").replace("UYU", "").replace("U$S", "").replace("US$", "")
    s = re.sub(r"[^\d\.,\s\-]", "", s)
    s = _collapse_spaces(s)
    return s


def _parse_amount_strict(s: str) -> Optional[float]:
    """
    Parseo con miles y decimales:
      "4.018,01" / "4 018,01" -> 4018.01
      "27.752" / "27 752" -> 27752.0
      "2 105" -> 2105.0
    """
    s = _strip_money_junk(s)
    if not re.search(r"\d", s):
        return None

    neg = s.startswith("-")
    if neg:
        s = s[1:].strip()

    s = re.sub(r"\s+", " ", s).strip()

    if "," in s:
        head, tail = s.rsplit(",", 1)
        if re.fullmatch(r"\d{1,2}", tail):
            head = head.replace(".", "").replace(" ", "")
            n = f"{head}.{tail}"
            try:
                v = float(n)
                return -v if neg else v
            except Exception:
                return None

    if "." in s:
        head, tail = s.rsplit(".", 1)
        if re.fullmatch(r"\d{1,2}", tail):
            head = head.replace(" ", "").replace(".", "")
            n = f"{head}.{tail}"
            try:
                v = float(n)
                return -v if neg else v
            except Exception:
                return None

    s2 = s.replace(".", "").replace(" ", "")
    if not s2.isdigit():
        return None
    try:
        v = float(int(s2))
        return -v if neg else v
    except Exception:
        return None


def _parse_amount_digits_as_cents(digits: str) -> Optional[float]:
    digits = digits.lstrip("0") or "0"
    if len(digits) < 3:
        return None
    whole = digits[:-2] or "0"
    frac = digits[-2:]
    try:
        return float(f"{int(whole)}.{int(frac):02d}")
    except Exception:
        return None


def parse_amount_candidates(token: str, *, kind: str, ctx: str = "") -> list[float]:
    """
    Devuelve varias interpretaciones posibles (para OCR sucio).
    kind: "total" | "neto" | "iva" | "generic"
    """
    out: list[float] = []
    ctx_up = norm_key(ctx)

    for v in _expand_ambiguous(token):
        cleaned = _strip_money_junk(v)
        if not cleaned:
            continue

        strict = _parse_amount_strict(cleaned)
        if strict is not None:
            out.append(strict)
            continue

        digits = re.sub(r"\D", "", cleaned)
        if not digits:
            continue

        # Evitar capturar CAE/RUT/etc (suelen ser 10-12 dígitos)
        if len(digits) >= 10:
            continue

        # Para TOTAL/NETO/IVA, 6-9 dígitos suelen ser "...cc" (centésimos) en OCR
        if kind in {"total", "neto", "iva"} and 6 <= len(digits) <= 9:
            as_cents = _parse_amount_digits_as_cents(digits)
            if as_cents is not None:
                out.append(as_cents)

        # Interpretación como entero (ej. "27.752" -> "27752")
        if len(digits) >= 4:
            try:
                out.append(float(int(digits)))
            except Exception:
                pass

        # Evitar confundir tasas (101/224) con dinero
        if kind == "iva" and len(digits) <= 3 and any(w in ctx_up for w in ("TASA", "%", "IVA")):
            pass

    uniq = []
    seen = set()
    for x in out:
        k = round(float(x), 4)
        if k not in seen:
            seen.add(k)
            uniq.append(float(x))
    return uniq


def format_money_uy(v: Optional[float]) -> Optional[str]:
    """4919.61 -> "4.919,61" """
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


# ---------------------------
# Extractores: total / subtotal / IVA
# ---------------------------

TOTAL_MARKERS = ("TOTAL A PAGAR", "TOTAL PAGAR", "TOTAL", "IOIAL", "TOIAL", "TOTAI", "TOIM", "T0TAL")
TOTAL_IVA_MARKERS = ("TOTAL IVA",)


def _line_tokens(line: str) -> list[str]:
    return [m.group(0) for m in AMOUNT_TOKEN_RE.finditer(line)]


def _bad_money_line(line: str) -> bool:
    u = norm_key(line)
    return any(b in u for b in BAD_MONEY_CONTEXT)


def extract_total(text: str, *, max_total: float) -> Optional[float]:
    """
    Extrae total con heurísticas:
    - Busca líneas con TOTAL (o variantes OCR) y toma números de esa línea o hasta 2 líneas siguientes.
    - Filtra basura (CAE/códigos largos) y valores fuera de rango.
    """
    if not text:
        return None

    lines = normalize_text_block(text).splitlines()
    best_val: Optional[float] = None
    best_score = -1_000_000.0

    for i, line in enumerate(lines):
        u = norm_key(line)
        if not any(m in u for m in TOTAL_MARKERS):
            continue
        if any(m in u for m in TOTAL_IVA_MARKERS):
            continue

        window = lines[i : i + 3]
        window_u = norm_key(" ".join(window))
        ctx_penalty = -80.0 if any(b in window_u for b in BAD_MONEY_CONTEXT) else 0.0

        for rel, ln in enumerate(window):
            for tok in _line_tokens(ln):
                for v in parse_amount_candidates(tok, kind="total", ctx=ln):
                    if v is None:
                        continue
                    if not (0 < abs(v) < max_total):
                        continue

                    score = 0.0
                    score += 120.0 if rel == 0 else (90.0 if rel == 1 else 70.0)
                    score += 8.0 if ("," in tok or re.search(r"[.,]\d{2}\b", tok)) else 0.0
                    score += min(v / max_total, 1.0) * 5.0
                    score += ctx_penalty
                    if _bad_money_line(ln):
                        score -= 40.0

                    if score > best_score:
                        best_score = score
                        best_val = v

    if best_val is not None:
        return best_val

    # Fallback: elegir el mayor "monto razonable" del texto, ignorando líneas sospechosas
    candidates: list[float] = []
    for line in lines:
        if _bad_money_line(line):
            continue
        for tok in _line_tokens(line):
            for v in parse_amount_candidates(tok, kind="generic", ctx=line):
                if v is not None and 0 < abs(v) < max_total:
                    candidates.append(v)

    return max(candidates) if candidates else None


def extract_subtotal_sin_iva(text: str, *, max_total: float) -> Optional[float]:
    """Detecta importe sin IVA a partir de etiquetas comunes."""
    if not text:
        return None

    labels = (
        "SUBTOTAL",
        "TOTAL SIN IVA",
        "IMPORTE NETO",
        "NETO GRAVADO",
        "SUBTOTAL GRAVADO",
    )

    lines = normalize_text_block(text).splitlines()
    for i, line in enumerate(lines):
        u = norm_key(line)
        if not any(lbl in u for lbl in labels):
            continue
        if "IVA" in u and "TOTAL" in u:
            continue

        window = [line] + lines[i + 1 : i + 3]
        for ln in window:
            for tok in _line_tokens(ln):
                for v in parse_amount_candidates(tok, kind="neto", ctx=ln):
                    if v is not None and 0 <= v < max_total:
                        return v
    return None


def extract_iva_total(text: str, *, max_total: float) -> Optional[float]:
    """
    Busca IVA total o suma IVA 10% + IVA 22%.
    En PDFs uruguayos suele estar claro ("Total iva (22%) 231,18").
    """
    if not text:
        return None

    txt = normalize_text_block(text)
    lines = txt.splitlines()

    iva10 = None
    iva22 = None
    iva_total = None

    def pick_money_in_line(line: str) -> Optional[float]:
        for tok in _line_tokens(line):
            for v in parse_amount_candidates(tok, kind="iva", ctx=line):
                if v is not None and 0 <= v < max_total:
                    return v
        return None

    for line in lines:
        u = norm_key(line)
        if "TOTAL IVA" in u:
            v = pick_money_in_line(line)
            if v is not None:
                iva_total = v
                break

        if "IVA" in u and "10" in u:
            v = pick_money_in_line(line)
            if v is not None:
                iva10 = v

        if "IVA" in u and "22" in u:
            v = pick_money_in_line(line)
            if v is not None:
                iva22 = v

    if iva_total is not None:
        return iva_total

    if iva10 is None and iva22 is None:
        return None

    return (iva10 or 0.0) + (iva22 or 0.0)


def looks_like_all_22(text: str) -> bool:
    """
    Heurística simple: si el documento menciona IVA 22% / tasa básica varias veces,
    probablemente el total = neto * 1.22.
    """
    u = norm_key(text)
    hits = 0
    hits += 2 if "TASA BASICA" in u or "TASE BASICE" in u or "TASE BABICE" in u else 0
    hits += u.count(" 22")
    hits += 2 if "IVA 22" in u or "IVA 224" in u else 0
    return hits >= 3


def estimate_total_from_neto(text_totals: str, *, max_total: float) -> Optional[float]:
    """
    Busca un NETO en el bloque de totales y estima TOTAL = NETO * 1.22 si parece toda tasa básica.
    """
    if not text_totals:
        return None
    if not looks_like_all_22(text_totals):
        return None

    lines = normalize_text_block(text_totals).splitlines()

    for i, line in enumerate(lines):
        u = norm_key(line)
        if ("MON" in u and "IVA" in u and ("TAS" in u or "TUS" in u or "TASA" in u)):
            window = " ".join(lines[i + 1 : i + 3])
            vals: list[float] = []
            for tok in AMOUNT_TOKEN_RE.findall(window):
                for v in parse_amount_candidates(tok, kind="neto", ctx=window):
                    if v is not None and 0 < v < max_total:
                        vals.append(v)
            if vals:
                neto = max(vals)
                est = round(neto * 1.22, 2)
                if 0 < est < max_total:
                    return est
    return None


# ---------------------------
# OCR / Extractores de texto
# ---------------------------

class OCRBackend:
    def __init__(self, preferred_backend: str = "auto") -> None:
        self.preferred_backend = preferred_backend
        self._easy_reader = None

    def has_easyocr(self) -> bool:
        return easyocr is not None

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
        import numpy as np  # type: ignore
        reader = self._get_easy_reader()
        arr = np.array(img.convert("RGB"))  # type: ignore
        lines = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)

    def ocr_image_tess(self, img) -> str:
        cfg = "--oem 3 --psm 6"
        for lang in ("spa", "eng"):
            try:
                return pytesseract.image_to_string(img, lang=lang, config=cfg)  # type: ignore
            except Exception:
                continue
        return ""

    def _iter_backends(self) -> list[str]:
        available: list[str] = []
        if self.has_easyocr():
            available.append("easyocr")
        if self.has_tesseract():
            available.append("tesseract")

        if self.preferred_backend in available and self.preferred_backend != "auto":
            return [self.preferred_backend] + [b for b in available if b != self.preferred_backend]
        return available

    def ocr_image(self, img) -> tuple[str, str]:
        if Image is None:
            raise RuntimeError("Falta pillow (PIL) para OCR de imágenes.")

        img = self._preprocess_image(img)
        backends = self._iter_backends()
        if not backends:
            raise RuntimeError("No hay backend OCR disponible (easyocr o pytesseract+tesseract).")

        errors: list[str] = []
        for backend in backends:
            try:
                if backend == "easyocr":
                    txt = self.ocr_image_easy(img)
                else:
                    txt = self.ocr_image_tess(img)
                normalized = normalize_text_block(txt)
                if normalized:
                    return normalized, f"image_ocr_{backend}"
            except Exception as exc:
                errors.append(f"{backend}: {exc}")

        raise RuntimeError("; ".join(errors) if errors else "OCR falló por razones misteriosas (clásico).")


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

def compute_importe_sin_iva(
    *,
    subtotal_label: Optional[float],
    total: Optional[float],
    iva_total: Optional[float],
) -> tuple[Optional[float], Optional[str]]:
    if subtotal_label is not None:
        return round(subtotal_label, 2), "subtotal_label"
    if total is None or iva_total is None:
        return None, None
    return round(total - iva_total, 2), "total_menos_iva"


def parse_invoice_from_text(
    *,
    text_full: str,
    text_header: str,
    text_totals: str,
    path: Path,
    fuente: str,
    max_total: float,
) -> InvoiceResult:
    razon = derive_razon_social_from_filename(path)
    es_nc = is_credit_note(text_full, path)

    serie, folio = extract_serie_folio(text_header or text_full)
    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_emisor = extract_rut_emisor(text_header or text_full)
    fecha = pick_best_fecha_documento(text_header or text_full)

    subtotal_label = extract_subtotal_sin_iva(text_totals or text_full, max_total=max_total)

    total_num = extract_total(text_totals, max_total=max_total) or extract_total(text_full, max_total=max_total)

    iva_total = extract_iva_total(text_totals, max_total=max_total) or extract_iva_total(text_full, max_total=max_total)

    # Si el total salió vacío, intentamos estimar (tasa básica 22%) a partir del neto
    if total_num is None:
        est = estimate_total_from_neto(text_totals, max_total=max_total) or estimate_total_from_neto(text_full, max_total=max_total)
        if est is not None:
            total_num = est

    # Si total existe pero parece basura vs estimación, preferimos estimación
    est2 = estimate_total_from_neto(text_totals, max_total=max_total) or estimate_total_from_neto(text_full, max_total=max_total)
    if total_num is not None and est2 is not None:
        if abs(total_num - est2) / max(est2, 1.0) > 0.25:
            total_num = est2

    total_str = format_money_uy(total_num)

    sin_iva_num, sin_iva_fuente = compute_importe_sin_iva(subtotal_label=subtotal_label, total=total_num, iva_total=iva_total)
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

    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.40)
    totals_img = crop_rel(img, 0.45, 0.55, 1.00, 0.98)

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


def parse_path(root: Path, *, as_json: bool, debug: bool, preferred_backend: str, max_total: float) -> list[dict[str, Any]]:
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
                max_total=max_total,
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
    ap.add_argument(
        "--max-total",
        type=float,
        default=1_000_000.0,
        help="Filtro anti-basura OCR: total máximo permitido (UYU). Ajustá si manejás facturas enormes.",
    )
    args = ap.parse_args(argv)

    parse_path(Path(args.path), as_json=args.json, debug=args.debug, preferred_backend=args.backend, max_total=args.max_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
