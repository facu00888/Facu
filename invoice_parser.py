# ruff: noqa: E501
r"""
invoice_parser.py

Extractor simple de metadatos de e-Facturas (UY) desde:
- PDFs con texto (pdfplumber/PyPDF2)
- Imágenes (OCR con EasyOCR si está disponible; sino intenta pytesseract si hay tesseract instalado)

Uso (Windows):
  python invoice_parser.py "C:\Proyectos\Facu\Facturas" --json --debug --backend auto
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
# Regex / utilidades
# ---------------------------

DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

# Dinero "normal" UY: 1.234,56 o 123,45
MONEY_RE = re.compile(r"\b(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))\b")

# OCR mugriento: números sin separador (evitar años => mínimo 5 dígitos por defecto)
MONEY_RE_LOOSE_TOTAL = re.compile(r"\b(-?\d{5,})\b")
MONEY_RE_LOOSE_ANY = re.compile(r"\b(-?\d{3,})\b")  # solo para IVA allow_short

RUT_RE = re.compile(r"\b(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})\b")


def _safe_upper(s: str | None) -> str:
    return (s or "").upper()


def _collapse_spaces(s: str | None) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())


def normalize_text_block(text: str) -> str:
    lines = [_collapse_spaces(line) for line in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def parse_money_uy(s: str) -> Optional[float]:
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

    Para evitar capturar años tipo 2027 => por defecto requiere >= 5 dígitos.
    allow_short=True permite 3-4 dígitos (para IVA leído como 224, etc.).
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

    if allow_short:
        if len(digits) < 3:
            return None
    else:
        if len(digits) < 5:
            return None

    whole = digits[:-2] or "0"
    frac = digits[-2:]
    try:
        v = float(f"{int(whole)}.{int(frac):02d}")
        return -v if neg else v
    except Exception:
        return None


def parse_money_with_fallback(s: str, *, allow_short: bool = False) -> Optional[float]:
    return parse_money_uy(s) or parse_money_uy_loose(s, allow_short=allow_short)


def format_money_uy(v: Optional[float]) -> Optional[str]:
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


def extract_fecha_documento(text: str) -> Optional[str]:
    """
    - Prioriza "FECHA DE DOCUMENTO: dd/mm/aaaa"
    - Maneja el caso "FECHA VENCIMIENTO ... dd/mm/aaaa dd/mm/aaaa" -> toma la primera.
    - Fallback: heurística general.
    """
    if not text:
        return None
    up = _safe_upper(text)

    m = re.search(r"\bFECHA\s+DE\s+DOCUMENTO\b[^0-9]{0,30}(\d{1,2}/\d{1,2}/\d{4})", up)
    if m:
        return m.group(1)

    # Caso típico BIMBO: "Fecha Vencimiento Moneda 1/10/2025 12/11/2025"
    m = re.search(
        r"\bFECHA\b.{0,40}\bVENCIMIENTO\b.{0,40}(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})",
        up,
        flags=re.S,
    )
    if m:
        return m.group(1)

    # Heurística general
    best = None
    best_score = -10_000.0
    for dmatch in DATE_RE.finditer(up):
        d = dmatch.group(1)
        i = dmatch.start()
        ctx = up[max(0, i - 50) : i + 50]

        score = 0.0
        if "FECHA DE DOCUMENTO" in ctx:
            score += 12.0
        elif re.search(r"\bFECHA\b", ctx):
            score += 6.0

        # castigo fuerte a vencimientos/CAE
        if "VENC" in ctx or "VTO" in ctx:
            score -= 25.0
        if "CAE" in ctx:
            score -= 8.0

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

    # muchas facturas solo dicen "RUC" al principio
    m = re.search(r"\bRUC\b[^0-9]{0,30}(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3}|\d{11,12})", up)
    if m:
        return _clean_rut(m.group(1))

    m = RUT_RE.search(up)
    if m:
        return _clean_rut(m.group(1))

    return None


def extract_serie_folio_from_filename(path: Path) -> tuple[Optional[str], Optional[str]]:
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
    if not text:
        return None, None

    patterns = [
        r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMERO\b[^0-9]{0,20}(\d{3,})",
        r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\s*([0-9]{3,})",
        r"\bNUMERO\b[^A-Z0-9]{0,20}([A-Z])\s*([0-9]{3,})",
        r"\b([A-Z])\s*-?\s*(\d{3,})\b",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if not m:
            continue
        serie = m.group(1).upper()
        folio_raw = m.group(2)
        digits = re.sub(r"\D", "", folio_raw)
        if not (3 <= len(digits) <= 8):
            continue
        folio = digits.lstrip("0") or "0"
        return serie, folio

    return None, None


def extract_iva_total(text: str) -> Optional[float]:
    if not text:
        return None
    up = _safe_upper(text)

    money_pat = MONEY_RE.pattern + "|" + MONEY_RE_LOOSE_ANY.pattern

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

    money_pat = MONEY_RE.pattern + "|" + MONEY_RE_LOOSE_TOTAL.pattern

    def _collect(pattern: str, score: int) -> None:
        for m in re.finditer(pattern, text, flags=re.I):
            v = parse_money_with_fallback(m.group(1))
            if v is not None:
                candidates.append((v, score))

    _collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,25}((?:" + money_pat + r"))", 100)
    _collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,25}((?:" + money_pat + r"))", 70)
    _collect(r"\bTOTAL\s*:\s*\$?\s*((?:" + money_pat + r"))", 75)

    if candidates:
        candidates.sort(key=lambda t: (t[1], t[0]), reverse=True)
        return candidates[0][0]

    # fallback: mayor valor "normal" que aparezca
    vals = [parse_money_with_fallback(m.group(1)) for m in MONEY_RE.finditer(text)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals)


def extract_rate_amount(text: str, rate: int, kind: str) -> Optional[float]:
    """
    kind:
      - "gravado": busca "Subtotal gravado (22%) X"
      - "iva": busca "Total iva (22%) X"
    """
    if not text:
        return None
    up = _safe_upper(text)
    money_pat = MONEY_RE.pattern + "|" + MONEY_RE_LOOSE_TOTAL.pattern

    if kind == "gravado":
        patterns = [
            rf"\bSUBTOTAL\s+GRAVADO\b.{0,40}\b{rate}\b.{0,10}[%\)]?.{{0,20}}((?:{money_pat}))",
            rf"\bNETO\s+GRAVADO\b.{0,40}\b{rate}\b.{0,10}[%\)]?.{{0,20}}((?:{money_pat}))",
        ]
    else:  # iva
        patterns = [
            rf"\bTOTAL\s+IVA\b.{0,40}\b{rate}\b.{0,10}[%\)]?.{{0,20}}((?:{money_pat}))",
            rf"\bI\.?V\.?A\.?\b.{0,20}\b{rate}\b.{0,10}[%\)]?.{{0,20}}((?:{money_pat}))",
        ]

    for pat in patterns:
        m = re.search(pat, up, flags=re.S)
        if m:
            v = parse_money_with_fallback(m.group(1))
            if v is not None:
                return v
    return None


def detect_single_tax_rate(text: str) -> Optional[float]:
    """
    Devuelve 0.22 o 0.10 si parece que el documento usa solo una tasa (o la otra es 0,00).
    """
    if not text:
        return None
    up = _safe_upper(text)

    g10 = extract_rate_amount(up, 10, "gravado")
    g22 = extract_rate_amount(up, 22, "gravado")
    i10 = extract_rate_amount(up, 10, "iva")
    i22 = extract_rate_amount(up, 22, "iva")

    def _nz(x: Optional[float]) -> bool:
        return x is not None and abs(x) > 0.0001

    # Si hay evidencia de 10 y 22 no-cero -> no podemos asumir tasa única
    if (_nz(g10) or _nz(i10)) and (_nz(g22) or _nz(i22)):
        return None

    # Si 22 aparece y 10 es 0 o no aparece
    if ("22" in up or "224" in up) and not (_nz(g10) or _nz(i10)):
        return 0.22

    # Si 10 aparece y 22 no aparece/no-cero
    if "10" in up and not (_nz(g22) or _nz(i22)):
        return 0.10

    return None


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

        backends = self._iter_backends()
        if not backends:
            raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract.")

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
            except Exception as exc:  # pragma: no cover
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


def parse_invoice_from_text(
    text_full: str,
    text_header: str,
    text_totals: str,
    path: Path,
    fuente: str,
) -> InvoiceResult:
    razon = derive_razon_social_from_filename(path)
    es_nc = is_credit_note(text_full, path)

    # SERIE/FOLIO: intentar header -> full; luego filename.
    serie, folio = extract_serie_folio(text_header) if text_header else (None, None)
    if not serie or not folio:
        serie, folio = extract_serie_folio(text_full)

    if not serie or not folio:
        s2, f2 = extract_serie_folio_from_filename(path)
        if s2 and f2:
            serie, folio = s2, f2

    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    # RUT EMISOR: priorizar texto completo (header es el que suele “alucinar” números)
    rut_emisor = extract_rut_emisor(text_full) or extract_rut_emisor(text_header)

    # FECHA: extractor dedicado
    fecha = extract_fecha_documento(text_full) or extract_fecha_documento(text_header)

    # Totales/IVA
    total_num = extract_total(text_totals) or extract_total(text_full)
    iva_total_raw = extract_iva_total(text_totals) or extract_iva_total(text_full)

    sin_iva_num: Optional[float] = None
    sin_iva_fuente: Optional[str] = None

    # 1) Si IVA parece coherente, usar total - iva
    if total_num is not None and iva_total_raw is not None:
        if 0 < iva_total_raw < total_num:
            base = total_num - iva_total_raw
            if base > 0:
                ratio = iva_total_raw / base
                if 0.03 <= ratio <= 0.35:  # tolerante
                    sin_iva_num = round(base, 2)
                    sin_iva_fuente = "total_menos_iva"

    # 2) Si no, y parece haber tasa única, usar total/(1+tasa)
    if sin_iva_num is None and total_num is not None:
        rate = detect_single_tax_rate("\n".join([text_totals or "", text_full or ""]))
        if rate in (0.10, 0.22):
            sin_iva_num = round(total_num / (1.0 + rate), 2)
            sin_iva_fuente = f"total_div_{int(rate*100)}"

    return InvoiceResult(
        fecha=fecha,
        serie=serie,
        folio=folio,
        serie_y_folio=serie_y_folio,
        razon_social=razon,
        rut_emisor=rut_emisor,
        es_nota_de_credito=es_nc,
        importe_total_con_iva=format_money_uy(total_num),
        importe_total_con_iva_num=total_num,
        importe_sin_iva=format_money_uy(sin_iva_num),
        importe_sin_iva_num=sin_iva_num,
        importe_sin_iva_fuente=sin_iva_fuente,
        _archivo=str(path),
        _fuente=fuente,
    )


def process_image(path: Path, ocr: OCRBackend, debug: bool) -> tuple[str, str, str, str]:
    if Image is None:
        raise RuntimeError("Falta pillow (PIL) para leer imágenes.")
    img = Image.open(path)

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

            # Incluso si falla OCR, intentar sacar serie/folio del nombre para no perder todo
            serie, folio = extract_serie_folio_from_filename(path)
            serie_y_folio = f"{serie}-{folio}" if serie and folio else None

            results.append(
                InvoiceResult(
                    fecha=None,
                    serie=serie,
                    folio=folio,
                    serie_y_folio=serie_y_folio,
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
