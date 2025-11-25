# invoice_parser.py
# -*- coding: utf-8 -*-
"""
Parser de e-Facturas (UY) desde carpeta con PDFs e imágenes (OCR).
- Extrae: fecha del documento (NO vencimiento CAE), serie/folio, RUT emisor, razón social, total con IVA, neto sin IVA (cuando se puede).
- Soporta: montos con separadores UY (4.919,61) y también con espacio (1 071,00).
- CLI:
    python invoice_parser.py "C:\Ruta\Facturas" --json --debug
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ---------- Dependencias opcionales ----------
EASYOCR_AVAILABLE = True
try:
    import easyocr  # type: ignore
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import numpy as np  # type: ignore
except Exception as e:
    np = None  # type: ignore

try:
    from PIL import Image, ImageOps, ImageEnhance  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageOps = None  # type: ignore
    ImageEnhance = None  # type: ignore

PDFPLUMBER_AVAILABLE = True
try:
    import pdfplumber  # type: ignore
except Exception:
    PDFPLUMBER_AVAILABLE = False

PYMUPDF_AVAILABLE = True
try:
    import fitz  # PyMuPDF  # type: ignore
except Exception:
    PYMUPDF_AVAILABLE = False

PYPDF_AVAILABLE = True
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PYPDF_AVAILABLE = False


# ---------- Regex base ----------
# Fecha "normal"
DATE_RE = re.compile(r"\b([0-3]?\d/[01]?\d/\d{4})\b")

# Fecha tolerante a OCR (0a/10/2025, O8/1O/2O25, etc.)
DATE_FUZZY_RE = re.compile(
    r"\b([0-3]?[0-9A-Za-z]{1,2})[\/\-\.\']([01]?[0-9A-Za-z]{1,2})[\/\-\.\'](\d{4})\b"
)

# Montos: 4.919,61 | 1 071,00 | 8516,99 | 8516.99
AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[.\s]\d{3})*|\d+)[,\.]\d{2}(?!\d)")

# RUT UY (12 dígitos) a veces se imprime con espacios/puntos
RUT_CANDIDATE_RE = re.compile(r"\b(\d[\d\.\s]{10,}\d)\b")

SERIE_RE = re.compile(r"\bserie\b[^A-Z0-9]{0,10}([A-Z])\b", re.IGNORECASE)
NUMERO_RE = re.compile(r"\b(n[uú]mero|numero|folio)\b[^0-9]{0,12}(\d{3,})\b", re.IGNORECASE)


# ---------- Utilidades de normalización ----------
OCR_DIGIT_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    # Esta es la que te pegó en PURA PALTA: "0a/10/2025" -> "08/10/2025"
    "a": "8", "A": "8",
})

def _clean_spaces(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.replace("\u00a0", " ").split())

def _strip_parens(s: str) -> str:
    # Saca "(22%)" y similares para que no contaminen matches
    return re.sub(r"\([^)]*\)", " ", s or "")

def parse_uy_amount(raw: str) -> Optional[float]:
    """
    Acepta: '4.919,61', '1 071,00', '$ 8.516,99', '4919.61' (OCR a veces inventa)
    """
    if not raw:
        return None
    s = raw.strip()
    s = s.replace("\u00a0", " ")
    s = s.replace("$", "").replace("UYU", "").strip()
    s = re.sub(r"[^0-9,.\s]", "", s)
    s = s.replace(" ", "")

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # Solo puntos
        if s.count(".") >= 2:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(s)
    except ValueError:
        return None

def format_uy_amount(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    # Formato UY: miles con punto, decimales con coma
    s = f"{value:,.2f}"  # ej 4,919.61 en locale US
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # -> 4.919,61
    return s

def _valid_date(d: int, m: int) -> bool:
    return 1 <= d <= 31 and 1 <= m <= 12

def _parse_fuzzy_date(day_s: str, mon_s: str, year_s: str) -> Optional[str]:
    day_s2 = day_s.translate(OCR_DIGIT_MAP)
    mon_s2 = mon_s.translate(OCR_DIGIT_MAP)
    year_s2 = year_s.translate(OCR_DIGIT_MAP)

    if not (day_s2.isdigit() and mon_s2.isdigit() and year_s2.isdigit()):
        return None

    d = int(day_s2)
    m = int(mon_s2)
    y = int(year_s2)

    if y < 1900 or y > 2100:
        return None
    if not _valid_date(d, m):
        return None

    return f"{d:02d}/{m:02d}/{y:04d}"  # normaliza

def iter_dates_with_context(text: str):
    lo = (text or "").lower()

    # 1) Fechas “limpias”
    for m in DATE_RE.finditer(text or ""):
        d = m.group(1)
        a = max(0, m.start() - 80)
        b = min(len(text), m.end() + 80)
        ctx = lo[a:b]
        yield d, m.start(), ctx

    # 2) Fechas “sucias” OCR
    for m in DATE_FUZZY_RE.finditer(text or ""):
        d = _parse_fuzzy_date(m.group(1), m.group(2), m.group(3))
        if not d:
            continue
        a = max(0, m.start() - 80)
        b = min(len(text), m.end() + 80)
        ctx = lo[a:b]
        yield d, m.start(), ctx


# ---------- Extracción de campos ----------
def extract_invoice_date(text: str) -> Optional[str]:
    """
    Intenta agarrar la FECHA DEL DOCUMENTO / FECHA, evitando CAE/fecha emisor/vencimiento CAE.
    """
    best = None
    best_score = -10**9

    for d, pos, ctx in iter_dates_with_context(text):
        score = 0

        if "fecha de documento" in ctx or "fecha del documento" in ctx:
            score += 300
        if re.search(r"\bfecha\b", ctx):
            score += 80

        # Penalizaciones: ojos bien abiertos con CAE
        if "cae" in ctx:
            score -= 300
        if "fecha emisor" in ctx:
            score -= 250
        if "vencim" in ctx:
            score -= 40  # vencimiento no es la fecha de emisión

        # Preferí cosas más arriba (pero no tanto)
        score -= (pos / 8000)

        if score > best_score:
            best_score = score
            best = d

    return best

def extract_due_date(text: str) -> Optional[str]:
    """
    Vencimiento de pago (no vencimiento CAE).
    """
    best = None
    best_score = -10**9

    for d, pos, ctx in iter_dates_with_context(text):
        score = 0
        if "vencim" in ctx:
            score += 200
        if "cae" in ctx:
            score -= 400  # NO queremos vencimiento CAE
        if "fecha de documento" in ctx:
            score -= 60

        score -= (pos / 8000)

        if score > best_score:
            best_score = score
            best = d

    return best

def extract_total_amount(text: str) -> Optional[float]:
    """
    Busca TOTAL A PAGAR / TOTAL. Evita Total IVA/Subtotal.
    """
    if not text:
        return None

    t = _clean_spaces(text)
    t = _strip_parens(t)
    lo = t.lower()

    # 1) Prioridad: 'total a pagar' (UY)
    for key in ["total a pagar", "total a pagar:", "total a pagar."]:
        idx = lo.find(key)
        if idx != -1:
            snippet = t[idx: idx + 160]
            snippet = _strip_parens(snippet)
            m = AMOUNT_RE.search(snippet)
            if m:
                val = parse_uy_amount(m.group(0))
                if val is not None:
                    return val

    # 2) Luego 'total' pero filtrando
    # Tomamos la última ocurrencia de "total" que NO sea "total iva"
    for m_total in re.finditer(r"\btotal\b", lo):
        a = max(0, m_total.start() - 20)
        b = min(len(lo), m_total.start() + 20)
        around = lo[a:b]
        if "total iva" in around:
            continue
        snippet = t[m_total.start(): m_total.start() + 160]
        snippet = _strip_parens(snippet)
        m = AMOUNT_RE.search(snippet)
        if m:
            val = parse_uy_amount(m.group(0))
            if val is not None:
                return val

    # 3) Fallback: mayor monto “probable”, evitando IVA/subtotal
    candidates: List[float] = []
    for m in AMOUNT_RE.finditer(t):
        raw = m.group(0)
        val = parse_uy_amount(raw)
        if val is None:
            continue
        a = max(0, m.start() - 60)
        ctx = lo[a:m.start()]
        bad = ["subtotal", "total iva", "monto neto", "iva ", "iva:", "tasa", "gravado"]
        if any(x in ctx for x in bad):
            continue
        candidates.append(val)

    return max(candidates) if candidates else None

def extract_iva_total(text: str) -> Optional[float]:
    """
    Intenta extraer el total IVA (sumado) si aparece como:
    - "Total IVA (10%): X" y "Total IVA (22%): Y"
    - o "Monto IVA 22%: X" etc.
    """
    if not text:
        return None
    t = _clean_spaces(text)
    lo = t.lower()

    iva_vals: List[float] = []

    patterns = [
        r"total\s*iva\s*\(10%\)\s*[:\-]?\s*(" + AMOUNT_RE.pattern + r")",
        r"total\s*iva\s*\(22%\)\s*[:\-]?\s*(" + AMOUNT_RE.pattern + r")",
        r"monto\s*iva\s*22%\s*[:\-]?\s*(" + AMOUNT_RE.pattern + r")",
        r"monto\s*iva\s*10%\s*[:\-]?\s*(" + AMOUNT_RE.pattern + r")",
        r"\biva\s*22%\b[^0-9]{0,20}(" + AMOUNT_RE.pattern + r")",
        r"\biva\s*10%\b[^0-9]{0,20}(" + AMOUNT_RE.pattern + r")",
    ]

    for pat in patterns:
        for m in re.finditer(pat, lo, flags=re.IGNORECASE):
            raw = m.group(1)
            val = parse_uy_amount(raw)
            if val is not None:
                iva_vals.append(val)

    if iva_vals:
        # Si aparecen varias, sumamos (típico 10% + 22%)
        return float(sum(iva_vals))

    return None

def extract_net_amount(text: str, total: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    """
    Intenta extraer neto sin IVA. Devuelve (valor, fuente).
    Fuentes:
      - 'subtotal_gravado' (sumando subtotales gravados)
      - 'total_menos_iva' (si se pudo extraer IVA)
    """
    if not text:
        return None, None

    t = _clean_spaces(text)
    lo = t.lower()

    # Caso UY clásico: "Subtotal gravado (10%)" y "(22%)"
    gravados: List[float] = []

    for pat in [
        r"subtotal\s*gravado\s*\(10%\)\s*[:\-]?\s*(" + AMOUNT_RE.pattern + r")",
        r"subtotal\s*gravado\s*\(22%\)\s*[:\-]?\s*(" + AMOUNT_RE.pattern + r")",
        r"monto\s*neto.*tasa\s*minima[^0-9]{0,20}(" + AMOUNT_RE.pattern + r")",
        r"monto\s*neto.*tasa\s*basica[^0-9]{0,20}(" + AMOUNT_RE.pattern + r")",
    ]:
        for m in re.finditer(pat, lo, flags=re.IGNORECASE):
            raw = m.group(1)
            val = parse_uy_amount(raw)
            if val is not None:
                gravados.append(val)

    if gravados:
        return float(sum(gravados)), "subtotal_gravado"

    # Fallback: neto = total - IVA (si encontramos IVA)
    iva_total = extract_iva_total(t)
    if total is not None and iva_total is not None:
        net = total - iva_total
        # Evitar negativos con OCR loco
        if net > 0:
            return net, "total_menos_iva"

    return None, None

def extract_rut_emisor(text: str) -> Optional[str]:
    """
    Busca el RUT/RUC EMISOR (12 dígitos) evitando comprador/receptor.
    """
    if not text:
        return None

    lo = text.lower()
    best = None
    best_score = -10**9

    for m in RUT_CANDIDATE_RE.finditer(text):
        raw = m.group(1)
        digits = re.sub(r"\D", "", raw)
        if len(digits) != 12:
            continue

        a = max(0, m.start() - 80)
        b = min(len(text), m.end() + 80)
        ctx = lo[a:b]

        score = 0
        if "rut emisor" in ctx or "ruc emisor" in ctx:
            score += 300
        if "rut" in ctx or "ruc" in ctx:
            score += 50
        if "comprador" in ctx or "receptor" in ctx:
            score -= 250
        if "rut comprador" in ctx or "ruc comprador" in ctx:
            score -= 400

        score -= (m.start() / 8000)

        if score > best_score:
            best_score = score
            best = digits

    # Fallback: si no hay contexto, agarrar el primer 12 dígitos y rezar (humano style)
    if not best:
        all12 = re.findall(r"\b\d{12}\b", re.sub(r"\D", " ", text))
        if all12:
            best = all12[0]

    return best

def extract_serie_folio(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    serie = None
    folio = None

    ms = SERIE_RE.search(text)
    if ms:
        serie = ms.group(1).strip().upper()

    # Buscar "Número" junto a "Serie" en una ventana corta
    for mn in NUMERO_RE.finditer(text):
        folio_cand = mn.group(2)
        if folio_cand:
            folio = folio_cand
            break

    return serie, folio

def fallback_serie_folio_from_filename(filepath: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(filepath).stem
    # "A3519972" o "A049009" o "A49404"
    m = re.search(r"\bA(\d{4,})\b", stem, flags=re.IGNORECASE)
    if m:
        return "A", m.group(1)
    return None, None

def extract_es_nota_credito(text: str, filename: str) -> bool:
    lo = (text or "").lower()
    fn = (filename or "").lower()
    return ("nota de credito" in lo) or ("nota de crédito" in lo) or ("nota de credito" in fn) or ("nota de crédito" in fn)

def extract_vendor_name(text: str, filename: str) -> Optional[str]:
    """
    Intenta sacar razón social del emisor.
    - PDF: suele venir en primeras líneas.
    - OCR: heurística simple + fallback por nombre de archivo.
    """
    if text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # 1) si aparece "RAZON SOCIAL" o "NOMBRE O DENOMINACION" intentamos capturar algo cercano
        joined = "\n".join(lines[:40]).lower()
        for key in ["razon social", "razón social", "nombre o denominacion", "nombre o denominación"]:
            idx = joined.find(key)
            if idx != -1:
                # mira unas líneas después
                chunk = "\n".join(lines[:60])
                # toma líneas "potenciales"
                for ln in chunk.splitlines():
                    up = ln.upper()
                    if any(bad in up for bad in ["RUT", "RUC", "E-FACTURA", "SERIE", "NUMERO", "FORMA", "PAGO", "VENC", "MONEDA"]):
                        continue
                    if len(ln) >= 4 and sum(c.isalpha() for c in ln) >= 4:
                        # Favorecer mayúsculas (muchas facturas vienen así)
                        return ln.strip()
        # 2) primera línea “buena” (y no genérica)
        for ln in lines[:15]:
            up = ln.upper()
            if any(bad in up for bad in ["RUT", "RUC", "E-FACTURA", "FACTURA", "CFE", "SERIE", "NUMERO", "FORMA", "PAGO", "VENC", "MONEDA", "COMPROBANTE"]):
                continue
            if len(ln) >= 4 and sum(c.isalpha() for c in ln) >= 4:
                return ln.strip()

    # Fallback por archivo: "BIMBO A3519972 CREDITO.jpeg" -> "BIMBO"
    stem = Path(filename).stem
    stem = re.sub(r"\s+A\d{4,}.*$", "", stem, flags=re.IGNORECASE).strip()
    if stem:
        return stem

    return None


# ---------- OCR helpers ----------
@dataclass
class OCRResult:
    full_text: str
    total_region_text: str
    header_region_text: str

def _ensure_deps_for_ocr():
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr no está instalado. Instalá con: pip install easyocr")
    if Image is None or np is None:
        raise RuntimeError("Faltan dependencias para OCR (Pillow y/o numpy). Instalá con: pip install pillow numpy")

def _preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    # Blanco y negro suave + contraste, sin ponernos artísticos
    g = ImageOps.grayscale(img)
    g = ImageEnhance.Contrast(g).enhance(1.6)
    g = ImageEnhance.Sharpness(g).enhance(1.2)
    return g

def _ocr_image(reader: "easyocr.Reader", img: "Image.Image", allowlist: Optional[str] = None) -> str:
    arr = np.array(img)
    # detail=0 devuelve solo strings
    # paragraph=True ayuda con layouts raros
    try:
        lines = reader.readtext(arr, detail=0, paragraph=True, allowlist=allowlist)
        return "\n".join([str(x) for x in lines if str(x).strip()])
    except TypeError:
        # Algunas versiones no soportan allowlist en readtext
        lines = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join([str(x) for x in lines if str(x).strip()])

def ocr_with_regions(reader: "easyocr.Reader", image_path: Path) -> OCRResult:
    _ensure_deps_for_ocr()

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Full
    full_img = _preprocess_for_ocr(img)
    full_text = _ocr_image(reader, full_img)

    # Región TOTAL (abajo derecha)
    # Muchas eFacturas ponen el total ahí. Cortamos algo generoso.
    x1 = int(w * 0.55)
    y1 = int(h * 0.55)
    x2 = w
    y2 = h
    total_crop = img.crop((x1, y1, x2, y2))
    total_crop = _preprocess_for_ocr(total_crop)
    total_region_text = _ocr_image(reader, total_crop, allowlist="0123456789.,$/ UYUTOTALtotalapagarPAGARpagar")

    # Región header (arriba) para fecha/serie/número/rut
    x1h = 0
    y1h = 0
    x2h = w
    y2h = int(h * 0.35)
    header_crop = img.crop((x1h, y1h, x2h, y2h))
    header_crop = _preprocess_for_ocr(header_crop)
    header_region_text = _ocr_image(reader, header_crop)

    # Unimos todo: lo regional sirve para extracción (sin perder el full)
    return OCRResult(
        full_text=_clean_spaces(full_text),
        total_region_text=_clean_spaces(total_region_text),
        header_region_text=_clean_spaces(header_region_text),
    )


# ---------- PDF text helpers ----------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Intenta extraer texto real (no OCR) desde PDF.
    """
    text_parts: List[str] = []

    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        text_parts.append(t)
        except Exception:
            pass

    if not text_parts and PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                t = page.get_text("text") or ""
                if t.strip():
                    text_parts.append(t)
        except Exception:
            pass

    if not text_parts and PYPDF_AVAILABLE:
        try:
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
        except Exception:
            pass

    return _clean_spaces("\n".join(text_parts))


# ---------- Core pipeline ----------
SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def list_invoice_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    files.sort(key=lambda x: x.name.lower())
    return files

def parse_one_file(path: Path, reader: Optional["easyocr.Reader"], debug: bool = False) -> Dict[str, Any]:
    filename = path.name
    text = ""
    fuente = None

    total_override = None
    header_text = ""

    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
        fuente = "pdf_text"
        # Si el PDF viene “como imagen” y no hay texto, último recurso: OCR (si hay easyocr)
        if not text.strip() and reader is not None and Image is not None:
            try:
                # Renderizamos la primera página con PyMuPDF si está disponible
                if PYMUPDF_AVAILABLE:
                    doc = fitz.open(str(path))
                    page = doc[0]
                    pix = page.get_pixmap(dpi=250)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # OCR
                    full_img = _preprocess_for_ocr(img)
                    text = _ocr_image(reader, full_img)
                    fuente = "pdf_ocr"
            except Exception:
                pass
    else:
        if reader is None:
            raise RuntimeError("No hay OCR disponible (easyocr). Instalá easyocr o procesá PDFs con texto.")
        ocr = ocr_with_regions(reader, path)
        # Usamos full + header para fecha/rut/serie; y total_region para total
        text = (ocr.full_text + "\n" + ocr.header_region_text).strip()
        header_text = ocr.header_region_text
        fuente = "image_ocr"

        # Total override desde región
        total_override = extract_total_amount(ocr.total_region_text)

    if debug:
        print(f"\n=== {filename} ({fuente}) ===")
        print(text)
        print("=== FIN ===\n", flush=True)

    # Extracciones
    es_nc = extract_es_nota_credito(text, filename)
    rut_emisor = extract_rut_emisor(text)
    razon_social = extract_vendor_name(text, filename)

    serie, folio = extract_serie_folio(text)
    if (not serie) or (not folio):
        s2, f2 = fallback_serie_folio_from_filename(str(path))
        serie = serie or s2
        folio = folio or f2

    fecha_doc = extract_invoice_date(text)
    # Si la fecha viene floja en el OCR (ej '1/10/2025' en vez de '13/10/2025'), intentar apoyar con header
    if (not fecha_doc) and header_text:
        fecha_doc = extract_invoice_date(header_text)

    total = total_override if total_override is not None else extract_total_amount(text)

    neto, neto_fuente = extract_net_amount(text, total)

    record: Dict[str, Any] = {
        "fecha": fecha_doc,
        "serie": serie,
        "folio": folio,
        "serie_y_folio": f"{serie}-{folio}" if serie and folio else None,
        "razon_social": razon_social,
        "rut_emisor": rut_emisor,
        "es_nota_de_credito": bool(es_nc),
        "importe_total_con_iva": format_uy_amount(total),
        "importe_total_con_iva_num": float(total) if total is not None else None,
        "importe_sin_iva": format_uy_amount(neto) if neto is not None else None,
        "importe_sin_iva_num": float(neto) if neto is not None else None,
        "importe_sin_iva_fuente": neto_fuente,
        "_archivo": str(path),
        "_fuente": fuente,
    }

    return record

def build_reader() -> Optional["easyocr.Reader"]:
    if not EASYOCR_AVAILABLE:
        return None
    # 'es' suele andar bien para eFacturas en UY
    # gpu=False para que no explote si no hay CUDA
    return easyocr.Reader(["es"], gpu=False)


# ---------- CLI ----------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parsea facturas (UY) desde PDFs e imágenes.")
    parser.add_argument("path", help="Carpeta o archivo a procesar (ej: C:\\Proyectos\\Facu\\Facturas)")
    parser.add_argument("--json", action="store_true", help="Imprime resultado en JSON al final.")
    parser.add_argument("--debug", action="store_true", help="Imprime el texto extraído por archivo.")
    parser.add_argument("--out", default=None, help="Guarda JSON en archivo (opcional).")
    args = parser.parse_args(argv)

    target = Path(args.path)

    if not target.exists():
        print(f"[WARN] No existe: {target}")
        if args.json:
            print("[]")
        return 0

    files = list_invoice_files(target)
    if not files:
        print("[WARN] No se encontraron archivos soportados.")
        if args.json:
            print("[]")
        return 0

    reader = None
    if any(p.suffix.lower() != ".pdf" for p in files):
        if EASYOCR_AVAILABLE:
            print("Using CPU. Note: This module is much faster with a GPU.")
            reader = build_reader()
        else:
            print("[WARN] Hay imágenes pero easyocr no está instalado. Solo se procesarán PDFs con texto.")
            reader = None

    results: List[Dict[str, Any]] = []
    for p in files:
        try:
            # Si es imagen y no hay OCR, saltar
            if p.suffix.lower() != ".pdf" and reader is None:
                continue
            rec = parse_one_file(p, reader, debug=args.debug)
            results.append(rec)
        except Exception as e:
            # No reventar por una factura rota, porque los humanos aman el caos
            err = {
                "_archivo": str(p),
                "_error": str(e),
            }
            results.append(err)
            if args.debug:
                print(f"[ERROR] {p.name}: {e}", file=sys.stderr)

    if args.json or args.out:
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        if args.json:
            print(payload)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
