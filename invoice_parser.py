# ruff: noqa: E501
r"""
invoice_parser.py

Extractor (UY) de metadatos de comprobantes desde:
- PDFs con texto (pdfplumber / PyPDF2)
- Imágenes (OCR con EasyOCR si está disponible; si no, intenta pytesseract si hay Tesseract instalado)

Campos:
- fecha
- serie, folio, serie_y_folio
- razon_social (preferentemente desde el nombre de archivo; fallback por texto)
- rut_emisor (scoring para no confundir con rut receptor/cliente)
- es_nota_de_credito (true si detecta "NOTA DE CRÉDITO" en texto o nombre)
- importe_total_con_iva (+ num)
- importe_sin_iva (+ num + fuente) (solo si es consistente, no inventa)

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
else:
    pdfplumber = None  # type: ignore

pypdf2_spec = importlib.util.find_spec("PyPDF2")
if pypdf2_spec:
    from PyPDF2 import PdfReader  # type: ignore
else:
    PdfReader = None  # type: ignore

numpy_spec = importlib.util.find_spec("numpy")
if numpy_spec:
    import numpy as np  # type: ignore
else:
    np = None  # type: ignore

pil_spec = importlib.util.find_spec("PIL.Image")
if pil_spec:
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore
else:
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore

easyocr_spec = importlib.util.find_spec("easyocr")
if easyocr_spec:
    import easyocr  # type: ignore
else:
    easyocr = None  # type: ignore

pytesseract_spec = importlib.util.find_spec("pytesseract")
if pytesseract_spec:
    import pytesseract  # type: ignore
else:
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
# Regex base
# ---------------------------

DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
# Ej: 4.919,61  | 701,00  | 8516,99
MONEY_RE = re.compile(r"(?<!\d)(-?\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})|-?\d+(?:,\d{2}))(?!\d)")
# Ej: OCR sucio sin separadores: 401247 -> 4012,47 (solo como fallback)
MONEY_LOOSE_RE = re.compile(r"(?<!\d)(-?\d{4,})(?!\d)")
# RUT 11/12 dígitos o con separadores
RUT_RE = re.compile(r"\b(\d{11,12}|\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{3})\b")

# Palabras que suelen delatar “tabla de items” (ahí es donde aparece el trampa 1,00 / 2,00 etc.)
TABLE_WORDS = (
    "PIEZ", "PIEZA", "CANT", "CANTIDAD", "UNID", "UNIDAD", "ENVASE", "CODIGO", "CÓDIGO",
    "PRODUCTO", "DESCRIP", "UNITAR", "DESCUENTO", "PRECIO", "IMPORTE"
)

# ---------------------------
# Helpers texto
# ---------------------------

def _safe_upper(s: str) -> str:
    return (s or "").upper()

def _collapse_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())

def normalize_text_block(text: str) -> str:
    lines = [_collapse_spaces(line) for line in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def _strip_currency_and_noise(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00a0", " ")
    s = s.replace("UYU", "").replace("U$S", "").replace("USD", "").replace("$", "")
    s = s.replace("PESOS", "").replace("PESO", "")
    s = s.replace(" ", "")
    return s

def _fix_ocr_digit_garbage(token: str) -> str:
    """
    Corrige algunas confusiones típicas OCR dentro de tokens numéricos.
    Importante: solo se aplica a tokens que ya “parecen” un número.
    """
    t = token
    # Reemplazos seguros-ish
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace("I", "1").replace("l", "1").replace("|", "1")
    t = t.replace("S", "5").replace("s", "5")
    t = t.replace("B", "8")
    # Mantener separadores
    return t

def parse_money_flexible(s: str) -> Optional[float]:
    """
    Parse robusto de montos estilo UY, tolerando cosas como:
    - "3,129,00"  (mal) -> 3129.00
    - "4.919,61" -> 4919.61
    - "9.261,00" -> 9261.00
    - "8516,99"  -> 8516.99
    - "401247"   -> 4012.47 (fallback)
    """
    if not s:
        return None

    raw = _strip_currency_and_noise(s)
    raw = _fix_ocr_digit_garbage(raw)
    raw = re.sub(r"[^0-9,.\-]", "", raw)
    if not re.search(r"\d", raw):
        return None

    neg = raw.startswith("-")
    if neg:
        raw = raw[1:]

    # Caso con múltiples comas: usar la última como decimal si termina en 2 dígitos
    if raw.count(",") >= 2 and "." not in raw:
        parts = raw.split(",")
        if len(parts[-1]) == 2:
            raw = "".join(parts[:-1]) + "," + parts[-1]
        else:
            raw = "".join(parts)

    # Caso con múltiples puntos: usar el último como decimal si termina en 2 dígitos (rarísimo en UY)
    if raw.count(".") >= 2 and "," not in raw:
        parts = raw.split(".")
        if len(parts[-1]) == 2:
            raw = "".join(parts[:-1]) + "." + parts[-1]
        else:
            raw = "".join(parts)

    # Si tiene coma y punto, decidir decimal por el último separador
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            # 4.919,61 -> 4919.61
            raw2 = raw.replace(".", "").replace(",", ".")
        else:
            # 4,919.61 (no UY, pero por las dudas)
            raw2 = raw.replace(",", "")
        try:
            v = float(raw2)
            return -v if neg else v
        except Exception:
            pass

    # Solo coma: coma decimal si termina en 2 dígitos
    if "," in raw and "." not in raw:
        parts = raw.split(",")
        if len(parts[-1]) == 2:
            raw2 = "".join(parts[:-1]) + "." + parts[-1]
        else:
            raw2 = "".join(parts)
        try:
            v = float(raw2)
            return -v if neg else v
        except Exception:
            pass

    # Solo punto: sospechar miles y/o decimal
    if "." in raw and "," not in raw:
        parts = raw.split(".")
        if len(parts[-1]) == 2:
            raw2 = "".join(parts[:-1]) + "." + parts[-1]
        else:
            raw2 = "".join(parts)
        try:
            v = float(raw2)
            return -v if neg else v
        except Exception:
            pass

    # Fallback: “monto sin separadores”, interpretar últimos 2 dígitos como centésimos
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 4:
        whole = digits[:-2] or "0"
        frac = digits[-2:]
        try:
            v = float(f"{int(whole)}.{int(frac):02d}")
            return -v if neg else v
        except Exception:
            return None

    return None

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


# ---------------------------
# Derivaciones por filename
# ---------------------------

def derive_razon_social_from_filename(path: Path) -> Optional[str]:
    """
    "DEL SUR NOTA DE CREDITO A18911 CREDITO.jpeg" -> "DEL SUR"
    "BIMBO A3519972 CREDITO.jpeg" -> "BIMBO"
    """
    stem = _safe_upper(path.stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = _collapse_spaces(stem)

    # Cortar en " A123..."
    stem = re.split(r"\s+[A-Z]\d{3,}\b", stem)[0]  # ej "BIMBO A351..." => BIMBO
    stem = stem.replace("NOTA DE CREDITO", "").replace("NOTA DE CRÉDITO", "").strip()
    stem = re.sub(r"\s+CREDITO\b", "", stem).strip()
    stem = _collapse_spaces(stem)
    return stem or None

def derive_serie_folio_from_filename(path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Busca algo como: "A3519972" en el nombre del archivo.
    """
    name = _safe_upper(path.stem)
    m = re.search(r"\b([A-Z])\s*0*([0-9]{3,})\b", name)
    if m:
        return m.group(1), m.group(2).lstrip("0") or "0"
    return None, None


# ---------------------------
# Clasificación
# ---------------------------

def is_credit_note(text: str, path: Path) -> bool:
    x = _safe_upper(text) + " " + _safe_upper(path.name)
    return ("NOTA DE CREDITO" in x) or ("NOTA DE CRÉDITO" in x)


# ---------------------------
# Extracción fecha / serie / folio
# ---------------------------

def pick_best_date(text: str) -> Optional[str]:
    if not text:
        return None

    best = None
    best_score = -1e9
    up = _safe_upper(text)

    for m in DATE_RE.finditer(up):
        d = m.group(1)
        i = m.start()
        ctx = up[max(0, i - 60) : i + 60]

        score = 0
        if "FECHA DE DOCUMENTO" in ctx:
            score += 20
        elif re.search(r"\bFECHA\b", ctx):
            score += 10

        # penalizaciones
        if "VENC" in ctx or "VTO" in ctx:
            score -= 18
        if "CAE" in ctx or "RANGO" in ctx:
            score -= 10
        if "FECHA EMISOR" in ctx or "EMISOR:" in ctx:
            score -= 12

        # bonus por aparecer relativamente arriba (en facturas suele estar arriba)
        score += max(0.0, (3000 - i) / 3000.0)

        if score > best_score:
            best_score = score
            best = d

    return best

def extract_serie_folio(text: str, path: Path) -> tuple[Optional[str], Optional[str]]:
    if text:
        m = re.search(
            r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMERO\b[^0-9]{0,20}(\d{3,})",
            text,
            flags=re.I | re.S,
        )
        if m:
            return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

        m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\s*0*([0-9]{3,})\b", text, flags=re.I)
        if m:
            return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

        m = re.search(r"\bNUMERO\b[^A-Z0-9]{0,20}([A-Z])\s*0*([0-9]{3,})\b", text, flags=re.I)
        if m:
            return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

        # fallback: "A 049009" etc
        m = re.search(r"\b([A-Z])\s*-?\s*0*([0-9]{3,})\b", text, flags=re.I)
        if m:
            return m.group(1).upper(), (m.group(2).lstrip("0") or "0")

    # filename fallback
    return derive_serie_folio_from_filename(path)


# ---------------------------
# Extracción razón social (fallback)
# ---------------------------

def extract_razon_social_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    up = normalize_text_block(_safe_upper(text))

    # Típico PDF: "RAZON SOCIAL 2188... MODELO NATURAL SRL"
    m = re.search(r"\bRAZON\s+SOCIAL\b[^\n]{0,80}\n?([A-Z0-9 .,'\-()]{3,80})", up)
    if m:
        cand = _collapse_spaces(m.group(1))
        # evitar que agarre un RUT/numero
        if not re.fullmatch(r"[0-9 .\-]{6,}", cand):
            return cand

    # Otras etiquetas
    m = re.search(r"\bDENOMINACION\b[^\n]{0,60}\n?([A-Z0-9 .,'\-()]{3,80})", up)
    if m:
        cand = _collapse_spaces(m.group(1))
        if not re.fullmatch(r"[0-9 .\-]{6,}", cand):
            return cand

    return None


# ---------------------------
# Extracción RUT EMISOR (scoring)
# ---------------------------

def _clean_rut(raw: str) -> str:
    return re.sub(r"[^0-9]", "", raw or "")

def extract_rut_emisor(text: str) -> Optional[str]:
    """
    Score por proximidad a "RUT/RUC EMISOR", penaliza receptor/comprador.
    """
    if not text:
        return None

    up = _safe_upper(text)

    candidates: list[tuple[int, str]] = []

    for m in RUT_RE.finditer(up):
        raw = m.group(1)
        rut = _clean_rut(raw)
        if len(rut) not in (11, 12):
            continue

        i = m.start()
        ctx = up[max(0, i - 80) : i + 80]

        score = 0
        if re.search(r"\bRU[TC]\s*EMISOR\b", ctx):
            score += 40
        if re.search(r"\bEMISOR\b", ctx):
            score += 15
        if re.search(r"\bRUC\b", ctx) or re.search(r"\bRUT\b", ctx):
            score += 8

        # penalizar receptor/comprador
        if "RECEPTOR" in ctx or "COMPRADOR" in ctx or "CLIENTE" in ctx:
            score -= 35
        if re.search(r"\bRU[TC]\s*RECEPTOR\b", ctx):
            score -= 50

        # Bonus si aparece relativamente arriba (en muchos comprobantes el rut emisor está arriba)
        score += max(0, (2500 - i) // 200)  # escalonado

        candidates.append((score, rut))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


# ---------------------------
# Extracción montos (total / iva / sin iva)
# ---------------------------

def _find_money_tokens(text: str) -> list[tuple[float, int, str]]:
    """
    Devuelve lista de (valor, posición, token_original) para tokens tipo dinero.
    Usa MONEY_RE y fallback MONEY_LOOSE_RE.
    """
    if not text:
        return []

    out: list[tuple[float, int, str]] = []

    for m in MONEY_RE.finditer(text):
        token = m.group(1)
        v = parse_money_flexible(token)
        if v is not None:
            out.append((v, m.start(), token))

    # Fallback “loose”: solo si no encontramos casi nada
    if len(out) < 2:
        for m in MONEY_LOOSE_RE.finditer(text):
            token = m.group(1)
            # evitar años muy obvios 2025 etc
            if token.lstrip("-") in {"2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028"}:
                continue
            v = parse_money_flexible(token)
            if v is not None:
                out.append((v, m.start(), token))

    return out

def extract_total_with_scoring(text: str) -> Optional[float]:
    """
    Busca TOTAL A PAGAR / TOTAL / TOTAL: y puntúa para evitar capturar "1,00" en tablas.
    """
    if not text:
        return None

    up = _safe_upper(text)

    patterns = [
        (r"\bTOTAL\s*A\s*PAGAR\b", 120),
        (r"\bTOTAL\s+PAGAR\b", 110),
        (r"\bIMPORTE\s+TOTAL\b", 105),
        (r"\bTOTAL\s*:\b", 95),
        (r"\bTOTAL\b", 70),
    ]

    money_pat = r"(" + MONEY_RE.pattern + r"|" + MONEY_LOOSE_RE.pattern + r")"

    candidates: list[tuple[int, float]] = []

    for label, base_score in patterns:
        for m in re.finditer(label, up, flags=re.I):
            i = m.start()
            ctx = up[max(0, i - 90) : i + 140]

            # ignorar "TOTAL IVA"
            if re.search(r"\bTOTAL\s*IVA\b", ctx):
                continue

            # buscar dinero cerca
            near = up[m.end() : m.end() + 140]
            mm = re.search(money_pat, near)
            if not mm:
                # a veces el monto está antes en la misma línea (OCR raro)
                before = up[max(0, i - 140) : i]
                mm2 = re.search(money_pat + r"\s*$", before)
                mm = mm2

            if not mm:
                continue

            token = mm.group(0)
            v = parse_money_flexible(token)
            if v is None:
                continue

            score = base_score

            # Preferir cosas "al final" del documento (los totales suelen estar abajo)
            score += int(min(30, (i / max(1, len(up))) * 30))

            # Penalizar contexto de tabla
            if any(w in ctx for w in TABLE_WORDS):
                score -= 70

            # Penalizar montos ridículamente chicos salvo que sea el único
            if abs(v) < 10:
                score -= 40

            # Bonus si aparece "A PAGAR" en ctx aunque el label no lo sea
            if "A PAGAR" in ctx:
                score += 25

            candidates.append((score, v))

    if not candidates:
        # fallback: elegir el mayor monto "plausible" (evita agarrar 401247 como 4012.47 si hay 8516.99 etc)
        vals = _find_money_tokens(up)
        if not vals:
            return None
        # Filtrar valores muy chicos
        plausible = [v for (v, _, _) in vals if abs(v) >= 10]
        if plausible:
            return max(plausible)
        return max(v for (v, _, _) in vals)

    candidates.sort(key=lambda t: (t[0], abs(t[1])), reverse=True)
    return candidates[0][1]

def extract_iva_total(text: str) -> Optional[float]:
    """
    Intenta:
    - "Total IVA (22%)"
    - "IVA 22%"
    - "Total iva (10%)"
    - "Total IVA" (genérico)
    Devuelve suma 10%+22% si corresponde.
    """
    if not text:
        return None

    up = _safe_upper(text)

    def money_near(label_pattern: str, window: int = 160) -> Optional[float]:
        m = re.search(label_pattern, up, flags=re.I)
        if not m:
            return None
        snippet = up[m.end() : m.end() + window]
        mm = re.search(r"(" + MONEY_RE.pattern + r"|" + MONEY_LOOSE_RE.pattern + r")", snippet)
        if not mm:
            return None
        return parse_money_flexible(mm.group(0))

    # Caso directo
    v = money_near(r"\bTOTAL\s*IVA\b")
    # Si aparece "TOTAL IVA" pero en realidad la línea suele ser "Total iva (22%)"
    iva10 = money_near(r"\bTOTAL\s*IVA\b[^\n]{0,40}10")
    iva22 = money_near(r"\bTOTAL\s*IVA\b[^\n]{0,40}22")

    # Alternativas "IVA 10%" "IVA 22%"
    if iva10 is None:
        iva10 = money_near(r"\bI\.?V\.?A\.?\s*10")
    if iva22 is None:
        iva22 = money_near(r"\bI\.?V\.?A\.?\s*22")

    # Si conseguimos desglosado, mejor que el genérico
    if iva10 is not None or iva22 is not None:
        return (iva10 or 0.0) + (iva22 or 0.0)

    return v

def extract_sin_iva_candidate(text: str) -> Optional[float]:
    """
    Busca etiquetas típicas para monto sin IVA / neto:
    - TOTAL SIN IVA
    - IMPORTE NETO
    - SUBTOTAL (cuando realmente es neto)
    - SUBTOTAL GRAVADO (22%) + (10%) (si hay suma)
    """
    if not text:
        return None
    up = _safe_upper(text)

    def money_after(label: str) -> Optional[float]:
        m = re.search(label, up, flags=re.I)
        if not m:
            return None
        snippet = up[m.end() : m.end() + 180]
        mm = re.search(r"(" + MONEY_RE.pattern + r"|" + MONEY_LOOSE_RE.pattern + r")", snippet)
        if not mm:
            return None
        return parse_money_flexible(mm.group(0))

    # Etiquetas directas
    for label in [
        r"\bTOTAL\s+SIN\s+IVA\b",
        r"\bIMPORTE\s+NETO\b",
        r"\bMONTO\s+NETO\b",
        r"\bNETO\s+GRAVADO\b",
    ]:
        v = money_after(label)
        if v is not None:
            return v

    # Subtotal gravado (22%/10%) puede existir. Si están los dos, sumarlos.
    g22 = money_after(r"\bSUBTOTAL\s+GRAVADO\b[^\n]{0,20}22")
    g10 = money_after(r"\bSUBTOTAL\s+GRAVADO\b[^\n]{0,20}10")
    if g22 is not None or g10 is not None:
        return (g22 or 0.0) + (g10 or 0.0)

    # SUBTOTAL genérico (último recurso)
    v = money_after(r"\bSUBTOTAL\b")
    return v

def is_consistent(total: Optional[float], sin_iva: Optional[float], iva: Optional[float], tol_abs: float = 2.0, tol_rel: float = 0.02) -> bool:
    """
    Verifica si sin_iva + iva ~= total.
    """
    if total is None or sin_iva is None or iva is None:
        return False
    expected = sin_iva + iva
    diff = abs(expected - total)
    if diff <= tol_abs:
        return True
    if total != 0 and (diff / abs(total)) <= tol_rel:
        return True
    return False

def compute_importe_sin_iva(total: Optional[float], iva_total: Optional[float], text: str) -> tuple[Optional[float], Optional[str]]:
    """
    Prioridad:
    1) Si hay etiqueta de sin IVA y es consistente con IVA+total -> usarla
    2) Si no, si total e IVA son confiables -> total - IVA
    Si no “cierra”, devolvemos None (porque inventar es para humanos).
    """
    sin_iva_label = extract_sin_iva_candidate(text)

    if sin_iva_label is not None and is_consistent(total, sin_iva_label, iva_total):
        return round(sin_iva_label, 2), "label_consistente"

    if total is not None and iva_total is not None:
        cand = round(total - iva_total, 2)
        # validación mínima: si sale negativo raro, no lo uses
        if cand >= 0:
            return cand, "total_menos_iva"

    return None, None


# ---------------------------
# OCR backend
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
            # ES + EN por si metió textos mixtos
            self._easy_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)  # type: ignore
        return self._easy_reader

    def _preprocess_image(self, img):
        img = ImageOps.exif_transpose(img)  # type: ignore
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.2)
        img = ImageEnhance.Sharpness(img).enhance(1.3)
        return img

    def ocr_image_easy(self, img) -> str:
        reader = self._get_easy_reader()
        arr = np.array(img.convert("RGB"))  # type: ignore
        lines = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(lines)

    def ocr_image_tess(self, img) -> str:
        cfg = "--oem 3 --psm 6"
        # spa suele ser mejor para tus docs
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
            except Exception as exc:
                errors.append(f"{backend}: {exc}")

        if errors:
            raise RuntimeError("; ".join(errors))
        raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract.")


def crop_rel(img, l: float, t: float, r: float, b: float):
    w, h = img.size
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


# ---------------------------
# PDF text extractor
# ---------------------------

def extract_text_from_pdf(path: Path) -> tuple[str, str]:
    if pdfplumber is not None:
        try:
            chunks: list[str] = []
            with pdfplumber.open(str(path)) as pdf:  # type: ignore
                for page in pdf.pages:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        chunks.append(txt)
            out = normalize_text_block("\n".join(chunks))
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
            out = normalize_text_block("\n".join(chunks))
            if out:
                return out, "pdf_text"
        except Exception:
            pass

    return "", "pdf_text"


# ---------------------------
# Procesadores (img/pdf)
# ---------------------------

def process_image(path: Path, ocr: OCRBackend, debug: bool) -> tuple[str, str, str, str]:
    if Image is None:
        raise RuntimeError("Falta pillow (PIL) para leer imágenes.")
    img = Image.open(path)

    # Crop header (donde suele estar RUT/serie/fecha)
    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.42)

    # Candidatos de “zona totales”: probamos 2 recortes y combinamos
    totals_img_1 = crop_rel(img, 0.40, 0.55, 1.00, 1.00)  # bottom-right más grande
    totals_img_2 = crop_rel(img, 0.00, 0.65, 1.00, 1.00)  # banda inferior completa

    text_full, fuente = ocr.ocr_image(img)
    text_header, _ = ocr.ocr_image(header_img)
    txt1, _ = ocr.ocr_image(totals_img_1)
    txt2, _ = ocr.ocr_image(totals_img_2)

    # Mezclamos totales para aumentar chance de encontrar “Total a pagar”
    text_totals = normalize_text_block("\n".join([txt1, txt2]))

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


# ---------------------------
# Parse final
# ---------------------------

def parse_invoice_from_text(
    text_full: str,
    text_header: str,
    text_totals: str,
    path: Path,
    fuente: str,
) -> InvoiceResult:
    es_nc = is_credit_note(text_full, path)

    # razón social: filename fuerte, fallback texto
    razon = derive_razon_social_from_filename(path) or extract_razon_social_from_text(text_header or text_full)

    # serie/folio: header fuerte, fallback filename
    serie, folio = extract_serie_folio(text_header or text_full, path)
    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_emisor = extract_rut_emisor(text_header or text_full)
    fecha = pick_best_date(text_header or text_full)

    # Total e IVA: preferimos TOTAlS crop, luego full
    total_num = extract_total_with_scoring(text_totals) or extract_total_with_scoring(text_full)
    iva_total = extract_iva_total(text_totals) or extract_iva_total(text_full)

    # sin IVA con validación (no inventa si no cierra)
    sin_iva_num, sin_iva_fuente = compute_importe_sin_iva(total_num, iva_total, text_totals + "\n" + text_full)

    # Si es nota de crédito: total y sin IVA deben quedar en negativo
    if es_nc:
        if total_num is not None and total_num > 0:
            total_num = -total_num
        if sin_iva_num is not None and sin_iva_num > 0:
            sin_iva_num = -sin_iva_num

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


# ---------------------------
# IO
# ---------------------------

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
