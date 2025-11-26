# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Opcionales (lazy-ish)
# ---------------------------
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from PIL import Image, ImageEnhance, ImageOps, ImageFile  # type: ignore
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageOps = None  # type: ignore


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


DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
MONEY_RE = re.compile(r"\b(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))\b")
MONEY_RE_LOOSE = re.compile(r"\b(-?\d{4,})\b")  # OCR sucio (401247 -> 4012.47)
RUT_RE = re.compile(r"\b(\d{11,12})\b")


# ---------------------------
# Utilidades base
# ---------------------------
def _safe_upper(s: str) -> str:
    return (s or "").upper()


def _collapse_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())


def normalize_text_block(text: str) -> str:
    lines = [_collapse_spaces(line) for line in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def normalize_digit_group_spaces(text: str) -> str:
    """
    Arregla "4 018,01" (espacio como separador de miles) a "4018,01".
    Solo elimina espacios si después vienen EXACTAMENTE 3 dígitos.
    """
    if not text:
        return text
    # repetimos porque "1 234 567,89" tiene 2 espacios relevantes
    prev = None
    cur = text
    for _ in range(4):
        prev = cur
        cur = re.sub(r"(?<=\d)\s(?=\d{3}(\D|$))", "", cur)
        if cur == prev:
            break
    return cur


def _only_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _parse_ddmmyyyy(s: str) -> Optional[date]:
    try:
        d, m, y = s.split("/")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def parse_money_uy(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    if not re.search(r"\d", s):
        return None
    s = normalize_digit_group_spaces(s)
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


def parse_money_uy_loose(s: str) -> Optional[float]:
    # "401247" -> 4012.47, pero evita comerse IDs largos
    if not s:
        return None
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    digits = _only_digits(s)
    if not digits:
        return None
    if len(digits) > 9:  # anti RUT/CAE
        return None
    if len(digits) < 4:
        return None
    whole = digits[:-2] or "0"
    frac = digits[-2:]
    try:
        v = float(f"{int(whole)}.{int(frac):02d}")
        return -v if neg else v
    except Exception:
        return None


def parse_money_token(s: str) -> Optional[float]:
    v = parse_money_uy(s)
    if v is not None:
        return v
    return parse_money_uy_loose(s)


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
# Extracción: fecha / RUT / serie-folio / razón social
# ---------------------------
def pick_best_date(text: str) -> Optional[str]:
    if not text:
        return None

    text = normalize_digit_group_spaces(text)
    today = date.today()
    min_ok = today - timedelta(days=365 * 10)
    max_ok = today + timedelta(days=370)  # mata CAE 2027 si estás en 2025

    best: Optional[str] = None
    best_score = -1e9

    for m in DATE_RE.finditer(text):
        dstr = m.group(1)
        idx = m.start()
        dobj = _parse_ddmmyyyy(dstr)
        if dobj is None or dobj < min_ok or dobj > max_ok:
            continue

        score = 0.0
        window = text[max(0, idx - 50) : min(len(text), idx + 50)]
        wup = _safe_upper(window)

        if re.search(r"FECHA\s+DE\s+DOCUMENTO", wup):
            score += 90
        elif re.search(r"\bFECHA\b", wup):
            score += 25

        if re.search(r"\bVENC(?:IMIENTO)?\b|\bVTO\b", wup):
            score -= 60
        if re.search(r"\bCAE\b|\bRANGO\s+DE\s+CAE\b|\bFECHA\s+EMISOR\b", wup):
            score -= 80

        score += max(0, 2500 - idx) / 2500.0

        if score > best_score:
            best_score = score
            best = dstr

    return best


def extract_rut_emisor(text: str) -> Optional[str]:
    if not text:
        return None
    up = _safe_upper(text)

    m = re.search(r"\bRUT\s*EMISOR\b[^0-9]{0,40}(\d{11,12})", up)
    if m:
        return _only_digits(m.group(1))

    m = re.search(r"\bRUC\b[^0-9]{0,30}(\d{11,12})", up)
    if m:
        return _only_digits(m.group(1))

    m = RUT_RE.search(up)
    return _only_digits(m.group(1)) if m else None


def extract_serie_folio(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None

    t = normalize_digit_group_spaces(text)
    up = _safe_upper(t)

    # SERIE A NUMERO 049009
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b[^A-Z0-9]{0,40}\bNUMER[O0]\b[^0-9]{0,25}0*([0-9][0-9\s]{3,})\b", up, flags=re.S)
    if m:
        serie = m.group(1).upper()
        folio = _only_digits(m.group(2)).lstrip("0") or "0"
        if len(folio) >= 4:
            return serie, folio

    # SERIE A 049009 (sin "NUMERO")
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b[^0-9]{0,80}0*([0-9][0-9\s]{3,})\b", up, flags=re.S)
    if m:
        serie = m.group(1).upper()
        folio = _only_digits(m.group(2)).lstrip("0") or "0"
        if len(folio) >= 4:
            return serie, folio

    # SERIE 3519972 (le falta la letra, típico OCR)
    m = re.search(r"\bSERIE\b[^0-9]{0,40}0*([0-9][0-9\s]{3,})\b", up, flags=re.S)
    if m:
        folio = _only_digits(m.group(1)).lstrip("0") or "0"
        if len(folio) >= 4:
            return None, folio

    # "e-Factura .... 3519973" o "Tipo de Documento .... 3519972" (solo si está cerca del top)
    head = "\n".join(normalize_text_block(t).splitlines()[:25])
    hup = _safe_upper(head)
    m = re.search(r"\b(E-?FACTURA|FACTURA|TIPO\s+DE\s+DOCUMENTO)\b[^0-9]{0,60}0*([0-9][0-9\s]{3,})\b", hup, flags=re.S)
    if m:
        folio = _only_digits(m.group(2)).lstrip("0") or "0"
        if len(folio) >= 4:
            return None, folio

    return None, None


_STOP_LABELS = {
    "DENOMINACION",
    "DENOMINACIÓN",
    "NOMBRE",
    "RAZON SOCIAL",
    "RAZÓN SOCIAL",
    "DOMICILIO FISCAL",
    "DIRECCION",
    "DIRECCIÓN",
    "SUCURSAL",
    "CREDITO",
    "CRÉDITO",
    "FORMA DE PAGO",
    "MONEDA",
    "VENCIMIENTO",
    "FECHA",
    "RUC",
    "RUT",
    "DOCUMENTO",
    "TIPO DE DOCUMENTO",
    "SERIE",
    "NUMERO",
    "NÚMERO",
    "PRODUCTO",
    "CODIGO",
    "CÓDIGO",
    "TOTAL",
    "TOTAL A PAGAR",
    "IVA",
}


def _looks_like_label_only(s: str) -> bool:
    u = _safe_upper(_collapse_spaces(s))
    if not u:
        return True
    # si es exactamente un rótulo conocido, afuera
    if u in _STOP_LABELS:
        return True
    # si solo son palabras de rótulo
    if all(w in _STOP_LABELS for w in u.split()):
        return True
    return False


def extract_razon_social_emisor(text_full: str, text_header: str) -> Optional[str]:
    """
    Intenta sacar el EMISOR (no el comprador).
    """
    full = text_full or ""
    header = text_header or ""
    full_n = normalize_text_block(full)
    header_n = normalize_text_block(header)

    # 1) Estilo PDF: "<EMISOR> RUT RECEPTOR"
    m = re.search(r"\b(.{3,80}?)\s+RUT\s+RECEPTOR\b", full_n, flags=re.I)
    if m:
        cand = _collapse_spaces(m.group(1)).strip(" -:")
        if cand and not _looks_like_label_only(cand):
            return cand

    # 2) Heurística con header: buscar candidato ANTES de "COMPRADOR/RECEPTOR"
    lines = header_n.splitlines()
    cut = len(lines)
    for i, ln in enumerate(lines):
        if re.search(r"\b(COMPRADOR|RECEPTOR)\b", _safe_upper(ln)):
            cut = i
            break
    scope = lines[:cut] if cut > 0 else lines[:25]

    def score_line(ln: str, idx: int) -> int:
        s = _collapse_spaces(ln)
        u = _safe_upper(s)
        if not s or len(s) < 3 or len(s) > 60:
            return -999
        if _looks_like_label_only(s):
            return -999
        if re.search(r"\d{6,}", s):  # demasiados dígitos: probable RUT/folio/teléfono
            return -999
        if "MODELO NATURAL" in u:  # tu empresa (comprador), no emisor
            return -800
        if re.search(r"\bSRL\b|\bS\.R\.L\b", u):  # suele ser comprador acá
            return -200

        bad_words = [
            "TIMOTE",
            "SUCURSAL",
            "CREDITO",
            "FORMA",
            "PAGO",
            "VENCIMIENTO",
            "MONEDA",
            "LUGAR",
            "ORDEN",
        ]
        if any(w in u for w in bad_words):
            return -150

        letters = sum(ch.isalpha() for ch in s)
        score = 0
        score += min(50, letters * 2)
        if u == s:  # todo mayúscula
            score += 15
        score += max(0, 30 - idx)  # preferir arriba
        return score

    best = None
    best_score = -999
    for i, ln in enumerate(scope[:30]):
        sc = score_line(ln, i)
        if sc > best_score:
            best_score = sc
            best = _collapse_spaces(ln).strip(" -:")

    if best and best_score > 10:
        return best

    return None


def is_credit_note(text: str, path: Path) -> bool:
    x = _safe_upper(text) + " " + _safe_upper(path.name)
    return ("NOTA DE CREDITO" in x) or ("NOTA DE CRÉDITO" in x)


def extract_amount_after_label(text: str, label_regex: str) -> Optional[float]:
    if not text:
        return None
    t = normalize_digit_group_spaces(text)
    up = _safe_upper(t)
    m = re.search(label_regex + r"[^0-9\-]{0,40}(" + MONEY_RE.pattern + r")", up, flags=re.S)
    if not m:
        return None
    return parse_money_token(m.group(1))


def detect_vat_rates(text: str) -> set[float]:
    """
    Detecta tasas "activas":
    - si aparece 10% pero el subtotal/IVA de 10% es 0, se ignora (caso PDF típico)
    """
    up = _safe_upper(normalize_digit_group_spaces(text or ""))

    sub10 = extract_amount_after_label(up, r"SUBTOTAL\s+GRAVADO\(?\s*10\s*%\)?")
    iva10 = extract_amount_after_label(up, r"TOTAL\s+IVA\s*\(?\s*10\s*%\)?")
    sub22 = extract_amount_after_label(up, r"SUBTOTAL\s+GRAVADO\(?\s*22\s*%\)?")
    iva22 = extract_amount_after_label(up, r"TOTAL\s+IVA\s*\(?\s*22\s*%\)?")

    rates: set[float] = set()

    if (sub10 is not None and sub10 > 0.01) or (iva10 is not None and iva10 > 0.01):
        rates.add(0.10)
    if (sub22 is not None and sub22 > 0.01) or (iva22 is not None and iva22 > 0.01):
        rates.add(0.22)

    # fallback para OCR de imágenes (no siempre tenemos subtotales legibles)
    if not rates:
        if re.search(r"\b22\s*%|\bIVA\s*22", up):
            rates.add(0.22)
        if re.search(r"\b10\s*%|\bIVA\s*10", up):
            rates.add(0.10)

    return rates


# ---------------------------
# Totales / IVA / neto
# ---------------------------
def extract_iva_total(text: str) -> Optional[float]:
    if not text:
        return None
    up = _safe_upper(normalize_digit_group_spaces(text))

    m = re.search(r"\bTOTAL\s*IVA\b[^0-9\-]{0,25}(" + MONEY_RE.pattern + r")", up)
    if m:
        return parse_money_token(m.group(1))

    iva22 = extract_amount_after_label(up, r"TOTAL\s+IVA\s*\(?\s*22\s*%\)?")
    iva10 = extract_amount_after_label(up, r"TOTAL\s+IVA\s*\(?\s*10\s*%\)?")

    if iva22 is None and iva10 is None:
        return None
    return round((iva22 or 0.0) + (iva10 or 0.0), 2)


def extract_total(text: str) -> Optional[float]:
    if not text:
        return None

    t = normalize_digit_group_spaces(text)

    candidates: list[tuple[float, int]] = []

    def collect(pattern: str, score: int) -> None:
        for m in re.finditer(pattern, t, flags=re.I | re.S):
            v = parse_money_token(m.group(1))
            if v is not None:
                candidates.append((v, score))

    collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,60}(" + MONEY_RE.pattern + r")", 140)
    collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,60}(" + MONEY_RE.pattern + r")", 110)
    collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,60}(" + MONEY_RE_LOOSE.pattern + r")", 90)

    # fallback: mayor monto "normal" en el bloque (pero evita elegir 785,06 si hay 4018,01)
    vals = [parse_money_token(m.group(1)) for m in MONEY_RE.finditer(t)]
    vals = [v for v in vals if v is not None]
    if vals:
        candidates.append((max(vals), 30))

    if not candidates:
        return None

    # ordenar por score y valor
    candidates.sort(key=lambda it: (it[1], it[0]), reverse=True)

    best_v = candidates[0][0]

    # si por alguna razón agarró un "018,01" pedorro, pero hay algo grande, usar lo grande
    bigs = [v for (v, sc) in candidates if v >= 1000]
    if best_v < 100 and bigs:
        return max(bigs)

    return best_v


def compute_importe_sin_iva(total: Optional[float], iva_total: Optional[float], vat_rates: set[float], text_full: str) -> tuple[Optional[float], Optional[str]]:
    if total is None:
        return None, None

    # Si IVA total parece consistente, preferir total - IVA
    if iva_total is not None and 0 <= iva_total < total:
        net = round(total - iva_total, 2)
        rate = iva_total / max(net, 0.01)
        if vat_rates:
            if any(abs(rate - r) <= 0.03 for r in vat_rates):
                return net, "total_menos_iva"
        else:
            if 0.07 <= rate <= 0.25:
                return net, "total_menos_iva"

    # si detectamos 1 sola tasa activa, dividir
    if vat_rates == {0.22}:
        return round(total / 1.22, 2), "total_div_22"
    if vat_rates == {0.10}:
        return round(total / 1.10, 2), "total_div_10"

    return None, None


# ---------------------------
# Limitación de recursos
# ---------------------------
def apply_thread_limits(n: int) -> None:
    n = max(1, int(n))
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"]:
        os.environ[var] = str(n)
    try:
        import torch  # type: ignore

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def set_low_priority_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


# ---------------------------
# OCR backend
# ---------------------------
class OCRBackend:
    def __init__(self, preferred_backend: str, max_dim: int, max_pixels: int, cpu_threads: int) -> None:
        self.preferred_backend = preferred_backend
        self.max_dim = max_dim
        self.max_pixels = max_pixels
        self.cpu_threads = cpu_threads
        self._easy_reader = None

    def has_easyocr(self) -> bool:
        if Image is None or np is None:
            return False
        try:
            import easyocr  # type: ignore  # noqa
            return True
        except Exception:
            return False

    def has_tesseract(self) -> bool:
        if Image is None:
            return False
        try:
            import pytesseract  # type: ignore  # noqa
            return shutil.which("tesseract") is not None
        except Exception:
            return False

    def _resize_cap(self, img):
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        scale_dim = min(1.0, self.max_dim / max(w, h)) if self.max_dim else 1.0
        scale_pix = min(1.0, (self.max_pixels / (w * h)) ** 0.5) if self.max_pixels else 1.0
        scale = min(scale_dim, scale_pix)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return img

    def _preprocess(self, img):
        img = ImageOps.exif_transpose(img)  # type: ignore
        img = self._resize_cap(img)
        img = img.convert("L")
        img = ImageOps.autocontrast(img)  # type: ignore
        img = ImageEnhance.Contrast(img).enhance(1.8)  # type: ignore
        img = ImageEnhance.Sharpness(img).enhance(1.4)  # type: ignore
        return img

    def _get_easy_reader(self):
        if self._easy_reader is None:
            apply_thread_limits(self.cpu_threads)
            import easyocr  # type: ignore

            self._easy_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)  # type: ignore
        return self._easy_reader

    def _iter_backends(self) -> list[str]:
        available: list[str] = []
        if self.has_easyocr():
            available.append("easyocr")
        if self.has_tesseract():
            available.append("tesseract")

        if self.preferred_backend != "auto" and self.preferred_backend in available:
            return [self.preferred_backend] + [b for b in available if b != self.preferred_backend]
        return available

    def ocr_easy(self, img) -> str:
        reader = self._get_easy_reader()
        arr = np.array(img.convert("RGB"))  # type: ignore
        lines = reader.readtext(arr, detail=0, paragraph=False)
        return "\n".join(lines) if lines else ""

    def ocr_tess(self, img) -> str:
        apply_thread_limits(self.cpu_threads)
        import pytesseract  # type: ignore

        cfg = "--oem 3 --psm 6"
        try:
            return pytesseract.image_to_string(img, lang="spa", config=cfg)
        except Exception:
            return pytesseract.image_to_string(img, lang="eng", config=cfg)

    def ocr(self, img) -> tuple[str, str]:
        if Image is None:
            raise RuntimeError("Falta pillow (PIL).")

        backends = self._iter_backends()
        if not backends:
            raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) o pytesseract + tesseract.")

        img = self._preprocess(img)
        errors: list[str] = []

        for b in backends:
            try:
                txt = self.ocr_easy(img) if b == "easyocr" else self.ocr_tess(img)
                txt = normalize_text_block(txt)
                if txt:
                    return txt, f"image_ocr_{b}"
            except Exception as exc:
                errors.append(f"{b}: {exc}")

        if errors:
            raise RuntimeError("OCR falló; " + "; ".join(errors))
        raise RuntimeError("OCR falló: no se obtuvo texto útil.")


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


def parse_invoice(text_header: str, text_totals: str, text_full: str, path: Path, fuente: str) -> InvoiceResult:
    full = text_full or ""
    header = text_header or full
    totals = text_totals or full
    combined = header + "\n" + totals + "\n" + full

    serie, folio = extract_serie_folio(header)
    if serie is None and folio is None:
        serie, folio = extract_serie_folio(full)

    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_emisor = extract_rut_emisor(header) or extract_rut_emisor(full)
    razon_social = extract_razon_social_emisor(full, header)

    fecha = pick_best_date(header) or pick_best_date(full)

    total_num = extract_total(totals) or extract_total(full)
    total_str = format_money_uy(total_num)

    vat_rates = detect_vat_rates(combined)
    iva_total = extract_iva_total(totals) or extract_iva_total(full)

    sin_iva_num, sin_iva_fuente = compute_importe_sin_iva(total_num, iva_total, vat_rates, combined)
    sin_iva_str = format_money_uy(sin_iva_num)

    return InvoiceResult(
        fecha=fecha,
        serie=serie,
        folio=folio,
        serie_y_folio=serie_y_folio,
        razon_social=razon_social,
        rut_emisor=rut_emisor,
        es_nota_de_credito=is_credit_note(combined, path),
        importe_total_con_iva=total_str,
        importe_total_con_iva_num=total_num,
        importe_sin_iva=sin_iva_str,
        importe_sin_iva_num=sin_iva_num,
        importe_sin_iva_fuente=sin_iva_fuente,
        _archivo=str(path),
        _fuente=fuente,
    )


def process_image_fast(path: Path, ocr: OCRBackend, debug: bool) -> tuple[str, str, str, str]:
    if Image is None:
        raise RuntimeError("Falta pillow (PIL).")

    img = Image.open(path)

    # Header: un poco más de la mitad superior
    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.58)

    # Totales: parte inferior completa (para no perder "TOTAL" por estar más a la izquierda)
    totals_img = crop_rel(img, 0.00, 0.52, 1.00, 0.99)

    text_header, fuente = ocr.ocr(header_img)
    text_totals, _ = ocr.ocr(totals_img)

    text_full = ""

    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print("\n---[OCR HEADER]---")
        print(text_header)
        print("\n---[OCR TOTALS]---")
        print(text_totals)
        print("=== FIN ===\n")

    return text_full, text_header, text_totals, fuente


def process_image_full(path: Path, ocr: OCRBackend, debug: bool) -> tuple[str, str]:
    if Image is None:
        raise RuntimeError("Falta pillow (PIL).")
    img = Image.open(path)
    txt, fuente = ocr.ocr(img)
    if debug:
        print(f"\n---[OCR FULL: {path.name}]---")
        print(txt)
    return txt, fuente


def process_pdf(path: Path, debug: bool) -> tuple[str, str, str, str]:
    text_full, fuente = extract_text_from_pdf(path)
    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print(text_full)
        print("=== FIN ===\n")
    return text_full, text_full, text_full, fuente


def iter_files(root: Path) -> Iterable[Path]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if root.is_file():
        if root.suffix.lower() in exts:
            yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _needs_full_pass(res: InvoiceResult) -> bool:
    if res.importe_total_con_iva_num is None:
        return True
    if (res.serie is None and res.folio is None):
        return True
    if res.razon_social is None or _looks_like_label_only(res.razon_social):
        return True
    # totals absurdos muy chicos
    if res.importe_total_con_iva_num is not None and res.importe_total_con_iva_num < 50:
        return True
    return False


def parse_path(root: Path, *, as_json: bool, debug: bool, backend: str, ocr_mode: str, max_dim: int, max_pixels: int, cpu_threads: int) -> list[dict[str, Any]]:
    if not root.exists():
        print(f"[WARN] No existe: {root}")
        return []

    apply_thread_limits(cpu_threads)
    ocr = OCRBackend(preferred_backend=backend, max_dim=max_dim, max_pixels=max_pixels, cpu_threads=cpu_threads)

    out_results: list[InvoiceResult] = []

    for path in sorted(iter_files(root)):
        try:
            if path.suffix.lower() == ".pdf":
                tf, th, tt, fuente = process_pdf(path, debug=debug)
                res = parse_invoice(text_header=th, text_totals=tt, text_full=tf, path=path, fuente=fuente)
                out_results.append(res)
                continue

            # imágenes
            tf, th, tt, fuente = process_image_fast(path, ocr=ocr, debug=debug)
            res = parse_invoice(text_header=th, text_totals=tt, text_full=tf, path=path, fuente=fuente)

            if ocr_mode == "full" or (ocr_mode == "fast" and _needs_full_pass(res)):
                full_txt, full_fuente = process_image_full(path, ocr=ocr, debug=debug and False)
                # reparse, ahora con full
                res = parse_invoice(text_header=th, text_totals=tt, text_full=full_txt, path=path, fuente=full_fuente)

            out_results.append(res)

        except Exception as e:
            if debug:
                print(f"[ERROR] {path.name}: {e}")
            out_results.append(
                InvoiceResult(
                    fecha=None,
                    serie=None,
                    folio=None,
                    serie_y_folio=None,
                    razon_social=None,
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

    out = [asdict(r) for r in out_results]
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for r in out:
            print(f"{Path(r['_archivo']).name} | {r.get('fecha')} | {r.get('serie_y_folio')} | {r.get('razon_social')} | Total: {r.get('importe_total_con_iva')}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Archivo o carpeta con facturas")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--backend", choices=["auto", "easyocr", "tesseract"], default="auto")
    ap.add_argument("--ocr-mode", choices=["fast", "full"], default="fast", help="fast=header+totals y full solo si hace falta")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Limita hilos (recomendado 1 o 2)")
    ap.add_argument("--max-dim", type=int, default=1600, help="Máximo lado de imagen (baja consumo)")
    ap.add_argument("--max-pixels", type=int, default=2_000_000, help="Máximo total píxeles (baja consumo)")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (Windows)")
    args = ap.parse_args(argv)

    if args.low_priority:
        set_low_priority_windows()

    parse_path(
        Path(args.path),
        as_json=args.json,
        debug=args.debug,
        backend=args.backend,
        ocr_mode=args.ocr_mode,
        max_dim=args.max_dim,
        max_pixels=args.max_pixels,
        cpu_threads=args.cpu_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
