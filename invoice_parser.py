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

        window = text[max(0, idx - 40) : min(len(text), idx + 40)]
        wup = _safe_upper(window)

        # bonus por FECHA (documento)
        if re.search(r"FECHA\s+DE\s+DOCUMENTO", wup):
            score += 80
        elif re.search(r"\bFECHA\b", wup):
            score += 25

        # penalizar vencimientos/CAE
        if re.search(r"\bVENC(?:IMIENTO)?\b|\bVTO\b", wup):
            score -= 70
        if re.search(r"\bCAE\b|\bRANGO\s+DE\s+CAE\b|\bFECHA\s+EMISOR\b", wup):
            score -= 80

        # OCR típico: "... UYU 13/10/2025"
        if re.search(r"\bUYU\b|\bPESO\b", wup):
            score += 10

        # preferir arriba del documento
        score += max(0, 2500 - idx) / 2500.0

        if score > best_score:
            best_score = score
            best = dstr

    return best


def extract_rut_emisor(text: str) -> Optional[str]:
    if not text:
        return None
    up = _safe_upper(text)

    # priorizar "RUT EMISOR"
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

    # SERIE A 049009
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,20}([A-Z])\b.*?\bNUMER[O0]\b[^0-9]{0,30}([0-9][0-9\s]{2,})", text, flags=re.I | re.S)
    if m:
        serie = m.group(1).upper()
        folio = _only_digits(m.group(2)).lstrip("0") or "0"
        if 2 <= len(folio) <= 9:
            return serie, folio

    # fallback simple "A 3519972"
    m = re.search(r"\b([A-Z])\s*0*([0-9][0-9\s]{2,})\b", text, flags=re.I)
    if m:
        serie = m.group(1).upper()
        folio = _only_digits(m.group(2)).lstrip("0") or "0"
        if 2 <= len(folio) <= 9:
            return serie, folio

    return None, None


def extract_razon_social_emisor(text: str) -> Optional[str]:
    """
    Heurística razonable para UY:
    - "<EMISOR> RUT RECEPTOR ..."
    - "RAZON SOCIAL: <...>" / "DENOMINACION: <...>" (cuando aparece)
    """
    if not text:
        return None

    # 1) "<EMISOR> RUT RECEPTOR"
    m = re.search(r"\b(.{3,80}?)\s+RUT\s+RECEPTOR\b", text, flags=re.I)
    if m:
        cand = _collapse_spaces(m.group(1))
        # filtrar basura típica de etiquetas
        if not re.search(r"\bSERIE\b|\bNUMERO\b|\bTIPO\b|\bDOCUMENTO\b", _safe_upper(cand)):
            return cand.strip(" -:")

    # 2) "RAZON SOCIAL <...>" (a veces del receptor, ojo)
    lines = normalize_text_block(text).splitlines()
    for i, ln in enumerate(lines):
        up = _safe_upper(ln)
        if re.search(r"\b(RAZON\s+SOCIAL|DENOMINACION|NOMBRE)\b", up):
            # mismo renglón con valor
            m2 = re.search(r"(RAZON\s+SOCIAL|DENOMINACION|NOMBRE)\s*[:\-]?\s*(.+)", ln, flags=re.I)
            if m2 and m2.group(2).strip():
                cand = _collapse_spaces(m2.group(2))
                if len(cand) >= 3:
                    return cand.strip(" -:")
            # o en la línea siguiente
            if i + 1 < len(lines):
                cand = _collapse_spaces(lines[i + 1])
                if len(cand) >= 3 and not re.search(r"\bRUT\b|\bRUC\b|\bDIRECCION\b|\bDOMICILIO\b", _safe_upper(cand)):
                    return cand.strip(" -:")

    return None


def is_credit_note(text: str, path: Path) -> bool:
    x = _safe_upper(text) + " " + _safe_upper(path.name)
    return ("NOTA DE CREDITO" in x) or ("NOTA DE CRÉDITO" in x)


def detect_vat_rates(text: str) -> set[float]:
    up = _safe_upper(text or "")
    rates: set[float] = set()
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
    up = _safe_upper(text)

    m = re.search(r"\bTOTAL\s*IVA\b[^0-9\-]{0,25}(" + MONEY_RE.pattern + r")", up)
    if m:
        return parse_money_token(m.group(1))

    # Caso e-factura DGI: "Total iva (22%) 231,18"
    m22 = re.search(r"\bTOTAL\s+IVA\b[^0-9]{0,80}22\D{0,10}(" + MONEY_RE.pattern + r")", up)
    m10 = re.search(r"\bTOTAL\s+IVA\b[^0-9]{0,80}10\D{0,10}(" + MONEY_RE.pattern + r")", up)
    v22 = parse_money_token(m22.group(1)) if m22 else 0.0
    v10 = parse_money_token(m10.group(1)) if m10 else 0.0
    if m22 or m10:
        return round((v22 or 0.0) + (v10 or 0.0), 2)

    return None


def extract_total(text: str) -> Optional[float]:
    if not text:
        return None

    candidates: list[tuple[float, int]] = []

    def collect(pattern: str, score: int) -> None:
        for m in re.finditer(pattern, text, flags=re.I | re.S):
            v = parse_money_token(m.group(1))
            if v is not None:
                candidates.append((v, score))

    collect(r"\bTOTAL\s*A\s*PAGAR\b[^0-9\-]{0,40}(" + MONEY_RE.pattern + r")", 120)
    collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,40}(" + MONEY_RE.pattern + r")", 90)
    collect(r"\bTOTAL\b(?!\s*IVA)(?!\s*A\s*PAGAR)[^0-9\-]{0,40}(" + MONEY_RE_LOOSE.pattern + r")", 70)

    if candidates:
        candidates.sort(key=lambda t: (t[1], t[0]), reverse=True)
        best = candidates[0][0]
        # anti locura
        if best > 5_000_000 and len(candidates) > 1 and candidates[1][0] < 500_000:
            return candidates[1][0]
        return best

    # último recurso: mayor monto con formato UY
    vals = [parse_money_token(m.group(1)) for m in MONEY_RE.finditer(text)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def compute_importe_sin_iva(total: Optional[float], iva_total: Optional[float], vat_rates: set[float], text_full: str) -> tuple[Optional[float], Optional[str]]:
    if total is None:
        return None, None

    if iva_total is not None and 0 <= iva_total < total:
        net = round(total - iva_total, 2)
        rate = iva_total / max(net, 0.01)
        if vat_rates:
            if any(abs(rate - r) <= 0.03 for r in vat_rates):
                return net, "total_menos_iva"
        else:
            if 0.07 <= rate <= 0.25:
                return net, "total_menos_iva"

    up = _safe_upper(text_full)
    has_22 = (0.22 in vat_rates) or (re.search(r"\bIVA\s*22|\b22\s*%", up) is not None)
    has_10 = (0.10 in vat_rates) or (re.search(r"\bIVA\s*10|\b10\s*%", up) is not None)

    if has_22 and has_10:
        return None, None
    if has_22:
        return round(total / 1.22, 2), "total_div_22"
    if has_10:
        return round(total / 1.10, 2), "total_div_10"
    return None, None


# ---------------------------
# Limitación de recursos
# ---------------------------
def apply_thread_limits(n: int) -> None:
    n = max(1, int(n))
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"]:
        os.environ[var] = str(n)
    # si torch ya está importado, limitarlo también
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
            # IMPORTANTE: limitar threads antes de que torch se ponga creativo
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
        # SOLO UNA pasada (menos CPU). paragraph=False suele ser más estable con facturas.
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

        img = self._preprocess(img)
        errors: list[str] = []
        for b in self._iter_backends():
            try:
                txt = self.ocr_easy(img) if b == "easyocr" else self.ocr_tess(img)
                txt = normalize_text_block(txt)
                if txt:
                    return txt, f"image_ocr_{b}"
            except Exception as exc:
                errors.append(f"{b}: {exc}")

        if errors:
            raise RuntimeError("OCR falló; " + "; ".join(errors))
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


def parse_invoice(text_header: str, text_totals: str, text_full: str, path: Path, fuente: str) -> InvoiceResult:
    full = text_full or ""
    header = text_header or full
    totals = text_totals or full
    combined = header + "\n" + totals + "\n" + full

    serie, folio = extract_serie_folio(header)
    serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_emisor = extract_rut_emisor(header) or extract_rut_emisor(full)
    razon_social = extract_razon_social_emisor(full) or extract_razon_social_emisor(header)

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


def process_image(path: Path, ocr: OCRBackend, debug: bool, debug_full: bool, ocr_mode: str) -> tuple[str, str, str, str]:
    if Image is None:
        raise RuntimeError("Falta pillow (PIL).")

    img = Image.open(path)

    # 2 regiones: header y totales (barato)
    header_img = crop_rel(img, 0.00, 0.00, 1.00, 0.50)
    totals_img = crop_rel(img, 0.35, 0.55, 1.00, 0.98)

    text_header, fuente = ocr.ocr(header_img)
    text_totals, _ = ocr.ocr(totals_img)

    text_full = ""
    if ocr_mode == "full" or (not text_header and not text_totals):
        # recién acá hacemos OCR full (caro)
        text_full, _ = ocr.ocr(img)

    if debug:
        print(f"\n=== {path.name} ({fuente}) ===")
        print("\n---[OCR HEADER]---")
        print(text_header)
        print("\n---[OCR TOTALS]---")
        print(text_totals)
        if debug_full:
            print("\n---[OCR FULL]---")
            print(text_full)
        print("=== FIN ===\n")

    return text_full, text_header, text_totals, fuente


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


def parse_path(root: Path, *, as_json: bool, debug: bool, debug_full: bool, backend: str, ocr_mode: str, max_dim: int, max_pixels: int, cpu_threads: int) -> list[dict[str, Any]]:
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
            else:
                tf, th, tt, fuente = process_image(path, ocr=ocr, debug=debug, debug_full=debug_full, ocr_mode=ocr_mode)

            res = parse_invoice(text_header=th, text_totals=tt, text_full=tf, path=path, fuente=fuente)
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
    ap.add_argument("--debug-full", action="store_true", help="Imprimir OCR full (pesado)")
    ap.add_argument("--backend", choices=["auto", "easyocr", "tesseract"], default="auto")
    ap.add_argument("--ocr-mode", choices=["fast", "full"], default="fast", help="fast=solo header+totals (recomendado)")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Limita hilos (recomendado 1 o 2)")
    ap.add_argument("--max-dim", type=int, default=1800, help="Máximo lado de imagen (baja consumo)")
    ap.add_argument("--max-pixels", type=int, default=3_000_000, help="Máximo total píxeles (baja consumo)")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (Windows)")
    args = ap.parse_args(argv)

    if args.low_priority:
        set_low_priority_windows()

    parse_path(
        Path(args.path),
        as_json=args.json,
        debug=args.debug,
        debug_full=args.debug_full,
        backend=args.backend,
        ocr_mode=args.ocr_mode,
        max_dim=args.max_dim,
        max_pixels=args.max_pixels,
        cpu_threads=args.cpu_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
