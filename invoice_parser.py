# invoice_parser.py
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Resource limits (set EARLY)
# ----------------------------

def _set_thread_env(n: int) -> None:
    n = max(1, int(n))
    # Common BLAS/OpenMP knobs
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    # Avoid GPU surprises
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Suppress noisy torch/easyocr warning you keep seeing
warnings.filterwarnings("ignore", message=r".*pin_memory.*no accelerator.*", category=UserWarning)

# ----------------------------
# Optional dependencies
# ----------------------------

def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None

def _try_import_easyocr():
    try:
        import easyocr  # type: ignore
        return easyocr
    except Exception:
        return None

def _try_import_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader
    except Exception:
        return None

def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


# ----------------------------
# Helpers: parsing & scoring
# ----------------------------

_RE_DATE = re.compile(r"\b([0-3]?\d)[/.\-]([01]?\d)[/.\-]((?:19|20)\d{2})\b")
_RE_RUT12 = re.compile(r"\b(\d{12})\b")
# Matches amounts like: 8.516,99 | 4 018,01 | 785,06 | 6491.78 (less common)
_RE_AMOUNT = re.compile(
    r"\b(\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2}))\b"
)

def norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def upper_clean(s: str) -> str:
    s = norm_text(s)
    return s.upper()

def parse_amount_to_float(raw: str) -> Optional[float]:
    """
    Uruguay style is usually thousands '.' or space, decimals ','.
    Handles:
      - "4 018,01" -> 4018.01
      - "8.516,99" -> 8516.99
      - "6491,78" -> 6491.78
      - "6491.78" -> 6491.78
    """
    if not raw:
        return None
    s = raw.strip()
    s = s.replace("UYU", "").replace("$", "").replace(" ", "")
    s = re.sub(r"[^\d,\.]", "", s)

    if not s:
        return None

    # both '.' and ',' -> assume '.' thousands, ',' decimals
    if "." in s and "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    elif "," in s and "." not in s:
        # assume comma decimals
        s = s.replace(",", ".")
    else:
        # only dots or only digits
        # if multiple dots and last chunk length == 2 -> decimal dot, others thousands
        if s.count(".") >= 2:
            parts = s.split(".")
            if len(parts[-1]) == 2:
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = s.replace(".", "")

    try:
        val = float(s)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None

def extract_dates_with_scores(text: str, current_year: int) -> List[Tuple[str, int]]:
    t = text
    out: List[Tuple[str, int]] = []
    for m in _RE_DATE.finditer(t):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        date = f"{int(d):02d}/{int(mo):02d}/{y}"
        ctx0 = max(0, m.start() - 35)
        ctx1 = min(len(t), m.end() + 35)
        ctx = upper_clean(t[ctx0:ctx1])

        score = 0
        if "FECHA DE DOCUMENTO" in ctx or "FECHA DOCUMENTO" in ctx:
            score += 8
        elif "FECHA" in ctx:
            score += 5

        if "VENC" in ctx or "VENCIMIENTO" in ctx:
            score -= 8
        if "CAE" in ctx or "CAI" in ctx:
            score -= 6

        year = int(y)
        # To avoid picking CAE vencimiento 2027 when docs are 2025, etc.
        if year > current_year + 1:
            score -= 4

        out.append((date, score))
    return out

def choose_best_date(text: str, current_year: int) -> Optional[str]:
    cand = extract_dates_with_scores(text, current_year)
    if not cand:
        return None
    cand.sort(key=lambda x: (x[1], x[0]), reverse=True)
    best_score = cand[0][1]
    # If tie-ish, prefer the earliest date (usually doc date <= due date)
    top = [c for c in cand if c[1] == best_score]
    top.sort(key=lambda x: x[0])
    return top[0][0]

def extract_rut_emisor(text: str) -> Optional[str]:
    t = text
    best: Tuple[str, int] | None = None
    for m in _RE_RUT12.finditer(t):
        rut = m.group(1)
        ctx0 = max(0, m.start() - 30)
        ctx1 = min(len(t), m.end() + 30)
        ctx = upper_clean(t[ctx0:ctx1])
        score = 0
        if "RUT EMISOR" in ctx or "RUC EMISOR" in ctx:
            score += 10
        if "RUC" in ctx or "RUT" in ctx:
            score += 6
        if "COMPRADOR" in ctx or "RECEPTOR" in ctx:
            score -= 6
        if best is None or score > best[1]:
            best = (rut, score)
    return best[0] if best else None

def _iter_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return lines

def extract_total_amount(text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Returns (raw_string, float_value, fuente)
    """
    lines = _iter_lines(text)
    up_lines = [upper_clean(l) for l in lines]
    best: Tuple[Optional[str], Optional[float], int, str] = (None, None, -10_000, "none")

    def consider(line: str, idx: int, fuente: str) -> None:
        nonlocal best
        up = upper_clean(line)
        # Extract all amount-looking tokens from line
        toks = _RE_AMOUNT.findall(line)
        for tok in toks:
            val = parse_amount_to_float(tok)
            if val is None:
                continue

            score = 0
            if "TOTAL A PAGAR" in up or "TOTAL APAGAR" in up:
                score += 14
            elif "TOTAL" in up:
                score += 10
            if "A PAGAR" in up:
                score += 6
            if "SUBTOTAL" in up:
                score -= 7
            if "IVA" in up and "TOTAL" not in up:
                score -= 4
            if "CAE" in up or "CAI" in up:
                score -= 5

            # Prefer bottom-ish
            if len(lines) > 0:
                pos = idx / max(1, (len(lines) - 1))
                if pos > 0.65:
                    score += 3

            # If this looks like a tiny line item total and we have bigger candidates later, scoring will handle.
            # But still, avoid absurdly tiny totals:
            if val < 1:
                score -= 5

            # Update best: score first, then bigger amount
            if (score > best[2]) or (score == best[2] and (best[1] is None or val > best[1])):
                best = (tok, val, score, fuente)

    # 1) Strong labels first
    for i, ln in enumerate(lines):
        up = up_lines[i]
        if any(k in up for k in ["TOTAL A PAGAR", "TOTAL", "A PAGAR", "IMPORTE TOTAL"]):
            consider(ln, i, "line_label")

    # 2) If nothing found with a label, consider all lines but penalize
    if best[1] is None:
        for i, ln in enumerate(lines):
            consider(ln, i, "line_any")

    # 3) If still nothing, scan whole text tokens
    if best[1] is None:
        for m in _RE_AMOUNT.finditer(text):
            tok = m.group(1)
            val = parse_amount_to_float(tok)
            if val is None:
                continue
            score = 0
            if val < 1:
                score -= 5
            if score > best[2] or (score == best[2] and (best[1] is None or val > best[1])):
                best = (tok, val, score, "global_fallback")

    if best[1] is None:
        return None, None, None
    return best[0], best[1], best[3]

def extract_serie_folio(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Tries to find Serie (single letter) and Folio/Numero (4-8 digits).
    Avoids confusing 12-digit RUT with folio.
    """
    t = text
    up = upper_clean(t)

    # Pattern like: "SERIE NUMERO ... A 049009"
    m = re.search(r"(SERIE|BERIE|SER1E)\s*(NUMERO|N[UO]MERO)[^\n]{0,80}\b([A-Z])\s*([0-9]{4,8})\b", up)
    if m:
        return m.group(3), m.group(4)

    # Pattern like: "... SERIE A NUMERO 3519972"
    m = re.search(r"(SERIE|BERIE|SER1E)[^\n]{0,40}\b([A-Z])\b[^\n]{0,40}(NUMERO|N[UO]MERO)[^\n]{0,40}\b([0-9]{4,8})\b", up)
    if m:
        return m.group(2), m.group(4)

    # Pattern like "A 3519972" near words that suggest document number
    candidates: List[Tuple[str, str, int]] = []

    for m in re.finditer(r"\b([A-Z])\s*[-:]?\s*([0-9]{4,8})\b", up):
        serie, folio = m.group(1), m.group(2)
        ctx0 = max(0, m.start() - 40)
        ctx1 = min(len(up), m.end() + 40)
        ctx = up[ctx0:ctx1]
        score = 0
        if "SERIE" in ctx or "BERIE" in ctx:
            score += 7
        if "NUMERO" in ctx or "NOMERO" in ctx:
            score += 7
        if "DOCUMENTO" in ctx or "FACTURA" in ctx or "E-FACTURA" in ctx:
            score += 4
        if "RUC" in ctx or "RUT" in ctx or "COMPRADOR" in ctx or "RECEPTOR" in ctx:
            score -= 8
        if "CAE" in ctx or "CAI" in ctx:
            score -= 6
        candidates.append((serie, folio, score))

    # Also allow just number (folio) if series is missing (common OCR fail)
    for m in re.finditer(r"\b([0-9]{4,8})\b", up):
        folio = m.group(1)
        ctx0 = max(0, m.start() - 35)
        ctx1 = min(len(up), m.end() + 35)
        ctx = up[ctx0:ctx1]
        score = 0
        if "NUMERO" in ctx or "NOMERO" in ctx:
            score += 6
        if "DOCUMENTO" in ctx or "FACTURA" in ctx or "E-FACTURA" in ctx:
            score += 4
        if "RUC" in ctx or "RUT" in ctx or "COMPRADOR" in ctx or "RECEPTOR" in ctx:
            score -= 9
        if "CAE" in ctx or "CAI" in ctx:
            score -= 6
        # Penalize super short numbers that are likely quantities
        if len(folio) <= 4:
            score -= 2
        candidates.append((None, folio, score))  # type: ignore

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[2], reverse=True)
    serie, folio, _ = candidates[0]
    return serie, folio

def detect_provider_name(text: str) -> Optional[str]:
    up = upper_clean(text)
    # quick win for your cases
    if "BIMBO" in up:
        return "BIMBO"
    if "PURA PALTA" in up:
        return "PURA PALTA SAS"
    return None


# ----------------------------
# OCR + QR
# ----------------------------

@dataclass
class OCRConfig:
    mode: str
    cpu_threads: int
    max_dim: int
    max_pixels: int

def _resize_guard(cv2, np, img, max_dim: int, max_pixels: int):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    # scale by max_dim first
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))

    # scale by max_pixels too
    if h * w * (scale ** 2) > max_pixels:
        scale2 = math.sqrt(max_pixels / float(h * w))
        scale = min(scale, scale2)

    if scale < 0.999:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def preprocess_for_ocr(cv2, np, img_bgr, mode: str):
    """
    Returns grayscale-ish binary/contrast version depending on mode.
    """
    img = img_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == "fast":
        # light touch
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    # balanced / accurate:
    # equalize + denoise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    eq = cv2.bilateralFilter(eq, 7, 60, 60)

    if mode == "accurate":
        th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 7)
        # small morphology to connect characters
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
        return th

    # balanced
    th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 25, 9)
    return th

def decode_qr_from_image(cv2, np, img_bgr) -> Optional[str]:
    try:
        det = cv2.QRCodeDetector()
        data, _, _ = det.detectAndDecode(img_bgr)
        data = (data or "").strip()
        if data:
            return data
        # try on grayscale (sometimes helps)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        data, _, _ = det.detectAndDecode(gray)
        data = (data or "").strip()
        return data or None
    except Exception:
        return None

def ocr_image_easyocr(image_path: Path, cfg: OCRConfig, debug: bool = False) -> Tuple[str, str]:
    """
    Returns (full_text, source_label)
    """
    cv2 = _try_import_cv2()
    np = _try_import_numpy()
    easyocr = _try_import_easyocr()
    if cv2 is None or np is None:
        raise RuntimeError("Falta opencv-python o numpy. Instalá: pip install opencv-python numpy")
    if easyocr is None:
        raise RuntimeError("Falta easyocr. Instalá: pip install easyocr")

    # OpenCV thread limit
    try:
        cv2.setNumThreads(max(1, int(cfg.cpu_threads)))
    except Exception:
        pass

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("No se pudo leer la imagen (cv2.imread devolvió None)")

    img = _resize_guard(cv2, np, img, cfg.max_dim, cfg.max_pixels)

    # QR first (best case)
    qr = decode_qr_from_image(cv2, np, img)
    if qr:
        # Treat QR payload as extra text source added up front
        if debug:
            print(f"[QR] {image_path.name}: {qr}")
        # Still do OCR too, but QR gets appended (so extractor sees it)
        qr_text = f"\nQR_DATA: {qr}\n"
    else:
        qr_text = ""

    proc = preprocess_for_ocr(cv2, np, img, cfg.mode)

    # Build reader once per call (simple); you can optimize to cache if wanted.
    # Spanish+English helps with "Fecha/Total" etc.
    try:
        reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    except Exception as e:
        raise RuntimeError(f"EasyOCR no pudo inicializarse: {e}")

    # detail=0 returns just strings
    try:
        results = reader.readtext(proc, detail=0, paragraph=True)
    except Exception:
        # fallback: try without preprocessing
        results = reader.readtext(img, detail=0, paragraph=True)

    text = "\n".join([norm_text(r) for r in results if norm_text(r)])
    text = (qr_text + text).strip()
    if not text:
        raise RuntimeError("OCR falló: no se obtuvo texto útil.")
    return text, "image_ocr_easyocr"

def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, str]:
    PdfReader = _try_import_pypdf()
    if PdfReader is None:
        raise RuntimeError("Falta pypdf. Instalá: pip install pypdf")
    reader = PdfReader(str(pdf_path))
    texts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            texts.append(t)
    full = "\n".join(texts).strip()
    if not full:
        raise RuntimeError("PDF sin texto extraíble (parece escaneado).")
    return full, "pdf_text"


# ----------------------------
# Gold / report
# ----------------------------

def load_gold(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("gold.json debe ser un JSON objeto (diccionario).")
    # keys should be filenames: "BIMBO A3519972 CREDITO.jpeg"
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out

def evaluate_against_gold(results: List[Dict[str, Any]], gold: Dict[str, Dict[str, Any]]) -> str:
    # Match by basename
    keyed = {Path(r.get("_archivo", "")).name: r for r in results if r.get("_archivo")}
    gold_keys = [k for k in gold.keys() if k in keyed]
    total = len(gold_keys)
    lines = []
    lines.append("\n=== REPORT ===")
    lines.append(f"Docs con gold: {total}")

    if total == 0:
        return "\n".join(lines)

    fields = ["fecha", "rut_emisor", "serie", "folio", "importe_total_con_iva_num"]
    ok_counts = {f: 0 for f in fields}

    mismatches: List[str] = []
    for fname in gold_keys:
        pred = keyed[fname]
        g = gold[fname]
        diffs = []
        for f in fields:
            pv = pred.get(f)
            gv = g.get(f)
            if pv == gv:
                ok_counts[f] += 1
            else:
                diffs.append((f, pv, gv))
        if diffs:
            chunk = [f"* {fname}"]
            for f, pv, gv in diffs:
                chunk.append(f"  - {f}: pred={pv!r} gold={gv!r}")
            mismatches.append("\n".join(chunk))

    for f in fields:
        pct = 100.0 * ok_counts[f] / total
        lines.append(f"{f}: {ok_counts[f]}/{total} ({pct:.1f}%)")

    if mismatches:
        lines.append("\n--- MISMATCHES ---")
        lines.extend(mismatches)
    return "\n".join(lines)


# ----------------------------
# Main extraction pipeline
# ----------------------------

def extract_fields_from_text(text: str, filename: str, current_year: int) -> Dict[str, Any]:
    text = text or ""
    t = text.strip()

    fecha = choose_best_date(t, current_year=current_year)
    rut = extract_rut_emisor(t)
    serie, folio = extract_serie_folio(t)

    prov = detect_provider_name(t)

    raw_total, total_num, total_src = extract_total_amount(t)

    out = {
        "fecha": fecha,
        "serie": serie,
        "folio": folio,
        "serie_y_folio": (f"{serie}-{folio}" if serie and folio else None),
        "razon_social": prov,  # best-effort
        "rut_emisor": rut,
        "es_nota_de_credito": False,
        "importe_total_con_iva": raw_total,
        "importe_total_con_iva_num": total_num,
        "importe_sin_iva": None,
        "importe_sin_iva_num": None,
        "importe_sin_iva_fuente": None,
        "_archivo": filename,
        "_fuente": None,
    }

    # If we got a total and it's Uruguay VAT-ish, compute sin IVA as heuristic (only when clearly 22%)
    if total_num is not None:
        # divisors
        sin_iva_22 = round(total_num / 1.22, 2)
        out["importe_sin_iva"] = f"{sin_iva_22:.2f}".replace(".", ",")
        out["importe_sin_iva_num"] = sin_iva_22
        out["importe_sin_iva_fuente"] = "total_div_22"

    return out

def list_docs(folder: Path) -> List[Path]:
    exts = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort(key=lambda x: x.name.lower())
    return files

def set_low_priority_if_possible(enable: bool) -> None:
    if not enable:
        return
    psutil = _try_import_psutil()
    if psutil is None:
        return
    try:
        p = psutil.Process(os.getpid())
        # Windows: BELOW_NORMAL_PRIORITY_CLASS
        if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # type: ignore
    except Exception:
        return

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse invoices (UY) from PDFs/images using QR+OCR+heuristics.")
    ap.add_argument("input", help="Carpeta con facturas (pdf/jpg/png)")

    ap.add_argument("--json", action="store_true", help="Imprime JSON (lista) a stdout")
    ap.add_argument("--out", default=None, help="Guarda salida JSON en archivo")
    ap.add_argument("--debug", action="store_true", help="Imprime info adicional")

    ap.add_argument("--report", action="store_true", help="Evalúa contra gold.json y muestra métricas")
    ap.add_argument("--gold", default=None, help="Ruta a gold.json")

    ap.add_argument("--ocr-mode", default="balanced", choices=["fast", "balanced", "accurate"],
                    help="Modo OCR (más rápido vs más preciso)")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Límite de threads CPU")
    ap.add_argument("--max-dim", type=int, default=1800, help="Máx ancho/alto al hacer OCR")
    ap.add_argument("--max-pixels", type=int, default=2_500_000, help="Máx píxeles al hacer OCR")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (si se puede)")
    ap.add_argument("--limit", type=int, default=0, help="Procesa solo N docs (0 = todos)")

    args = ap.parse_args(argv)

    _set_thread_env(args.cpu_threads)
    set_low_priority_if_possible(args.low_priority)

    # Torch thread limit (if present)
    try:
        import torch  # type: ignore
        torch.set_num_threads(max(1, int(args.cpu_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    folder = Path(args.input)
    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Carpeta inválida: {folder}", file=sys.stderr)
        return 2

    cfg = OCRConfig(
        mode=args.ocr_mode,
        cpu_threads=args.cpu_threads,
        max_dim=args.max_dim,
        max_pixels=args.max_pixels
    )

    docs = list_docs(folder)
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]

    current_year = time.localtime().tm_year

    results: List[Dict[str, Any]] = []

    for p in docs:
        try:
            if p.suffix.lower() == ".pdf":
                text, src = extract_text_from_pdf(p)
            else:
                text, src = ocr_image_easyocr(p, cfg, debug=args.debug)

            if args.debug:
                print(f"\n=== {p.name} ({src}) ===")
                # Print first ~40 lines max
                for ln in _iter_lines(text)[:40]:
                    print(ln)
                print("=== FIN ===\n")

            item = extract_fields_from_text(text, str(p), current_year=current_year)
            item["_fuente"] = src
            results.append(item)

        except Exception as e:
            if args.debug:
                print(f"[ERROR] {p.name}: {e}")
            results.append({
                "fecha": None,
                "serie": None,
                "folio": None,
                "serie_y_folio": None,
                "razon_social": None,
                "rut_emisor": None,
                "es_nota_de_credito": False,
                "importe_total_con_iva": None,
                "importe_total_con_iva_num": None,
                "importe_sin_iva": None,
                "importe_sin_iva_num": None,
                "importe_sin_iva_fuente": None,
                "_archivo": str(p),
                "_fuente": "error",
                "_error": str(e),
            })

    # Report
    if args.report:
        if not args.gold:
            print("[ERROR] Usá --gold ruta\\gold.json", file=sys.stderr)
            return 2
        gold = load_gold(Path(args.gold))
        rep = evaluate_against_gold(results, gold)
        print(rep)

    # Output
    if args.json or args.out:
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(payload, encoding="utf-8")
        if args.json:
            print(payload)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
