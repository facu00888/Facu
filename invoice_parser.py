# invoice_parser.py
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Helpers: fechas y "números Uruguay"
# -----------------------------
DATE_RE = r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)"


def _norm(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def parse_date(s: str) -> Optional[str]:
    m = re.search(DATE_RE, s or "")
    if not m:
        return None
    d, mn, y = map(int, m.groups())
    if not (1 <= d <= 31 and 1 <= mn <= 12 and 2000 <= y <= 2100):
        return None
    return f"{d:02d}/{mn:02d}/{y:04d}"


def parse_decimal_uy(s: str) -> Optional[float]:
    """
    Parse numbers like:
    - 8.516,99  -> 8516.99
    - 549,68    -> 549.68
    - 6046,00   -> 6046.00
    """
    if s is None:
        return None
    s = (s or "").strip()
    m = re.search(r"[-–]?\s*\d{1,3}(?:\.\d{3})*(?:,\d{1,4})?|\d+(?:,\d{1,4})?", s)
    if not m:
        return None
    tok = m.group(0).replace("–", "-").replace(" ", "")
    neg = tok.startswith("-")
    tok = tok.lstrip("-")

    if "." in tok and "," in tok:
        tok = tok.replace(".", "").replace(",", ".")
    elif "," in tok and "." not in tok:
        tok = tok.replace(",", ".")

    try:
        val = float(tok)
        return -val if neg else val
    except ValueError:
        return None


def fmt_decimal_uy(v: float) -> str:
    # 12345.67 -> "12.345,67"
    s = f"{v:,.2f}"  # "12,345.67" in en_US style
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def split_lines(text: str) -> List[str]:
    lines = [_norm(ln) for ln in (text or "").splitlines()]
    return [ln for ln in lines if ln]


def find_line_index(lines: List[str], pattern: str, flags=re.I) -> Optional[int]:
    rgx = re.compile(pattern, flags)
    for i, ln in enumerate(lines):
        if rgx.search(ln):
            return i
    return None


def after_nonempty(lines: List[str], idx: int, k: int = 1) -> List[str]:
    out = []
    j = idx + 1
    while j < len(lines) and len(out) < k:
        if lines[j].strip():
            out.append(lines[j].strip())
        j += 1
    return out


def set_low_priority_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        IDLE_PRIORITY_CLASS = 0x40
        kernel32.SetPriorityClass(handle, IDLE_PRIORITY_CLASS)
    except Exception:
        pass


def limit_threads(cpu_threads: int) -> None:
    """
    Best-effort para no dejar la PC clavada al 100%.
    """
    if cpu_threads and cpu_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
        try:
            import torch  # type: ignore
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass


# -----------------------------
# PDF text extraction
# -----------------------------
def extract_pdf_text(path: str) -> str:
    # Prefer pdfplumber; fall back to pypdf.
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        pass

    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception as e:
        raise RuntimeError(f"No pude leer el PDF '{path}': {e}")


# -----------------------------
# OCR (opcional, para imágenes/scans)
# -----------------------------
def _resize_image_for_ocr(img, max_dim: int, max_pixels: int):
    w, h = img.size
    if max_pixels and (w * h) > max_pixels:
        scale = (max_pixels / float(w * h)) ** 0.5
        w2 = max(1, int(w * scale))
        h2 = max(1, int(h * scale))
        img = img.resize((w2, h2))
        w, h = img.size

    if max_dim and max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        w2 = max(1, int(w * scale))
        h2 = max(1, int(h * scale))
        img = img.resize((w2, h2))
    return img


def ocr_image_easyocr(path: str, mode: str, max_dim: int, max_pixels: int) -> str:
    try:
        import easyocr  # type: ignore
    except Exception as e:
        raise RuntimeError(f"easyocr no está instalado: {e}")

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pillow no está instalado: {e}")

    reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)

    img = Image.open(path)
    img = _resize_image_for_ocr(img, max_dim=max_dim, max_pixels=max_pixels)

    # Preprocesado suave si hay OpenCV
    if mode in {"balanced", "accurate"}:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            arr = np.array(img)
            if arr.ndim == 3:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = arr
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            thr = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6
            )
            img = Image.fromarray(thr)
        except Exception:
            pass

    try:
        import numpy as np  # type: ignore
        np_img = np.array(img)
    except Exception:
        np_img = img

    paragraph = True if mode != "fast" else False
    detail = 0
    text = reader.readtext(np_img, detail=detail, paragraph=paragraph)

    if isinstance(text, list):
        content = "\n".join([t for t in text if str(t).strip()])
    else:
        content = str(text or "")

    content = content.strip()
    if len(content) < 10:
        raise RuntimeError("OCR falló: no se obtuvo texto útil.")
    return content


# -----------------------------
# Parsing DGI-ish PDFs
# -----------------------------
def parse_rut_tipo_doc(lines: List[str]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    idx = find_line_index(lines, r"\bRUT\b.*\bEMISOR\b.*\bTIPO\b.*\bDOCUMENTO\b")
    if idx is None:
        idx = find_line_index(lines, r"\bRUT\b.*\bEMISOR\b")
    if idx is None:
        return None, None, debug

    nxt = after_nonempty(lines, idx, k=2)
    line = nxt[0] if nxt else ""
    debug["rut_doc_line"] = line

    rut = None
    m = re.search(r"\b(\d{12})\b", line)
    if m:
        rut = m.group(1)

    blob = " ".join([line] + nxt[1:])
    tipo = None
    for cand in [
        "e-Factura",
        "e-Ticket",
        "e-Remito",
        "e-Nota de Crédito",
        "e-Nota de Credito",
        "Nota de Crédito",
        "Nota de Credito",
        "Factura",
        "Ticket",
    ]:
        if re.search(re.escape(cand), blob, re.I):
            tipo = cand.replace("Credito", "Crédito")
            break

    if not tipo:
        m2 = re.search(r"\b(\d{12})\b\s*(.+)$", line)
        if m2:
            maybe = _norm(m2.group(2))
            if 0 < len(maybe) <= 40:
                tipo = maybe

    return rut, tipo, debug


def parse_serie_numero_pago_venc(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    idx = find_line_index(lines, r"\bSERIE\b.*\bNUMERO\b.*\bFORMA\b.*\bPAGO\b")
    if idx is None:
        return None, None, None, None, debug

    nxt = after_nonempty(lines, idx, k=2)
    sn_line = nxt[0] if nxt else ""
    debug["sn_line"] = sn_line

    m = re.search(r"\b([A-Z]{1,2})\s*0*([0-9]{3,10})\b", sn_line)
    serie = m.group(1) if m else None
    numero = m.group(2).lstrip("0") if m else None
    if numero == "":
        numero = "0"

    forma = None
    for cand in ["Credito", "Crédito", "Contado", "Tarjeta", "Debito", "Débito", "Transferencia"]:
        if re.search(rf"\b{re.escape(cand)}\b", sn_line, re.I):
            forma = cand.replace("Debito", "Débito").replace("Credito", "Crédito")
            break

    venc = parse_date(sn_line)
    if not venc and len(nxt) > 1:
        venc = parse_date(nxt[1])
        debug["sn_line_2"] = nxt[1]

    return serie, numero, forma, venc, debug


def parse_pais_fecha_moneda(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    idx = find_line_index(lines, r"\bPAIS\b.*\bFECHA\b.*\bDOCUMENTO\b.*\bMONEDA\b")
    if idx is None:
        return None, None, None, debug

    nxt = after_nonempty(lines, idx, k=2)
    line = nxt[0] if nxt else ""
    debug["pfm_line"] = line

    m = re.search(r"\b([A-Z]{2})\b", line)
    pais = m.group(1) if m else None
    fecha = parse_date(line)
    moneda = None

    if fecha:
        dm = re.search(DATE_RE, line)
        after = line[dm.end():].strip() if dm else ""
        if after:
            moneda = after

    return pais, fecha, moneda, debug


def parse_receptor_y_emisor(lines: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    """
    Handles both:
      (A) "PURA PALTA SAS RUT RECEPTOR RAZON SOCIAL"
          "218849... MODELO NATURAL..."
      (B) "RUT RECEPTOR RAZON SOCIAL"
          "Cafe Bahia"
          "218849... MODELO NATURAL..."
    """
    debug: Dict[str, Any] = {}
    idx = find_line_index(lines, r"RUT\s+RECEPTOR.*RAZON\s+SOCIAL")
    if idx is None:
        return None, None, None, debug

    label_line = lines[idx]
    debug["rr_label_line"] = label_line

    emisor = None
    m_inline = re.search(r"^(.*?)\s+RUT\s+RECEPTOR", label_line, re.I)
    if m_inline:
        cand = _norm(m_inline.group(1))
        if cand and not re.search(r"SERIE|NUMERO|FORMA|PAGO|VENCIMIENTO|RUT|TIPO|DOCUMENTO", cand, re.I):
            emisor = cand
            debug["emisor_inline"] = cand

    nxt = after_nonempty(lines, idx, k=4)
    debug["rr_block"] = nxt

    rut_rec = None
    razon_rec = None

    if nxt:
        first = nxt[0]

        if not re.search(r"\b\d{12}\b", first):
            if not emisor and len(first) <= 80:
                emisor = first
                debug["emisor_after_label"] = first

            if len(nxt) > 1:
                second = nxt[1]
                m2 = re.search(r"\b(\d{12})\b\s*(.+)$", second)
                if m2:
                    rut_rec = m2.group(1)
                    razon_rec = _norm(m2.group(2))
                    debug["receptor_line"] = second

        else:
            m2 = re.search(r"\b(\d{12})\b\s*(.+)$", first)
            if m2:
                rut_rec = m2.group(1)
                razon_rec = _norm(m2.group(2))
                debug["receptor_line"] = first

            if not emisor:
                for back in range(1, 8):
                    j = idx - back
                    if j < 0:
                        break
                    cand = lines[j]
                    if not cand:
                        continue
                    if re.search(r"\d{6,}", cand):
                        continue
                    if re.search(r"SERIE|NUMERO|FORMA|PAGO|VENCIMIENTO|RUT|TIPO|DOCUMENTO", cand, re.I):
                        continue
                    emisor = cand
                    debug["emisor_back"] = cand
                    break

    return rut_rec, razon_rec, emisor, debug


def parse_totals(lines: List[str]) -> Tuple[Optional[str], Optional[float], Optional[float], Dict[str, Any]]:
    debug: Dict[str, Any] = {}

    total_str = None
    total_num = None
    for ln in lines:
        if re.search(r"\bTotal\s+a\s+pagar\b", ln, re.I):
            m = re.search(r"Total\s+a\s+pagar\s*([0-9\.\,]+)", ln, re.I)
            if m:
                total_str = m.group(1)
                total_num = parse_decimal_uy(total_str)
                debug["total_line"] = ln
            break

    def _find(label_regex: str) -> Tuple[Optional[str], Optional[str]]:
        for ln in lines:
            if re.search(label_regex, ln, re.I):
                m = re.search(label_regex + r"\s*([0-9\.\,]+)", ln, re.I)
                if m:
                    return m.group(1), ln
        return None, None

    sub10, l10 = _find(r"Subtotal\s+gravado\(?10%\)?")
    sub22, l22 = _find(r"Subtotal\s+gravado\s*\(22%\)")
    subng, lng = _find(r"Subtotal\s+no\s+gravado")

    if l10:
        debug["sub10_line"] = l10
    if l22:
        debug["sub22_line"] = l22
    if lng:
        debug["sub_no_gravado_line"] = lng

    sub10n = parse_decimal_uy(sub10) if sub10 else None
    sub22n = parse_decimal_uy(sub22) if sub22 else None
    subngn = parse_decimal_uy(subng) if subng else None

    base = 0.0
    count = 0
    for v in (sub10n, sub22n, subngn):
        if v is not None:
            base += v
            count += 1
    base_num = round(base, 2) if count else None

    return total_str, total_num, base_num, debug


# -----------------------------
# Output schema
# -----------------------------
@dataclass
class InvoiceResult:
    fecha: Optional[str] = None
    serie: Optional[str] = None
    folio: Optional[str] = None
    serie_y_folio: Optional[str] = None

    razon_social: Optional[str] = None
    razon_social_receptor: Optional[str] = None

    rut_emisor: Optional[str] = None
    rut_receptor: Optional[str] = None

    tipo_documento: Optional[str] = None
    forma_pago: Optional[str] = None
    vencimiento: Optional[str] = None
    moneda: Optional[str] = None

    importe_total_con_iva: Optional[str] = None
    importe_total_con_iva_num: Optional[float] = None
    importe_sin_iva: Optional[str] = None
    importe_sin_iva_num: Optional[float] = None
    importe_sin_iva_fuente: Optional[str] = None

    es_nota_de_credito: bool = False

    _archivo: Optional[str] = None
    _fuente: Optional[str] = None
    _debug: Optional[Dict[str, Any]] = None


def parse_pdf_dgi(path: str, debug: bool = False) -> InvoiceResult:
    text = extract_pdf_text(path)
    lines = split_lines(text)

    inv = InvoiceResult(_archivo=path, _fuente="pdf_text")

    rut_emisor, tipo_doc, dbg1 = parse_rut_tipo_doc(lines)
    inv.rut_emisor = rut_emisor
    inv.tipo_documento = tipo_doc

    serie, folio, forma, venc, dbg2 = parse_serie_numero_pago_venc(lines)
    inv.serie = serie
    inv.folio = folio
    inv.forma_pago = forma
    inv.vencimiento = venc
    inv.serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_rec, razon_rec, emisor_name, dbg3 = parse_receptor_y_emisor(lines)
    inv.rut_receptor = rut_rec
    inv.razon_social_receptor = razon_rec
    inv.razon_social = emisor_name

    _, fecha, moneda, dbg4 = parse_pais_fecha_moneda(lines)
    inv.fecha = fecha
    inv.moneda = moneda

    total_str, total_num, base_num, dbg5 = parse_totals(lines)
    inv.importe_total_con_iva = total_str
    inv.importe_total_con_iva_num = total_num

    if base_num is not None:
        inv.importe_sin_iva_num = base_num
        inv.importe_sin_iva = fmt_decimal_uy(base_num)
        inv.importe_sin_iva_fuente = "subtotales"

    blob = " ".join(lines[:80])
    inv.es_nota_de_credito = bool(
        re.search(r"nota\s+de\s+cr[eé]dito", blob, re.I)
        or ((tipo_doc or "").lower().find("crédito") >= 0)
    )

    if debug:
        inv._debug = {
            "rut_tipo": dbg1,
            "serie_numero": dbg2,
            "receptor": dbg3,
            "pfm": dbg4,
            "totals": dbg5,
        }

    return inv


def parse_image(path: str, args) -> InvoiceResult:
    inv = InvoiceResult(_archivo=path, _fuente="image_ocr_easyocr")
    if args.no_ocr:
        inv._fuente = "no_ocr"
        return inv

    text = ocr_image_easyocr(
        path,
        mode=args.ocr_mode,
        max_dim=args.max_dim,
        max_pixels=args.max_pixels,
    )
    lines = split_lines(text)

    rut_emisor, tipo_doc, dbg1 = parse_rut_tipo_doc(lines)
    inv.rut_emisor = rut_emisor
    inv.tipo_documento = tipo_doc

    serie, folio, forma, venc, dbg2 = parse_serie_numero_pago_venc(lines)
    inv.serie = serie
    inv.folio = folio
    inv.forma_pago = forma
    inv.vencimiento = venc
    inv.serie_y_folio = f"{serie}-{folio}" if serie and folio else None

    rut_rec, razon_rec, emisor_name, dbg3 = parse_receptor_y_emisor(lines)
    inv.rut_receptor = rut_rec
    inv.razon_social_receptor = razon_rec
    inv.razon_social = emisor_name

    _, fecha, moneda, dbg4 = parse_pais_fecha_moneda(lines)
    inv.fecha = fecha
    inv.moneda = moneda

    total_str, total_num, base_num, dbg5 = parse_totals(lines)
    inv.importe_total_con_iva = total_str
    inv.importe_total_con_iva_num = total_num
    if base_num is not None:
        inv.importe_sin_iva_num = base_num
        inv.importe_sin_iva = fmt_decimal_uy(base_num)
        inv.importe_sin_iva_fuente = "subtotales"

    blob = " ".join(lines[:120])
    inv.es_nota_de_credito = bool(
        re.search(r"nota\s+de\s+cr[eé]dito", blob, re.I)
        or ((tipo_doc or "").lower().find("crédito") >= 0)
    )

    if args.debug:
        inv._debug = {
            "rut_tipo": dbg1,
            "serie_numero": dbg2,
            "receptor": dbg3,
            "pfm": dbg4,
            "totals": dbg5,
        }

    return inv


def iter_files(root: str) -> Iterable[str]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for base, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                yield os.path.join(base, fn)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        rows = [asdict(InvoiceResult())]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_gold(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("gold.json debe ser un objeto JSON (diccionario).")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[os.path.basename(k)] = v
    return out


def report_against_gold(results: List[Dict[str, Any]], gold: Dict[str, Dict[str, Any]]) -> str:
    fields = ["fecha", "rut_emisor", "serie", "folio", "importe_total_con_iva_num"]
    docs = [r for r in results if os.path.basename(r.get("_archivo") or "") in gold]
    lines: List[str] = []
    lines.append("\n=== REPORT ===")
    lines.append(f"Docs con gold: {len(docs)}")

    if not docs:
        return "\n".join(lines)

    ok_counts = {f: 0 for f in fields}
    mismatches: List[str] = []

    for r in docs:
        fn = os.path.basename(r.get("_archivo") or "")
        g = gold[fn]
        for f in fields:
            pred = r.get(f)
            exp = g.get(f)
            if pred is None or exp is None:
                mismatches.append(f"* {fn}\n  - {f}: pred={pred!r} gold={exp!r}")
                continue

            match = False
            if f.endswith("_num"):
                try:
                    match = abs(float(pred) - float(exp)) < 0.01
                except Exception:
                    match = False
            else:
                match = str(pred).strip() == str(exp).strip()

            if match:
                ok_counts[f] += 1
            else:
                mismatches.append(f"* {fn}\n  - {f}: pred={pred!r} gold={exp!r}")

    for f in fields:
        pct = 100.0 * ok_counts[f] / len(docs)
        lines.append(f"{f}: {ok_counts[f]}/{len(docs)} ({pct:.1f}%)")

    if mismatches:
        lines.append("\n--- MISMATCHES ---")
        lines.extend(mismatches[:200])

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parser simple de e-Facturas uruguayas (PDF primero; OCR opcional)."
    )
    p.add_argument("path", help="Carpeta con facturas o un archivo individual.")
    p.add_argument("--json", action="store_true", help="Imprime JSON por stdout.")
    p.add_argument("--csv", type=str, default=None, help="Ruta de salida CSV.")
    p.add_argument("--debug", action="store_true", help="Incluye _debug con pistas de parsing.")
    p.add_argument("--no-ocr", dest="no_ocr", action="store_true", help="No usar OCR (recomendado si solo usás PDFs con texto).")

    p.add_argument("--ocr-mode", choices=["fast", "balanced", "accurate"], default="balanced")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--max-dim", type=int, default=1800, help="Máximo lado para OCR (imágenes).")
    p.add_argument("--max-pixels", type=int, default=2_500_000, help="Máximo píxeles para OCR (imágenes).")
    p.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (Windows).")

    p.add_argument("--report", action="store_true", help="Muestra reporte comparando contra gold.json.")
    p.add_argument("--gold", type=str, default=None, help="Ruta a gold.json (para --report).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.low_priority:
        set_low_priority_windows()
    limit_threads(args.cpu_threads)

    paths: List[str] = []
    if os.path.isfile(args.path):
        paths = [args.path]
    else:
        paths = list(iter_files(args.path))

    results: List[Dict[str, Any]] = []
    for fp in paths:
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext == ".pdf":
                inv = parse_pdf_dgi(fp, debug=args.debug)
            else:
                inv = parse_image(fp, args)
            results.append(asdict(inv))
        except Exception as e:
            inv = InvoiceResult(_archivo=fp, _fuente="error")
            if args.debug:
                inv._debug = {"error": str(e)}
            results.append(asdict(inv))
            if args.debug:
                print(f"[ERROR] {os.path.basename(fp)}: {e}", file=sys.stderr)

    if args.report:
        if not args.gold:
            print("Te falta --gold ruta\\gold.json", file=sys.stderr)
            return 2
        gold = load_gold(args.gold)
        print(report_against_gold(results, gold))
        return 0

    if args.csv:
        write_csv(args.csv, results)

    if args.json or not args.csv:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
