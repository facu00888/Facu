#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
invoice_parser.py
Parser de facturas (UY) desde PDFs (texto) y opcionalmente imágenes (OCR).
- Prioriza PDF con texto (preciso, rápido, sin entrenamiento).
- Evita depender del nombre del archivo.
- Exporta CSV/XLSX con tipos correctos (montos numéricos NO como fechas).

Uso típico (recomendado):
  python invoice_parser.py "C:\\Proyectos\\Facu\\Facturas" --xlsx "C:\\Proyectos\\Facu\\salida.xlsx" --csv "C:\\Proyectos\\Facu\\salida.csv" --no-ocr

Reporte vs gold:
  python invoice_parser.py "C:\\Proyectos\\Facu\\Facturas" --report --gold "C:\\Proyectos\\Facu\\gold.json" --no-ocr
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --------------------------
# Config salida
# --------------------------

OUTPUT_FIELDS = [
    "fecha",
    "serie",
    "folio",
    "serie_y_folio",
    "razon_social",
    "razon_social_receptor",
    "rut_emisor",
    "rut_receptor",
    "tipo_documento",
    "forma_pago",
    "vencimiento",
    "moneda",
    "importe_total_con_iva",
    "importe_total_con_iva_num",
    "importe_sin_iva",
    "importe_sin_iva_num",
    "importe_sin_iva_fuente",
    "es_nota_de_credito",
    "_archivo",
    "_fuente",
    "_debug",
]

DATE_FIELDS = {"fecha", "vencimiento"}
MONEY_NUM_FIELDS = {"importe_total_con_iva_num", "importe_sin_iva_num"}


# --------------------------
# Utilidades
# --------------------------

def _normspace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _clean_lines(text: str) -> List[str]:
    lines: List[str] = []
    for l in (text or "").splitlines():
        l = l.strip()
        if not l:
            continue
        l = re.sub(r"\s+", " ", l)
        lines.append(l)
    return lines


def parse_date_uy(s: Any) -> Optional[_dt.date]:
    if not s:
        return None
    if isinstance(s, _dt.date) and not isinstance(s, _dt.datetime):
        return s
    if isinstance(s, _dt.datetime):
        return s.date()
    if not isinstance(s, str):
        s = str(s)
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return _dt.date(y, mo, d)
    except ValueError:
        return None


def fmt_date_uy(v: Any) -> Optional[str]:
    d = parse_date_uy(v)
    if not d:
        return None
    return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"


def parse_uy_amount_str(s: Any) -> Optional[str]:
    """Extrae el primer token numérico 'tipo UY' (1.234,56 / 1234,56 / 1234.56)."""
    if s is None:
        return None
    s = str(s)
    m = re.search(r"[-(]?\d[\d\.\,]*\d", s)
    return m.group(0) if m else None


def parse_uy_amount(s: Any) -> Optional[float]:
    """Convierte 1.234,56 -> 1234.56 (float)."""
    if s is None:
        return None
    s = str(s).strip().replace(" ", "")
    if not s:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    if s.startswith("-"):
        neg = True
        s = s[1:]

    s = re.sub(r"[^0-9\.,]", "", s)
    if not s:
        return None

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    else:
        # si hay más de un punto, asumimos miles y dejamos el último como decimal
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def is_pdf(path: str) -> bool:
    return os.path.splitext(path.lower())[1] == ".pdf"


def is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------
# PDF text extraction
# --------------------------

def extract_pdf_text(path: str) -> str:
    """
    Extrae texto de PDF. Primero intenta pdfplumber (suele andar bien),
    y si no está, intenta PyMuPDF (fitz).
    """
    try:
        import pdfplumber  # type: ignore
        texts: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                texts.append(t)
        return "\n".join(texts)
    except Exception as e_pdfplumber:
        # fallback a fitz
        try:
            import fitz  # type: ignore
            doc = fitz.open(path)
            texts = []
            for page in doc:
                texts.append(page.get_text("text"))
            return "\n".join(texts)
        except Exception as e_fitz:
            raise RuntimeError(f"No pude leer PDF (pdfplumber/fitz): {e_pdfplumber} / {e_fitz}") from e_fitz


# --------------------------
# Parsing específico UY (PDF-text)
# --------------------------

def parse_invoice_from_text_uy(text: str) -> Dict[str, Any]:
    """
    Heurísticas para e-Factura UY.
    No depende del nombre del archivo.
    """
    lines = _clean_lines(text)
    norm = _normspace(text)

    out: Dict[str, Any] = {
        "fecha": None,
        "serie": None,
        "folio": None,
        "serie_y_folio": None,
        "razon_social": None,
        "razon_social_receptor": None,
        "rut_emisor": None,
        "rut_receptor": None,
        "tipo_documento": None,
        "forma_pago": None,
        "vencimiento": None,
        "moneda": None,
        "importe_total_con_iva": None,
        "importe_total_con_iva_num": None,
        "importe_sin_iva": None,
        "importe_sin_iva_num": None,
        "importe_sin_iva_fuente": None,
        "es_nota_de_credito": False,
        "_debug": {},
    }

    # Nota de crédito?
    if re.search(r"nota\s+de\s+cr[eé]dito", norm, re.I) or re.search(r"\bNC\b", norm):
        out["es_nota_de_credito"] = True

    # RUT EMISOR + tipo documento
    for i, l in enumerate(lines):
        if re.search(r"RUT\s+EMISOR", l, re.I) and re.search(r"TIPO\s+DOCUMENTO", l, re.I):
            m = re.search(r"RUT\s+EMISOR\s+TIPO\s+DOCUMENTO\s+(\d{11,12})\s+(.+)$", l, re.I)
            if m:
                out["rut_emisor"] = m.group(1)
                out["tipo_documento"] = m.group(2).strip()
                out["_debug"]["rut_tipo_line"] = l
                break
            # si viene en la línea siguiente
            for j in range(i + 1, min(i + 5, len(lines))):
                m2 = re.search(r"(\d{11,12})\s+(.+)$", lines[j])
                if m2:
                    out["rut_emisor"] = m2.group(1)
                    out["tipo_documento"] = m2.group(2).strip()
                    out["_debug"]["rut_tipo_line"] = lines[j]
                    break
            break

    # SERIE/FOLIO/FORMA PAGO/VENCIMIENTO
    for i, l in enumerate(lines):
        if (re.search(r"SERIE", l, re.I)
            and re.search(r"NUMERO|NÚMERO", l, re.I)
            and re.search(r"FORMA\s+DE\s+PAGO", l, re.I)):
            blob = " ".join(lines[i:i + 4])
            m = re.search(r"\b([A-Z])\s*0*([0-9]{3,10})\s+(Credito|Contado)\b(?:\s+(\d{1,2}/\d{1,2}/\d{4}))?",
                          blob, re.I)
            if m:
                out["serie"] = m.group(1).upper()
                out["folio"] = m.group(2)
                out["forma_pago"] = m.group(3).capitalize()
                out["vencimiento"] = m.group(4) if m.group(4) else None
                out["_debug"]["serie_line"] = blob
            break

    if not out["folio"]:
        # fallback (por si el PDF rompe líneas raro)
        m = re.search(r"\b([A-Z])\s*0*([0-9]{3,10})\s+(Credito|Contado)\b", norm, re.I)
        if m:
            out["serie"] = m.group(1).upper()
            out["folio"] = m.group(2)
            out["forma_pago"] = m.group(3).capitalize()

    if out["serie"] and out["folio"]:
        out["serie_y_folio"] = f"{out['serie']}-{out['folio']}"

    # FECHA + MONEDA (evita enganchar vencimientos de CAE)
    for i, l in enumerate(lines):
        if re.search(r"PAIS\s+FECHA\s+DE\s+DOCUMENTO\s+MONEDA", l, re.I):
            blob = " ".join(lines[i:i + 3])
            m = re.search(
                r"\b([A-Z]{2})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(.+?)(?=\s+CONCEPTO|\s+DESC|\s+CANT|\s+IMPORTE|$)",
                blob, re.I
            )
            if m:
                out["fecha"] = m.group(2)
                out["moneda"] = m.group(3).strip()
                out["_debug"]["pfm_line"] = m.group(0)
            break

    if not out["fecha"]:
        # fallback simple
        m = re.search(r"\bUY\s+(\d{1,2}/\d{1,2}/\d{4})\s+([A-Za-zÁÉÍÓÚÑ\s]+)\b", norm, re.I)
        if m:
            out["fecha"] = m.group(1)
            out["moneda"] = m.group(2).strip()

    # RUT RECEPTOR / receptor name / razon_social (emisor)
    # (1) emisor inline antes de "RUT RECEPTOR RAZON SOCIAL" (caso PURA PALTA)
    for l in lines:
        if re.search(r"RUT\s+RECEPTOR\s+RAZON\s+SOCIAL", l, re.I):
            m = re.search(r"^(.*?)\s+RUT\s+RECEPTOR\s+RAZON\s+SOCIAL\b", l, re.I)
            if m:
                prefix = m.group(1).strip()
                if prefix and not re.search(r"RUT\s+EMISOR|TIPO\s+DOCUMENTO|SERIE\s+NUMERO", prefix, re.I):
                    out["razon_social"] = prefix

            m2 = re.search(r"RUT\s+RECEPTOR\s+RAZON\s+SOCIAL\s+(\d{11,12})\s+(.+)$", l, re.I)
            if m2:
                out["rut_receptor"] = m2.group(1)
                out["razon_social_receptor"] = m2.group(2).strip(" -")
            break

    # (2) layout común (pdfplumber “barre” columnas raro): después del label, viene EMISOR y luego el receptor
    if not out["razon_social"] or not out["rut_receptor"]:
        for i, l in enumerate(lines):
            if re.search(r"RUT\s+RECEPTOR\s+RAZON\s+SOCIAL", l, re.I):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and not out["razon_social"]:
                    if not re.search(r"\d{6,}", lines[j]):  # si tiene muchos dígitos no es nombre
                        out["razon_social"] = lines[j].strip()
                        out["_debug"]["rr_line"] = lines[j].strip()

                k = j + 1
                while k < len(lines):
                    m = re.search(r"(\d{11,12})\s+(.+)$", lines[k])
                    if m:
                        if not out["rut_receptor"]:
                            out["rut_receptor"] = m.group(1)
                        if not out["razon_social_receptor"]:
                            out["razon_social_receptor"] = m.group(2).strip()
                        break
                    k += 1
                break

    # Limpieza receptor
    if out["razon_social_receptor"]:
        out["razon_social_receptor"] = re.sub(r"^\(\d+\)\s*", "", out["razon_social_receptor"]).strip()

    # TOTAL (con IVA)
    m = re.search(r"Total\s+a\s+pagar\s*([0-9][0-9\.\,]*)", norm, re.I)
    if m:
        out["importe_total_con_iva"] = m.group(1)
        out["_debug"]["total_src"] = "total_a_pagar"
    else:
        for l in lines:
            if re.search(r"Total\s+a\s+pagar", l, re.I):
                am = parse_uy_amount_str(l)
                if am:
                    out["importe_total_con_iva"] = am
                    out["_debug"]["total_src"] = "total_line"
                    break

    out["importe_total_con_iva_num"] = parse_uy_amount(out["importe_total_con_iva"]) if out["importe_total_con_iva"] else None

    # SUBTOTALES (sin IVA)
    def find_amount(label_regex: str) -> Optional[str]:
        m3 = re.search(label_regex + r"\s*([0-9][0-9\.\,]*)", norm, re.I)
        return m3.group(1) if m3 else None

    sub10 = find_amount(r"Subtotal\s+gravado\s*\(10%\)")
    iva10 = find_amount(r"Total\s+iva\s*\(10%\)")
    sub22 = find_amount(r"Subtotal\s+gravado\s*\(22%\)")
    iva22 = find_amount(r"Total\s+iva\s*\(22%\)")
    subng = find_amount(r"Subtotal\s+no\s+gravado")

    out["_debug"].update({"sub10": sub10, "iva10": iva10, "sub22": sub22, "iva22": iva22, "sub_no_gravado": subng})

    if any(x is not None for x in [sub10, sub22, subng]):
        sin_iva = (parse_uy_amount(sub10) or 0.0) + (parse_uy_amount(sub22) or 0.0) + (parse_uy_amount(subng) or 0.0)
        out["importe_sin_iva_num"] = sin_iva
        out["importe_sin_iva_fuente"] = "subtotales"

    # (opcional) si querés string “bonito” para sin IVA, lo reconstruimos desde num
    if out["importe_sin_iva_num"] is not None:
        # formateo UY básico: miles con punto, decimales con coma
        val = float(out["importe_sin_iva_num"])
        s = f"{val:,.2f}"  # 1,234.56
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # 1.234,56
        out["importe_sin_iva"] = s

    return out


# --------------------------
# OCR (opcional, no recomendado si conseguís PDF-text)
# --------------------------

def set_low_priority_if_possible(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        # Windows:
        if hasattr(psutil, "IDLE_PRIORITY_CLASS"):
            p.nice(psutil.IDLE_PRIORITY_CLASS)
        else:
            p.nice(19)
    except Exception:
        pass


def clamp_threads(n: int) -> int:
    try:
        n = int(n)
    except Exception:
        n = 1
    return max(1, min(n, 64))


def set_thread_env(cpu_threads: int) -> None:
    # evita que numpy/blas/torch “se crean dueños del mundo”
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))


def ocr_image_easyocr(path: str, cpu_threads: int, max_dim: int, max_pixels: int) -> str:
    """
    OCR simple con easyocr. Si no está instalado, levanta error.
    Recomendación: usar --no-ocr y PDFs con texto.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import easyocr  # type: ignore
    except Exception as e:
        raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch) y opencv-python.") from e

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("No pude leer imagen.")

    h, w = img.shape[:2]
    scale = 1.0

    # límite por dimensión
    if max_dim and max(h, w) > max_dim:
        scale = min(scale, max_dim / float(max(h, w)))

    # límite por píxeles
    if max_pixels and (h * w) > max_pixels:
        scale = min(scale, (max_pixels / float(h * w)) ** 0.5)

    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # prepro clásico: gris + bilateral + threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 11)

    reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    # easyocr espera RGB
    thr_rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    res = reader.readtext(thr_rgb, detail=0, paragraph=True)
    return "\n".join(res)


# --------------------------
# IO: CSV / XLSX
# --------------------------

def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(r: Dict[str, Any]) -> Tuple:
        d = parse_date_uy(r.get("fecha"))
        # para ordenar, si no hay fecha, lo manda al final
        d_key = d or _dt.date(9999, 12, 31)
        serie = (r.get("serie") or "")
        folio = (r.get("folio") or "")
        try:
            folio_i = int(re.sub(r"\D", "", str(folio)) or "0")
        except Exception:
            folio_i = 0
        return (d_key, serie, folio_i, str(folio))
    return sorted(rows, key=key_fn)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            # fechas como dd/mm/yyyy
            for k in DATE_FIELDS:
                rr[k] = fmt_date_uy(rr.get(k))
            # debug como json
            rr["_debug"] = safe_json(rr.get("_debug"))
            w.writerow({k: rr.get(k) for k in OUTPUT_FIELDS})


def write_xlsx(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from openpyxl import Workbook  # type: ignore
    from openpyxl.utils import get_column_letter  # type: ignore

    wb = Workbook()
    ws = wb.active
    ws.title = "facturas"
    ws.append(OUTPUT_FIELDS)
    ws.freeze_panes = "A2"

    for r in rows:
        row_vals: List[Any] = []
        for h in OUTPUT_FIELDS:
            v = r.get(h)
            if h in DATE_FIELDS:
                row_vals.append(parse_date_uy(v))
            elif h in MONEY_NUM_FIELDS:
                if isinstance(v, (int, float)):
                    row_vals.append(float(v))
                else:
                    row_vals.append(parse_uy_amount(v))
            elif h.endswith("_num"):
                # otros num: mantenelos num
                if isinstance(v, (int, float)):
                    row_vals.append(float(v))
                else:
                    row_vals.append(parse_uy_amount(v))
            elif h == "_debug":
                row_vals.append(safe_json(v))
            else:
                row_vals.append(v)
        ws.append(row_vals)

    # Formatos (acá estaba tu bug: jamás uses formato fecha en montos)
    for col_idx, h in enumerate(OUTPUT_FIELDS, start=1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = min(max(len(h) + 2, 12), 48)
        for row_idx in range(2, ws.max_row + 1):
            c = ws.cell(row=row_idx, column=col_idx)
            if h in DATE_FIELDS and isinstance(c.value, _dt.date):
                c.number_format = "dd/mm/yyyy"
            elif h in MONEY_NUM_FIELDS and isinstance(c.value, (int, float)):
                c.number_format = '#,##0.00'
            elif h.endswith("_num") and isinstance(c.value, (int, float)):
                c.number_format = '0.00'

    wb.save(path)


# --------------------------
# Gold / Report
# --------------------------

def load_gold(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Formatos aceptados:
    1) dict: { "archivo.pdf": {campos...}, ... }
    2) list: [ {"_archivo": "archivo.pdf", ...}, ... ]  o {"archivo":"..."}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    if isinstance(data, list):
        out: Dict[str, Dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            fn = item.get("_archivo") or item.get("archivo") or item.get("file")
            if fn:
                out[str(fn)] = item
        return out

    return {}


def compare(pred: Dict[str, Any], gold: Dict[str, Any], fields: List[str]) -> Dict[str, Tuple[Any, Any]]:
    mismatches: Dict[str, Tuple[Any, Any]] = {}
    for f in fields:
        pv = pred.get(f)
        gv = gold.get(f)

        # normalizaciones
        if f in DATE_FIELDS:
            pv = fmt_date_uy(pv)
            gv = fmt_date_uy(gv)

        if f.endswith("_num"):
            try:
                pv_f = float(pv) if pv is not None else None
            except Exception:
                pv_f = None
            try:
                gv_f = float(gv) if gv is not None else None
            except Exception:
                gv_f = None

            if pv_f is None and gv_f is None:
                continue
            if pv_f is None or gv_f is None:
                mismatches[f] = (pv, gv)
                continue
            if abs(pv_f - gv_f) > 0.01:
                mismatches[f] = (pv_f, gv_f)
            continue

        if (pv or None) != (gv or None):
            mismatches[f] = (pv, gv)

    return mismatches


def run_report(rows: List[Dict[str, Any]], gold_path: str) -> int:
    gold = load_gold(gold_path)
    if not gold:
        print("\n=== REPORT ===")
        print("Gold vacío o inválido.")
        return 2

    # index por basename
    pred_by_name: Dict[str, Dict[str, Any]] = {os.path.basename(r.get("_archivo", "")): r for r in rows}
    targets = {k: v for k, v in gold.items() if k in pred_by_name}

    fields = ["fecha", "rut_emisor", "serie", "folio", "importe_total_con_iva_num"]

    print("\n=== REPORT ===")
    print(f"Docs con gold: {len(targets)}")

    if not targets:
        return 1

    correct = {f: 0 for f in fields}
    total = len(targets)
    mismatches_all: List[Tuple[str, Dict[str, Tuple[Any, Any]]]] = []

    for fn, g in targets.items():
        p = pred_by_name[fn]
        mm = compare(p, g, fields)
        if mm:
            mismatches_all.append((fn, mm))
        for f in fields:
            if f not in mm:
                correct[f] += 1

    for f in fields:
        pct = (correct[f] / total) * 100
        print(f"{f}: {correct[f]}/{total} ({pct:.1f}%)")

    if mismatches_all:
        print("\n--- MISMATCHES ---")
        for fn, mm in mismatches_all:
            print(f"* {fn}")
            for f, (pv, gv) in mm.items():
                print(f"  - {f}: pred={pv!r} gold={gv!r}")

    return 0


# --------------------------
# Runner
# --------------------------

def collect_files(input_path: str) -> List[str]:
    input_path = os.path.abspath(input_path)
    files: List[str] = []

    if os.path.isfile(input_path):
        return [input_path]

    for root, _, fnames in os.walk(input_path):
        for fn in fnames:
            ext = os.path.splitext(fn.lower())[1]
            if ext in {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                files.append(os.path.join(root, fn))

    files.sort()
    return files


def process_file(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    base = os.path.basename(path)
    try:
        if is_pdf(path):
            text = extract_pdf_text(path)
            if not _normspace(text):
                raise RuntimeError("PDF sin texto útil (probablemente escaneado).")
            parsed = parse_invoice_from_text_uy(text)
            parsed["_archivo"] = path
            parsed["_fuente"] = "pdf_text"
            if args.debug:
                dbg = parsed.get("_debug") or {}
                print(f"\n=== {base} (pdf_text) ===")
                print("---[DEBUG]---")
                print(safe_json(dbg))
                print("=== FIN ===")
            return parsed

        if is_image(path):
            if args.no_ocr:
                return {
                    **{k: None for k in OUTPUT_FIELDS},
                    "es_nota_de_credito": False,
                    "_archivo": path,
                    "_fuente": "skipped_image_no_ocr",
                    "_debug": {"error": "Imagen omitida (--no-ocr)."},
                }

            text = ocr_image_easyocr(path, cpu_threads=args.cpu_threads, max_dim=args.max_dim, max_pixels=args.max_pixels)
            if not _normspace(text):
                raise RuntimeError("OCR falló: no se obtuvo texto útil.")
            parsed = parse_invoice_from_text_uy(text)
            parsed["_archivo"] = path
            parsed["_fuente"] = "image_ocr_easyocr"
            if args.debug:
                print(f"\n=== {base} (image_ocr_easyocr) ===")
                print("---[OCR TEXT]---")
                print(text[:1500])
                print("=== FIN ===")
            return parsed

        return {
            **{k: None for k in OUTPUT_FIELDS},
            "es_nota_de_credito": False,
            "_archivo": path,
            "_fuente": "skipped_unknown",
            "_debug": {"error": "Tipo de archivo no soportado."},
        }

    except Exception as e:
        if args.debug:
            print(f"[ERROR] {base}: {e}")
        return {
            **{k: None for k in OUTPUT_FIELDS},
            "es_nota_de_credito": False,
            "_archivo": path,
            "_fuente": "error",
            "_debug": {"error": str(e)},
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Carpeta o archivo (PDF/JPG/PNG/...)")
    ap.add_argument("--json", action="store_true", help="Imprime JSON por stdout (default si no hay outputs).")
    ap.add_argument("--csv", default=None, help="Ruta de salida CSV.")
    ap.add_argument("--xlsx", default=None, help="Ruta de salida XLSX.")
    ap.add_argument("--no-ocr", action="store_true", help="No hace OCR a imágenes (recomendado si tenés PDFs con texto).")
    ap.add_argument("--debug", action="store_true", help="Debug verboso.")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (si se puede).")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Límite de threads para librerías pesadas.")
    ap.add_argument("--max-dim", type=int, default=1800, help="Máximo lado de imagen para OCR.")
    ap.add_argument("--max-pixels", type=int, default=2_500_000, help="Máx píxeles de imagen para OCR.")
    ap.add_argument("--report", action="store_true", help="Genera reporte vs gold.")
    ap.add_argument("--gold", default=None, help="Ruta a gold.json para --report.")

    args = ap.parse_args()

    args.cpu_threads = clamp_threads(args.cpu_threads)
    set_thread_env(args.cpu_threads)
    set_low_priority_if_possible(args.low_priority)

    files = collect_files(args.input)
    if not files:
        print("No encontré archivos para procesar.")
        return 1

    rows: List[Dict[str, Any]] = []
    for p in files:
        rows.append(process_file(p, args))

    rows = sort_rows(rows)

    # outputs
    wrote_any = False
    if args.csv:
        write_csv(rows, args.csv)
        wrote_any = True

    if args.xlsx:
        write_xlsx(rows, args.xlsx)
        wrote_any = True

    if args.report:
        if not args.gold:
            print("Falta --gold para --report.")
            return 2
        return run_report(rows, args.gold)

    if args.json or not wrote_any:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
