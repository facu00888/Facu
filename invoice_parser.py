#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------
# Utilidades de parsing
# ---------------------------

RUT_RE = re.compile(r"\b(\d{12})\b")
DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
MONEY_RE = re.compile(r"[-]?\d{1,3}(?:\.\d{3})*(?:,\d{2})|[-]?\d+(?:,\d{2})")

def norm_text(s: str) -> str:
    # Normaliza whitespace pero conserva saltos de línea (útil para "línea siguiente")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # colapsa espacios y tabs
    s = re.sub(r"[ \t]+", " ", s)
    # colapsa muchas líneas vacías
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def norm_space(s: str) -> str:
    # Para regex donde no importa el salto de línea
    return re.sub(r"\s+", " ", s).strip()

def parse_uy_money_to_float(val: str) -> Optional[float]:
    """
    "4.924,04" -> 4924.04
    "6046,00" -> 6046.0
    """
    if not val:
        return None
    v = val.strip()
    v = v.replace(".", "").replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return None

def pick_first(lines: List[str], bad_prefixes: Tuple[str, ...]) -> Optional[str]:
    for ln in lines:
        t = ln.strip()
        if not t:
            continue
        up = t.upper()
        if any(up.startswith(pfx) for pfx in bad_prefixes):
            continue
        # Filtra líneas demasiado “genéricas”
        if up in {"RUT EMISOR TIPO DOCUMENTO", "SERIE NUMERO FORMA DE PAGO VENCIMIENTO"}:
            continue
        return t
    return None

def line_after_label(text: str, label_re: re.Pattern) -> Optional[str]:
    """
    Busca label y devuelve la primera línea no vacía después del match.
    """
    m = label_re.search(text)
    if not m:
        return None
    tail = text[m.end():]
    for ln in tail.split("\n"):
        ln = ln.strip()
        if ln:
            return ln
    return None

def extract_money_after(label_regex: str, text_space: str) -> Optional[str]:
    # label_regex debe estar escapado si hace falta
    m = re.search(label_regex + r"\s*([-]?\d[\d\.\,]*)", text_space, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1)

def safe_strip_paren_prefix(s: str) -> str:
    # "(3532) MODELO..." -> "MODELO..."
    return re.sub(r"^\(\d+\)\s*", "", s.strip())

def parse_serie_numero_forma_venc(line: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Ejemplos:
      "A 492520 Contado 26/11/2025"
      "A 133716 Credito"
    """
    if not line:
        return None, None, None, None

    tokens = line.split()
    if len(tokens) < 2:
        return None, None, None, None

    serie = tokens[0] if re.fullmatch(r"[A-Za-z]", tokens[0]) else None
    folio = tokens[1] if re.fullmatch(r"\d{1,10}", tokens[1]) else None

    venc = None
    if tokens and DATE_RE.fullmatch(tokens[-1]):
        venc = tokens[-1]
        forma_tokens = tokens[2:-1]
    else:
        forma_tokens = tokens[2:]

    forma = " ".join(forma_tokens).strip() if forma_tokens else None
    return serie, folio, forma, venc

def parse_pais_fecha_moneda(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    "UY 26/11/2025 Peso Uruguayo"
    """
    if not line:
        return None, None, None
    tokens = line.split()
    if len(tokens) < 2:
        return None, None, None
    pais = tokens[0]
    fecha = tokens[1] if DATE_RE.fullmatch(tokens[1]) else None
    moneda = " ".join(tokens[2:]).strip() if len(tokens) > 2 else None
    return pais, fecha, moneda


# ---------------------------
# Modelo de salida
# ---------------------------

@dataclass
class InvoiceOut:
    fecha: Optional[str] = None
    serie: Optional[str] = None
    folio: Optional[str] = None
    serie_y_folio: Optional[str] = None

    razon_social: Optional[str] = None            # emisor (mejor que nada)
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


# ---------------------------
# PDF text parsing (DGI-style)
# ---------------------------

LABEL_RUT_EMISOR = re.compile(r"RUT\s*EMISOR\s*TIPO\s*DOCUMENTO", re.IGNORECASE)
LABEL_SERIE_NUM = re.compile(r"SERIE\s*NUMERO\s*FORMA\s*DE\s*PAGO\s*VENCIMIENTO", re.IGNORECASE)
LABEL_RUT_RECEPTOR = re.compile(r"RUT\s*RECEPTOR\s*RAZON\s*SOCIAL", re.IGNORECASE)
LABEL_PAIS_FECHA_MONEDA = re.compile(r"PAIS\s*FECHA\s*DE\s*DOCUMENTO\s*MONEDA", re.IGNORECASE)

def parse_dgi_pdf_text(text: str, debug: bool = False) -> InvoiceOut:
    raw = text
    text = norm_text(text)
    space = norm_space(text)

    out = InvoiceOut()
    dbg: Dict[str, Any] = {}

    # “Razón social” emisor: primera línea relevante
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out.razon_social = pick_first(
        lines,
        bad_prefixes=(
            "RUT EMISOR", "SERIE NUMERO", "RUT RECEPTOR", "DIRECCION",
            "PAIS FECHA", "CONCEPTO", "SUBTOTAL", "TOTAL", "CODIGO", "OBSERVACIONES"
        )
    )

    # Rut emisor + tipo documento en línea siguiente de la etiqueta
    line = line_after_label(text, LABEL_RUT_EMISOR)
    if line:
        # Ej: "215412670012 e-Factura"
        m = re.match(r"(\d{12})\s+(.+)$", line.strip())
        if m:
            out.rut_emisor = m.group(1)
            out.tipo_documento = m.group(2).strip()

    # Serie/numero/forma/venc
    sn_line = line_after_label(text, LABEL_SERIE_NUM)
    serie, folio, forma, venc = parse_serie_numero_forma_venc(sn_line or "")
    out.serie = serie
    out.folio = folio
    out.forma_pago = forma
    out.vencimiento = venc

    # Receptor
    rr_line = line_after_label(text, LABEL_RUT_RECEPTOR)
    if rr_line:
        # Ej: "218849400010 (3532) MODELO NATURAL SRL"
        m = re.match(r"(\d{12})\s*(.*)$", rr_line.strip())
        if m:
            out.rut_receptor = m.group(1)
            out.razon_social_receptor = safe_strip_paren_prefix(m.group(2))

    # País/fecha/moneda (fecha del documento)
    pfm_line = line_after_label(text, LABEL_PAIS_FECHA_MONEDA)
    pais, fecha_doc, moneda = parse_pais_fecha_moneda(pfm_line or "")
    out.fecha = fecha_doc or out.fecha
    out.moneda = moneda

    # Total a pagar
    total_str = extract_money_after(r"Total\s+a\s+pagar", space)
    if total_str:
        out.importe_total_con_iva = total_str
        out.importe_total_con_iva_num = parse_uy_money_to_float(total_str)

    # IVA y subtotales para neto (sin IVA)
    sub10 = extract_money_after(r"Subtotal\s+gravado\s*\(10%\)", space)
    iva10 = extract_money_after(r"Total\s+iva\s*\(10%\)", space)
    sub22 = extract_money_after(r"Subtotal\s+gravado\s*\(22%\)", space)
    iva22 = extract_money_after(r"Total\s+iva\s*\(22%\)", space)
    sub_nogr = extract_money_after(r"Subtotal\s+no\s+gravado", space)

    dbg["sub10"] = sub10
    dbg["iva10"] = iva10
    dbg["sub22"] = sub22
    dbg["iva22"] = iva22
    dbg["sub_no_gravado"] = sub_nogr
    dbg["sn_line"] = sn_line
    dbg["rr_line"] = rr_line
    dbg["pfm_line"] = pfm_line

    net_candidates: List[Tuple[str, Optional[float]]] = []

    def f(x: Optional[str]) -> Optional[float]:
        return parse_uy_money_to_float(x) if x else None

    # Neto por suma de subtotales (mejor si existen)
    s10, s22, sng = f(sub10), f(sub22), f(sub_nogr)
    if s10 is not None and s22 is not None:
        net = (s10 or 0.0) + (s22 or 0.0) + (sng or 0.0)
        net_candidates.append(("subtotales", net))

    # Neto por total - IVA (fallback)
    t = out.importe_total_con_iva_num
    i10, i22 = f(iva10), f(iva22)
    if t is not None and i10 is not None and i22 is not None:
        net_candidates.append(("total_menos_iva", t - i10 - i22))

    # Elegí la fuente más “confiable”
    chosen = None
    if net_candidates:
        # prioriza subtotales
        net_candidates.sort(key=lambda x: 0 if x[0] == "subtotales" else 1)
        chosen = net_candidates[0]

    if chosen and chosen[1] is not None:
        out.importe_sin_iva_num = round(chosen[1], 2)
        # formatea a "es-uy" para mantener consistencia
        out.importe_sin_iva = f"{out.importe_sin_iva_num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        out.importe_sin_iva_fuente = chosen[0]

    # Nota de crédito (heurística)
    out.es_nota_de_credito = bool(re.search(r"nota\s+de\s+cr[eé]dito", space, flags=re.IGNORECASE))

    if out.serie and out.folio:
        out.serie_y_folio = f"{out.serie}-{out.folio}"

    if debug:
        out._debug = dbg

    return out


# ---------------------------
# Lectura de PDF (texto real)
# ---------------------------

def extract_pdf_text(path: Path) -> str:
    import pdfplumber  # lazy import
    parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    return "\n".join(parts).strip()

def pdf_has_useful_text(text: str) -> bool:
    if not text or len(text.strip()) < 40:
        return False
    s = norm_space(text).upper()
    # etiquetas DGI típicas
    return ("RUT EMISOR" in s and "SERIE" in s and "TOTAL A PAGAR" in s) or ("RUT EMISOR" in s and "TIPO DOCUMENTO" in s)


# ---------------------------
# OCR (opcional y perezoso)
# ---------------------------

def set_low_priority_if_possible():
    # Windows: intenta bajar prioridad sin romper nada
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # Windows
        return
    except Exception:
        pass
    # fallback: nada

def ocr_image_easyocr(image_path: Path, cpu_threads: int = 1) -> str:
    # Importes lazies para no cargar torch si no hace falta
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

    try:
        import torch  # type: ignore
        torch.set_num_threads(max(1, cpu_threads))
    except Exception:
        pass

    try:
        import cv2  # type: ignore
        try:
            cv2.setNumThreads(max(1, cpu_threads))
        except Exception:
            pass
    except Exception:
        cv2 = None  # noqa

    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore
    import easyocr  # type: ignore

    img = Image.open(str(image_path)).convert("RGB")
    arr = np.array(img)

    # Preprocesado mínimo (sin volver loco al CPU)
    # Gris + umbral suave si cv2 está
    if cv2 is not None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
        arr = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)

    reader = easyocr.Reader(["es"], gpu=False)
    result = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join(result).strip()


# ---------------------------
# Reporte vs gold.json
# ---------------------------

def load_gold(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("gold.json debe ser un objeto JSON (dict) con keys=nombre_archivo.")
    # values dicts
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out

def compare_with_gold(preds: List[InvoiceOut], gold: Dict[str, Dict[str, Any]]) -> None:
    keys = ["fecha", "rut_emisor", "serie", "folio", "importe_total_con_iva_num"]

    total_docs = 0
    hits = {k: 0 for k in keys}
    mismatches: List[str] = []

    # Index preds por basename
    pred_map: Dict[str, InvoiceOut] = {Path(p._archivo or "").name: p for p in preds}

    for fname, g in gold.items():
        total_docs += 1
        p = pred_map.get(Path(fname).name)

        for k in keys:
            gv = g.get(k)
            pv = getattr(p, k) if p is not None else None

            # normaliza floats
            if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                ok = abs(float(gv) - float(pv)) < 0.01
            else:
                ok = (pv == gv)

            if ok:
                hits[k] += 1
            else:
                mismatches.append(f"* {fname}\n  - {k}: pred={pv!r} gold={gv!r}")

    print("\n=== REPORT ===")
    print(f"Docs con gold: {total_docs}")
    if total_docs == 0:
        return
    for k in keys:
        pct = 100.0 * hits[k] / total_docs
        print(f"{k}: {hits[k]}/{total_docs} ({pct:.1f}%)")

    if mismatches:
        print("\n--- MISMATCHES ---")
        # imprime agrupado “bonito”
        # (ya vienen en formato con * archivo)
        seen = set()
        for m in mismatches:
            head = m.split("\n", 1)[0]
            if head in seen:
                continue
            seen.add(head)
            # imprime todas las líneas de ese archivo
            bloc = [x for x in mismatches if x.startswith(head + "\n")]
            print(head)
            for b in bloc:
                print(b.split("\n", 1)[1])


# ---------------------------
# Main
# ---------------------------

def iter_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files: List[Path] = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)

def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    # Header: unión de keys
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Parser de facturas (prioridad: PDF con texto DGI).")
    ap.add_argument("path", help="Archivo o carpeta con facturas (pdf/jpg/png).")
    ap.add_argument("--json", action="store_true", help="Imprime salida JSON a stdout.")
    ap.add_argument("--csv", default=None, help="Escribe CSV al path indicado.")
    ap.add_argument("--debug", action="store_true", help="Incluye _debug por documento.")
    ap.add_argument("--no-ocr", action="store_true", help="Deshabilita OCR (recomendado si usás solo PDFs con texto).")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Threads CPU para OCR (si se usa).")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (si se puede).")

    ap.add_argument("--report", action="store_true", help="Compara contra gold.json y muestra métricas.")
    ap.add_argument("--gold", default=None, help="Path al gold.json (para --report).")

    args = ap.parse_args(argv)

    if args.low_priority:
        set_low_priority_if_possible()

    base = Path(args.path)
    files = iter_files(base)

    preds: List[InvoiceOut] = []
    rows: List[Dict[str, Any]] = []

    for f in files:
        out = InvoiceOut(_archivo=str(f))
        try:
            if f.suffix.lower() == ".pdf":
                txt = extract_pdf_text(f)
                if not pdf_has_useful_text(txt):
                    if args.no_ocr:
                        raise RuntimeError("PDF sin texto útil y OCR deshabilitado (--no-ocr).")
                    # OCR a PDF escaneado (caro). Si querés esto, mejor pedir PDF “copiable”.
                    raise RuntimeError("PDF parece escaneado (sin texto útil). Mejor conseguir PDF digital.")
                out = parse_dgi_pdf_text(txt, debug=args.debug)
                out._archivo = str(f)
                out._fuente = "pdf_text"
            else:
                if args.no_ocr:
                    raise RuntimeError("Imagen requiere OCR pero está deshabilitado (--no-ocr).")
                txt = ocr_image_easyocr(f, cpu_threads=max(1, args.cpu_threads))
                if not txt or len(txt.strip()) < 20:
                    raise RuntimeError("OCR falló: no se obtuvo texto útil.")
                out = parse_dgi_pdf_text(txt, debug=args.debug)  # reutiliza parser “por etiquetas”
                out._archivo = str(f)
                out._fuente = "image_ocr_easyocr"

        except Exception as e:
            out = InvoiceOut(_archivo=str(f), _fuente="error")
            if args.debug:
                out._debug = {"error": str(e)}

        # serie_y_folio
        if out.serie and out.folio:
            out.serie_y_folio = f"{out.serie}-{out.folio}"

        preds.append(out)
        d = asdict(out)
        if not args.debug:
            d.pop("_debug", None)
        rows.append(d)

    if args.report:
        if not args.gold:
            print("[ERROR] --report requiere --gold <path>", file=sys.stderr)
            return 2
        gold = load_gold(Path(args.gold))
        compare_with_gold(preds, gold)

    if args.csv:
        write_csv(rows, Path(args.csv))

    # default: json si se pide, sino igual imprimí json humano si no hay report/csv
    if args.json or (not args.report and not args.csv):
        print(json.dumps(rows, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
