#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
invoice_parser.py
- Parsea e-Facturas (UY) desde PDFs (texto) e imágenes (OCR).
- Extrae: fecha, RUT emisor, razón social emisor, serie+folio (cuando es confiable), total con IVA, neto sin IVA (heurístico),
  y report contra un gold.json (opcional).

Dependencias (según modo):
- PDF: pdfplumber
- OCR: easyocr, opencv-python, numpy, pillow, torch
Opcional:
- pytesseract (si querés agregar otro backend)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# --- Opcionales / pesadas ---
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None
    np = None

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

try:
    import torch  # type: ignore
except Exception:
    torch = None


# =========================
# Helpers de texto / parse
# =========================

DATE_RE = re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b")
RUT_DIGITS_RE = re.compile(r"\b(\d[\d\s]{10,16}\d)\b")  # captura 12 dígitos con espacios en medio
SERIE_RE = re.compile(r"\bSERIE\b[^A-Z0-9]{0,10}([A-Z])\b", re.IGNORECASE)
NUMERO_RE = re.compile(r"\bNUMER[O0]\b", re.IGNORECASE)

# términos genéricos que NO son razón social
GENERIC_NAME_STOP = {
    "RUT", "RUC", "DOCUMENTO", "TIPO", "SERIE", "NUMERO", "NOMBRE", "DENOMINACION",
    "DOMICILIO", "FISCAL", "DIRECCION", "CIUDAD", "PAIS", "FECHA", "VENCIMIENTO",
    "MONEDA", "FORMA", "PAGO", "SUCURSAL", "TOTAL", "SUBTOTAL", "IVA", "CAE", "CAI",
    "CODIGO", "SEGURIDAD", "OBSERVACIONES", "RECEPTOR", "EMISOR", "COMPRADOR",
}

def to_upper_safe(s: str) -> str:
    return (s or "").upper()

def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def strip_accents_basic(s: str) -> str:
    # suficiente para comparar tokens simples (sin dependencia extra)
    repl = {
        "Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U","Ü":"U","Ñ":"N",
        "á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u","ñ":"n",
    }
    for a,b in repl.items():
        s = s.replace(a,b)
    return s

def is_likely_company_line(line: str) -> bool:
    raw = line.strip()
    if len(raw) < 3:
        return False
    up = strip_accents_basic(raw.upper())
    # si es puro número / símbolos, fuera
    if not re.search(r"[A-Z]", up):
        return False
    # demasiadas palabras genéricas
    tokens = [t for t in re.split(r"[^A-Z0-9]+", up) if t]
    if not tokens:
        return False
    stop_hits = sum(1 for t in tokens if t in GENERIC_NAME_STOP)
    if stop_hits / max(1, len(tokens)) > 0.5:
        return False
    # evitar líneas con demasiados dígitos (direcciones, ruts, etc.)
    digit_ratio = sum(c.isdigit() for c in raw) / max(1, len(raw))
    if digit_ratio > 0.35:
        return False
    # suele ser empresa si contiene estos sufijos
    if any(x in up for x in [" SRL", " S.R.L", " S.A", " SA ", " SAS", " LTDA", " LLC"]):
        return True
    # o si casi todo es mayúscula (OCR)
    alpha = [c for c in up if c.isalpha()]
    if alpha:
        upper_ratio = sum(c.isupper() for c in alpha) / len(alpha)
        if upper_ratio > 0.75 and len(raw) <= 70:
            return True
    return False

def _amount_str_to_float(s: str) -> Optional[float]:
    """
    Convierte strings tipo:
    - "4.018,01" -> 4018.01
    - "4 018,01" -> 4018.01
    - "4018,01"  -> 4018.01
    - "6491.78"  -> 6491.78 (si no hay coma)
    - OCR raro: "4 018,0l" -> 4018.01 (corrige I/l->1, O->0)
    """
    if not s:
        return None

    s = s.strip()
    s = s.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    s = re.sub(r"[^\d,.\-\s]", "", s).strip()
    if not re.search(r"\d", s):
        return None

    neg = s.startswith("-")
    s = s[1:] if neg else s

    # quitar espacios internos
    s = re.sub(r"\s+", "", s)

    # si tiene coma, asumimos decimal coma (UY)
    if "," in s:
        # todo lo demás separador de miles
        s = s.replace(".", "")
        parts = s.split(",")
        if len(parts) > 2:
            # demasiadas comas, probablemente OCR basura
            return None
        int_part = parts[0] if parts[0] else "0"
        dec_part = parts[1] if len(parts) == 2 else ""
        if dec_part and len(dec_part) > 2:
            dec_part = dec_part[:2]
        if dec_part and len(dec_part) < 2:
            # "...,5" -> "...,50"
            dec_part = dec_part.ljust(2, "0")
        norm = int_part + (".{}".format(dec_part) if dec_part else "")
        try:
            v = float(norm)
        except Exception:
            return None
        return -v if neg else v

    # sin coma: puede ser decimal con punto o entero con miles
    # si hay un punto y tiene 2 decimales al final, lo tratamos como decimal
    if "." in s:
        last = s.rsplit(".", 1)[-1]
        if len(last) == 2 and last.isdigit():
            try:
                v = float(s)
            except Exception:
                return None
            return -v if neg else v
        # si no, lo tomamos como separador de miles (y sin decimales)
        s2 = s.replace(".", "")
        if not s2.isdigit():
            return None
        try:
            v = float(s2)
        except Exception:
            return None
        return -v if neg else v

    # solo dígitos
    if not s.isdigit():
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return -v if neg else v

def format_uy_amount(v: float) -> str:
    # 4018.01 -> "4.018,01"
    try:
        v = float(v)
    except Exception:
        return ""
    sign = "-" if v < 0 else ""
    v = abs(v)
    int_part = int(v)
    dec_part = int(round((v - int_part) * 100))
    # corregir rounding que empuja dec a 100
    if dec_part >= 100:
        int_part += 1
        dec_part = 0
    int_str = f"{int_part:,}".replace(",", ".")
    return f"{sign}{int_str},{dec_part:02d}"

def parse_amount_candidates(text_up: str) -> List[Tuple[str, float, int, int]]:
    """
    Devuelve candidatos de importes encontrados en el texto (sin contexto).
    Formato: (raw, value, start, end)
    """
    cands: List[Tuple[str, float, int, int]] = []
    # patrones comunes (incluye espacios como separador de miles por OCR)
    pat = re.compile(r"(?<!\d)(-?\d{1,3}(?:[.\s]\d{3})*(?:[,\.]\d{2})?)(?!\d)")
    for m in pat.finditer(text_up):
        raw = m.group(1)
        val = _amount_str_to_float(raw)
        if val is None:
            continue
        cands.append((raw, float(val), m.start(), m.end()))
    return cands

def score_amount_candidate(text_up: str, raw: str, val: float, a: int, b: int) -> float:
    """
    Score por contexto para elegir TOTAL.
    """
    ctx = text_up[max(0, a - 60): min(len(text_up), b + 60)]
    score = 0.0

    if val < 0:
        score -= 0.5
    if val < 50:
        score -= 2.0
    if val > 100_000_000:
        score -= 2.0

    if "TOTAL" in ctx and "SUBTOTAL" not in ctx:
        score += 2.5
    if "A PAGAR" in ctx or "TOTAL A PAGAR" in ctx:
        score += 2.5
    if "SUBTOTAL" in ctx:
        score -= 1.5
    if "IVA" in ctx and "TOTAL" not in ctx:
        score -= 0.2  # puede ser total, pero preferimos explícito TOTAL

    # penalizar si parece cantidad de línea (unitario)
    if "UNIT" in ctx or "UNITARIO" in ctx:
        score -= 0.6

    # preferir montos típicos de totales por tamaño relativo
    if 200 <= val <= 5_000_000:
        score += 0.4

    return score

def pick_total_by_lines(text: str) -> Optional[Tuple[str, float]]:
    """
    Intenta encontrar el TOTAL mirando línea por línea:
    - Busca líneas con 'TOTAL' (pero no 'SUBTOTAL')
    - Mira esa línea y la siguiente
    - Reconstruye importes tipo '4 018,01', '4.018,01', '4 018, 01'
    """
    orig = normalize_ws(text)
    lines_orig = orig.splitlines()
    lines_up = to_upper_safe(orig).splitlines()

    best_val: Optional[float] = None
    best_raw: Optional[str] = None

    for i, line_up in enumerate(lines_up):
        if "TOTAL" not in line_up:
            continue
        if "SUBTOTAL" in line_up:
            continue

        win_orig = lines_orig[i] if i < len(lines_orig) else ""
        # ventana: línea TOTAL + siguiente
        if i + 1 < len(lines_orig):
            win_orig += " " + lines_orig[i + 1]

        # OCR típicos
        tmp = win_orig.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
        tmp = re.sub(r"[^\d,.\s\-]", " ", tmp)
        tokens = [tok for tok in tmp.split() if any(c.isdigit() for c in tok)]

        candidates = set(tokens)

        # unir tokens contiguos por si el separador de miles quedó como espacio: "4" "018,01"
        for j in range(len(tokens) - 1):
            candidates.add(tokens[j] + tokens[j + 1])

        for raw in candidates:
            v = _amount_str_to_float(raw)
            if v is None:
                continue
            if v < 50:
                continue
            if best_val is None or v > best_val:
                best_val = v
                best_raw = raw

    if best_val is None or best_raw is None:
        return None

    return (format_uy_amount(best_val), round(best_val, 2))

def pick_best_total(text: str) -> Optional[Tuple[str, float, str]]:
    """
    Return (formatted_str, float_value, source_tag).
    """
    lt = pick_total_by_lines(text)
    if lt is not None:
        raw_fmt, val = lt
        return raw_fmt, val, "line_total"

    t_up = to_upper_safe(text)
    cands = parse_amount_candidates(t_up)
    if not cands:
        return None

    scored: List[Tuple[float, str, float]] = []
    for raw, val, a, b in cands:
        sc = score_amount_candidate(t_up, raw, val, a, b)
        scored.append((sc, raw, val))

    scored.sort(key=lambda x: (x[0], x[2]), reverse=True)
    best_sc, best_raw, best_val = scored[0]
    if best_sc >= 1.0:
        return (format_uy_amount(best_val), round(best_val, 2), "scored_context")

    plausible = [(raw, val) for sc, raw, val in scored if 50 <= val <= 10_000_000]
    if plausible:
        raw, val = max(plausible, key=lambda t: t[1])
        return (format_uy_amount(val), round(val, 2), "max_plausible")

    return None

def parse_dates(text: str) -> List[Tuple[str, str, float]]:
    """
    Devuelve lista de fechas candidatas:
    (fecha_str_dd/mm/yyyy, fuente, score)
    """
    up = to_upper_safe(text)
    out: List[Tuple[str, str, float]] = []
    for m in DATE_RE.finditer(up):
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        # validar básico
        try:
            d = int(dd); mo = int(mm); y = int(yyyy)
            if not (1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2100):
                continue
        except Exception:
            continue
        s = f"{int(dd):02d}/{int(mm):02d}/{yyyy}"

        a, b = m.start(), m.end()
        ctx = up[max(0, a - 50): min(len(up), b + 50)]

        score = 0.0
        # preferir "FECHA" / "FECHA DE DOCUMENTO"
        if "FECHA" in ctx:
            score += 2.0
        if "DOCUMENTO" in ctx:
            score += 0.8
        # penalizar vencimiento
        if "VENC" in ctx or "VTO" in ctx:
            score -= 1.5
        # penalizar CAE/CAI vencimiento
        if "CAE" in ctx or "CAI" in ctx:
            score -= 0.5
        # si hay varias, se decide por score y por estar más arriba (se aplica afuera si se pasa pos)
        out.append((s, "regex_date", score))
    return out

def pick_best_date(text: str) -> Optional[str]:
    up = to_upper_safe(text)
    cands = []
    for m in DATE_RE.finditer(up):
        s = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
        a, b = m.start(), m.end()
        ctx = up[max(0, a - 60): min(len(up), b + 60)]
        score = 0.0
        if "FECHA" in ctx:
            score += 2.0
        if "FECHA DE DOCUMENTO" in ctx:
            score += 2.0
        if "DOCUMENTO" in ctx:
            score += 0.7
        if "VENC" in ctx or "VTO" in ctx:
            score -= 1.7
        if "CAE" in ctx or "CAI" in ctx:
            score -= 0.7

        # preferir fechas dentro de los primeros ~1/3 del texto (factura suele)
        pos_ratio = a / max(1, len(up))
        score += max(0.0, 0.8 - pos_ratio)  # cuanto más arriba, más score

        cands.append((score, s))

    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def _clean_rut_digits(raw: str) -> Optional[str]:
    d = re.sub(r"\D", "", raw or "")
    if len(d) == 12:
        return d
    return None

def extract_ruts(text: str) -> List[Tuple[str, int]]:
    """
    Devuelve lista de (rut_12dig, start_pos)
    """
    up = to_upper_safe(text)
    out: List[Tuple[str, int]] = []
    for m in RUT_DIGITS_RE.finditer(up):
        rut = _clean_rut_digits(m.group(1))
        if rut:
            out.append((rut, m.start()))
    # dedup por orden
    seen = set()
    uniq = []
    for rut, pos in out:
        if rut in seen:
            continue
        seen.add(rut)
        uniq.append((rut, pos))
    return uniq

def pick_rut_emisor(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (rut_emisor, rut_receptor) si se puede.
    Heurística:
    - Rut receptor: cerca de "RUT RECEPTOR" o "RUC COMPRADOR"
    - Rut emisor: cerca de "RUT EMISOR" o "RUC" (y penaliza receptor)
    """
    up = to_upper_safe(text)
    ruts = extract_ruts(text)

    if not ruts:
        return None, None

    def score_rut(rut: str, pos: int) -> float:
        ctx = up[max(0, pos - 70): min(len(up), pos + 70)]
        s = 0.0
        if "RUT EMISOR" in ctx or "RUC" in ctx:
            s += 2.0
        if "RUT RECEPTOR" in ctx or "RUC COMPRADOR" in ctx or "COMPRADOR" in ctx:
            s -= 2.5
        if "RECEPTOR" in ctx:
            s -= 1.2
        if "EMISOR" in ctx:
            s += 1.2
        # preferir top
        pos_ratio = pos / max(1, len(up))
        s += max(0.0, 0.8 - pos_ratio)
        return s

    # receptor
    receptor = None
    best_r_sc = -1e9
    for rut, pos in ruts:
        ctx = up[max(0, pos - 70): min(len(up), pos + 70)]
        sc = 0.0
        if "RUT RECEPTOR" in ctx or "RUC COMPRADOR" in ctx or "COMPRADOR" in ctx:
            sc += 3.0
        sc += max(0.0, 0.6 - (pos / max(1, len(up))))
        if sc > best_r_sc:
            best_r_sc = sc
            receptor = rut if sc >= 1.5 else receptor

    # emisor
    emisor = None
    best_e_sc = -1e9
    for rut, pos in ruts:
        sc = score_rut(rut, pos)
        # si es igual al receptor, penalizar fuerte
        if receptor and rut == receptor:
            sc -= 5.0
        if sc > best_e_sc:
            best_e_sc = sc
            emisor = rut

    # si solo había uno, asumimos emisor
    if emisor and receptor and emisor == receptor:
        receptor = None

    return emisor, receptor

def guess_razon_social(text: str, rut_emisor: Optional[str], rut_receptor: Optional[str]) -> Optional[str]:
    """
    Mejora: intenta asociar razón social al emisor, evitando devolverte tu propia razón social (receptor)
    si aparece repetida.
    """
    orig = normalize_ws(text)
    up = to_upper_safe(orig)

    lines = [ln.strip() for ln in orig.splitlines() if ln.strip()]
    if not lines:
        return None

    # 1) si podemos, buscar un bloque cerca del rut emisor
    idx_em = up.find(rut_emisor) if rut_emisor else -1
    idx_rc = up.find(rut_receptor) if rut_receptor else -1

    candidates: List[Tuple[float, str]] = []

    def add_candidate(score: float, name: str):
        name = name.strip()
        if not name:
            return
        upn = strip_accents_basic(name.upper())
        # eliminar genéricos puros
        toks = [t for t in re.split(r"[^A-Z0-9]+", upn) if t]
        if not toks:
            return
        if all(t in GENERIC_NAME_STOP for t in toks):
            return
        # limpiar cosas raras
        name = re.sub(r"\s{2,}", " ", name).strip()
        if len(name) < 3:
            return
        candidates.append((score, name))

    # 1a) línea anterior / siguiente al emisor
    if idx_em >= 0:
        # ubicar línea aproximada
        upto = up[:idx_em]
        line_no = upto.count("\n")
        for k in [line_no - 2, line_no - 1, line_no, line_no + 1, line_no + 2]:
            if 0 <= k < len(lines):
                ln = lines[k]
                if is_likely_company_line(ln):
                    add_candidate(5.0 - abs(k - line_no) * 0.3, ln)

    # 1b) para PDFs, a veces viene tipo: "PURA PALTA SAS RUT RECEPTOR ..."
    # elegir token antes de "RUT RECEPTOR" si está
    m = re.search(r"([A-Z0-9 .&\-]{3,80})\s+RUT\s+RECEPTOR", up)
    if m:
        snippet = m.group(1).strip()
        if snippet and is_likely_company_line(snippet):
            add_candidate(4.5, snippet)

    # 2) top lines: buscar nombres probables en primeras 12 líneas
    for i, ln in enumerate(lines[:12]):
        if is_likely_company_line(ln):
            sc = 3.5 - (i * 0.1)
            # penalizar si está muy cerca del receptor rut (posible comprador)
            if rut_receptor and rut_receptor in up:
                # si el nombre aparece cerca del receptor, penaliza
                pos_n = up.find(strip_accents_basic(ln.upper()))
                if pos_n >= 0 and abs(pos_n - idx_rc) < 180:
                    sc -= 2.0
            add_candidate(sc, ln)

    # 3) evitar devolver basura tipo "DENOMINACION"
    cleaned: List[Tuple[float, str]] = []
    for sc, nm in candidates:
        upn = strip_accents_basic(nm.upper())
        if upn.strip() in GENERIC_NAME_STOP:
            continue
        # penalizar si contiene demasiadas palabras "modelo natural" etc. (receptor típico)
        # sin hardcode del nombre: penalizamos si cerca del rut_receptor
        if rut_receptor and rut_receptor in up:
            idxn = up.find(strip_accents_basic(nm.upper()))
            if idxn >= 0 and idx_rc >= 0 and abs(idxn - idx_rc) < 200:
                sc -= 1.5
        cleaned.append((sc, nm))

    if not cleaned:
        return None

    cleaned.sort(key=lambda x: x[0], reverse=True)
    return cleaned[0][1]

def is_nota_credito(text: str) -> bool:
    up = to_upper_safe(text)
    if re.search(r"\bNOTA\s+DE\s+CREDITO\b", up):
        return True
    # "N/C" o "NC" puede confundir con "credito" (forma de pago), lo dejamos conservador
    return False

def parse_serie_folio(text: str, is_pdf: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrae SERIE y NUMERO/FOLIO.
    - Para PDFs somos más permisivos.
    - Para imágenes: ultra conservador (solo cuando aparece claramente SERIE/NUMERO).
    """
    t = to_upper_safe(text)

    serie: Optional[str] = None
    folio: Optional[str] = None

    # SERIE <letra>
    m = re.search(r"\bSERIE\b[^A-Z0-9]{0,10}([A-Z])\b", t)
    if m:
        serie = m.group(1)

    # SERIE ... NUMERO ... <letra> <numero>
    m2 = re.search(
        r"\bSERIE\b.*?\bNUMER[O0]\b.*?\b([A-Z])\s+0?(\d{4,9})\b",
        t,
        flags=re.DOTALL,
    )
    if m2:
        if serie is None:
            serie = m2.group(1)
        if folio is None:
            f = m2.group(2)
            folio = f.lstrip("0") or f

    if is_pdf:
        # NUMERO <digits>
        m3 = re.search(r"\bNUMER[O0]\b\s*[:#]?\s*(\d{4,9})\b", t)
        if m3 and folio is None:
            f = m3.group(1)
            folio = f.lstrip("0") or f

        # A-049009 / A 049009 cerca de DOCUMENTO/SERIE/NUMERO
        m4 = re.search(r"\b([A-Z])\s*[-]?\s*0?(\d{4,9})\b", t)
        if m4:
            idx = m4.start()
            ctx = t[max(0, idx - 40): idx + 40]
            if ("SERIE" in ctx) or ("NUMER" in ctx) or ("DOCUMENTO" in ctx):
                if serie is None:
                    serie = m4.group(1)
                if folio is None:
                    f = m4.group(2)
                    folio = f.lstrip("0") or f

    # validaciones
    if folio is not None and (len(folio) < 4 or len(folio) > 9):
        folio = None

    return serie, folio

def compute_importe_sin_iva(text: str, total_val: Optional[float]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Heurística para neto (sin IVA):
    - Primero: intenta encontrar "SUBTOTAL" que parezca el neto.
    - Si no: si detecta "22" en contexto IVA, usa total/1.22
      si detecta "10" (minima), usa total/1.10
    """
    if total_val is None:
        return None, None, None

    up = to_upper_safe(text)

    # 1) intentar subtotal explícito (solo si está cerca de "SUBTOTAL" y no está duplicado raro)
    # ejemplo: "Subtotal ... 6.981,14"
    # (en PDFs a veces falla el texto; igual lo intentamos)
    m = re.search(r"\bSUBTOTAL\b[^0-9]{0,20}([0-9][0-9\.\s]*[,\.][0-9]{2})", up)
    if m:
        v = _amount_str_to_float(m.group(1))
        if v is not None and 50 <= v <= total_val:
            return format_uy_amount(v), round(v, 2), "subtotal_label"

    # 2) decidir tasa probable
    # Buscar señales de 22% (básica) o 10% (mínima)
    has22 = bool(re.search(r"\b22\b|\bBASICA\b|\bTASA\s+BASICA\b", up))
    has10 = bool(re.search(r"\b10\b|\bMINIMA\b|\bTASA\s+MINIMA\b", up))

    # preferimos 22 si aparece (la mayoría de las cosas en UY)
    if has22 and not has10:
        net = total_val / 1.22
        return format_uy_amount(net), round(net, 2), "total_div_22"
    if has10 and not has22:
        net = total_val / 1.10
        return format_uy_amount(net), round(net, 2), "total_div_10"

    # si aparecen ambas o ninguna, elegimos 22 por default pero marcamos heurístico
    net = total_val / 1.22
    return format_uy_amount(net), round(net, 2), "total_div_22"


# =========================
# OCR / PDF lectura
# =========================

_EASYOCR_READER = None

def configure_threads(cpu_threads: int):
    # limitar threads de libs nativas (mejora para que no te congele la PC)
    cpu_threads = max(1, int(cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)

    if torch is not None:
        try:
            torch.set_num_threads(cpu_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

def set_low_priority_windows():
    # baja prioridad (Windows). Si falla, no pasa nada.
    try:
        import ctypes
        import ctypes.wintypes

        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        IDLE_PRIORITY_CLASS = 0x00000040

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

def init_easyocr(lang: str = "es", gpu: bool = False):
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    if easyocr is None:
        raise RuntimeError("easyocr no está instalado.")
    _EASYOCR_READER = easyocr.Reader([lang], gpu=gpu, verbose=False)
    return _EASYOCR_READER

def limit_image_size(img_bgr, max_dim: int, max_pixels: int):
    h, w = img_bgr.shape[:2]
    scale = 1.0

    if max(h, w) > max_dim:
        scale = min(scale, max_dim / float(max(h, w)))

    if h * w > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(h * w)))

    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr

def preprocess_for_ocr(img_bgr, ocr_mode: str):
    """
    Preproceso simple y rápido:
    - gray
    - denoise leve
    - threshold (en balanced)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if ocr_mode in ("balanced", "accurate"):
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    else:
        # fast: sin threshold fuerte
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

def crop_rois(img, mode: str = "header_totals"):
    """
    Para reducir errores y recursos:
    - OCR solo header (arriba) + totals (abajo)
    """
    h, w = img.shape[:2]
    if mode == "header_totals":
        top = img[0:int(h * 0.45), :]
        bottom = img[int(h * 0.62):, :]
        return top, bottom
    # fallback: full
    return img, img

def ocr_easyocr_image(img_bgr, reader, ocr_mode: str, debug: bool = False) -> str:
    """
    Corre OCR con easyocr sobre imagen ya preprocesada (numpy array uint8).
    """
    # easyocr espera RGB o gray, pero acepta numpy.
    # detail=0 -> devuelve lista de strings
    kwargs = {}
    # workers en easyocr ayuda, pero queremos control de recursos con cpu_threads (ya seteamos torch/OMP)
    # canvas_size y mag_ratio pueden cambiar mucho el tiempo: lo dejamos moderado.
    if ocr_mode == "fast":
        kwargs.update(dict(detail=0, paragraph=False))
    else:
        kwargs.update(dict(detail=0, paragraph=True))

    try:
        out = reader.readtext(img_bgr, **kwargs)
    except TypeError:
        # compat por versiones: si paragraph no existe
        kwargs.pop("paragraph", None)
        out = reader.readtext(img_bgr, **kwargs)

    if not out:
        return ""
    if isinstance(out, list):
        # out = ["line1", "line2", ...]
        return "\n".join([str(x).strip() for x in out if str(x).strip()])
    return str(out).strip()

def read_pdf_text(path: str) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber no está instalado, no puedo leer PDFs por texto.")
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return normalize_ws("\n".join(text_parts))

def read_image_ocr(path: str, reader, ocr_mode: str, max_dim: int, max_pixels: int, debug: bool) -> Tuple[str, str, str]:
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python/numpy no está instalado.")
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("No se pudo leer la imagen (cv2.imread devolvió None).")

    img = limit_image_size(img, max_dim=max_dim, max_pixels=max_pixels)

    top_roi, bottom_roi = crop_rois(img, mode="header_totals")

    top_p = preprocess_for_ocr(top_roi, ocr_mode)
    bot_p = preprocess_for_ocr(bottom_roi, ocr_mode)

    header_txt = ocr_easyocr_image(top_p, reader, ocr_mode=ocr_mode, debug=debug)
    totals_txt = ocr_easyocr_image(bot_p, reader, ocr_mode=ocr_mode, debug=debug)

    combined = normalize_ws(header_txt + "\n\n" + totals_txt).strip()
    return combined, header_txt.strip(), totals_txt.strip()


# =========================
# Modelo de salida
# =========================

@dataclass
class InvoiceParsed:
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

def build_invoice_from_text(text: str, source: str, file_path: str) -> InvoiceParsed:
    text_n = normalize_ws(text)
    up = to_upper_safe(text_n)

    rut_emisor, rut_receptor = pick_rut_emisor(text_n)
    razon = guess_razon_social(text_n, rut_emisor=rut_emisor, rut_receptor=rut_receptor)

    fecha = pick_best_date(text_n)

    is_pdf = (source == "pdf_text")
    serie, folio = parse_serie_folio(text_n, is_pdf=is_pdf)

    # total
    total = pick_best_total(text_n)
    if total:
        total_fmt, total_val, total_src = total
    else:
        total_fmt, total_val, total_src = None, None, None

    # neto
    net_fmt, net_val, net_src = compute_importe_sin_iva(text_n, total_val)

    syf = None
    if serie and folio:
        syf = f"{serie}-{folio}"

    return InvoiceParsed(
        fecha=fecha,
        serie=serie,
        folio=folio,
        serie_y_folio=syf,
        razon_social=razon,
        rut_emisor=rut_emisor,
        es_nota_de_credito=is_nota_credito(text_n),
        importe_total_con_iva=total_fmt,
        importe_total_con_iva_num=total_val,
        importe_sin_iva=net_fmt,
        importe_sin_iva_num=net_val,
        importe_sin_iva_fuente=net_src,
        _archivo=file_path,
        _fuente=source,
    )


# =========================
# Report / gold
# =========================

def load_gold(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("gold.json debe ser un objeto JSON (dict).")
    return data

def compare_field(pred: Any, gold: Any) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    # números: tolerancia de 0.01
    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        return abs(float(pred) - float(gold)) <= 0.01
    # strings: comparar normalizadas
    if isinstance(pred, str) and isinstance(gold, str):
        return pred.strip() == gold.strip()
    return pred == gold

def do_report(parsed: List[InvoiceParsed], gold: Dict[str, Dict[str, Any]]) -> None:
    # clave: nombre base del archivo
    keyed = {os.path.basename(p._archivo): p for p in parsed}

    docs_with_gold = 0
    fields = ["fecha", "rut_emisor", "serie", "folio", "importe_total_con_iva_num"]
    ok_counts = {f: 0 for f in fields}
    total_counts = {f: 0 for f in fields}

    per_doc_lines = []

    for k, g in gold.items():
        if k not in keyed:
            continue
        docs_with_gold += 1
        p = keyed[k]
        doc_ok = True
        mism = []

        for f in fields:
            total_counts[f] += 1
            pv = getattr(p, f)
            gv = g.get(f)
            if compare_field(pv, gv):
                ok_counts[f] += 1
            else:
                doc_ok = False
                mism.append((f, pv, gv))

        if not doc_ok:
            per_doc_lines.append((k, mism))

    print("\n=== REPORT ===")
    print(f"Docs con gold: {docs_with_gold}")
    if docs_with_gold == 0:
        print("Tip: las keys del gold.json deben coincidir EXACTO con el nombre del archivo (basename) dentro de la carpeta de entrada.")
        return

    for f in fields:
        tot = total_counts[f]
        ok = ok_counts[f]
        pct = (ok / tot * 100.0) if tot else 0.0
        print(f"{f}: {ok}/{tot} ({pct:.1f}%)")

    if per_doc_lines:
        print("\n--- MISMATCHES ---")
        for fname, mism in per_doc_lines[:40]:
            print(f"* {fname}")
            for f, pv, gv in mism:
                print(f"  - {f}: pred={pv!r} gold={gv!r}")


# =========================
# CLI / Main
# =========================

def iter_input_files(input_path: str) -> List[str]:
    p = input_path
    files: List[str] = []
    if os.path.isdir(p):
        for root, _, fnames in os.walk(p):
            for fn in fnames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    files.append(os.path.join(root, fn))
    else:
        files.append(p)
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser(description="Parse de facturas (UY): PDF text + OCR imágenes.")
    ap.add_argument("input", help="Carpeta o archivo de entrada")
    ap.add_argument("--json", action="store_true", help="Imprime JSON (lista de resultados)")
    ap.add_argument("--debug", action="store_true", help="Log detallado por archivo")
    ap.add_argument("--report", action="store_true", help="Genera reporte comparando con gold")
    ap.add_argument("--gold", type=str, default=None, help="Ruta a gold.json (para --report)")

    ap.add_argument("--ocr-mode", type=str, default="balanced", choices=["fast", "balanced", "accurate"],
                    help="Modo OCR: fast/balanced/accurate (accurate es más lento).")
    ap.add_argument("--cpu-threads", type=int, default=1, help="Límite de threads CPU (recomendado 1-2 para no matar la PC).")
    ap.add_argument("--max-dim", type=int, default=1800, help="Máximo ancho/alto de imagen antes de OCR (resize).")
    ap.add_argument("--max-pixels", type=int, default=2_500_000, help="Máximo de píxeles antes de OCR (resize).")
    ap.add_argument("--low-priority", action="store_true", help="Baja prioridad del proceso (Windows).")

    args = ap.parse_args()

    configure_threads(args.cpu_threads)
    if args.low_priority:
        set_low_priority_windows()

    files = iter_input_files(args.input)
    if not files:
        print("No encontré archivos para procesar.", file=sys.stderr)
        sys.exit(1)

    # init OCR lazy
    reader = None
    out: List[InvoiceParsed] = []

    for path in files:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                txt = read_pdf_text(path)
                if args.debug:
                    print(f"\n=== {os.path.basename(path)} (pdf_text) ===")
                    print(txt[:6000] + ("\n...[truncado]..." if len(txt) > 6000 else ""))
                    print("=== FIN ===\n")
                inv = build_invoice_from_text(txt, source="pdf_text", file_path=path)
                out.append(inv)
            else:
                if easyocr is None:
                    raise RuntimeError("No hay backend OCR disponible. Instalá easyocr (+torch).")
                if cv2 is None or np is None:
                    raise RuntimeError("No hay backend de imagen: instalá opencv-python y numpy.")

                if reader is None:
                    reader = init_easyocr(lang="es", gpu=False)

                combined, header_txt, totals_txt = read_image_ocr(
                    path, reader=reader, ocr_mode=args.ocr_mode,
                    max_dim=args.max_dim, max_pixels=args.max_pixels, debug=args.debug
                )

                if args.debug:
                    print(f"\n=== {os.path.basename(path)} (image_ocr_easyocr) ===\n")
                    print("---[OCR HEADER]---")
                    print(header_txt if header_txt else "[vacio]")
                    print("\n---[OCR TOTALS]---")
                    print(totals_txt if totals_txt else "[vacio]")
                    print("\n=== FIN ===\n")

                if not combined or len(combined.strip()) < 20:
                    raise RuntimeError("OCR falló: no se obtuvo texto útil.")

                inv = build_invoice_from_text(combined, source="image_ocr_easyocr", file_path=path)
                out.append(inv)

        except Exception as e:
            # error por documento: igual devolver entrada con nulls (para que no se pierda el archivo)
            if args.debug:
                print(f"[ERROR] {os.path.basename(path)}: {e}")
            out.append(InvoiceParsed(
                fecha=None,
                serie=None,
                folio=None,
                serie_y_folio=None,
                razon_social=None,
                rut_emisor=None,
                es_nota_de_credito=False,
                importe_total_con_iva=None,
                importe_total_con_iva_num=None,
                importe_sin_iva=None,
                importe_sin_iva_num=None,
                importe_sin_iva_fuente=None,
                _archivo=path,
                _fuente="error"
            ))

    # Report
    if args.report:
        if not args.gold:
            print("Para --report necesitás --gold RUTA_A_gold.json", file=sys.stderr)
            sys.exit(2)
        gold = load_gold(args.gold)
        do_report(out, gold)
        return

    # JSON output
    if args.json or not args.debug:
        print(json.dumps([asdict(x) for x in out], ensure_ascii=False, indent=2))
    else:
        # modo debug sin json: muestra resumen
        for x in out:
            print(asdict(x))

if __name__ == "__main__":
    main()
