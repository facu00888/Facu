import argparse
import json
import os
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


# -------------------------
# Helpers
# -------------------------

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def looks_like_doc_id(s: str) -> bool:
    # OCR basura típica: cosas de 1 letra, símbolos, etc.
    s = (s or "").strip()
    if len(s) <= 1:
        return False
    if sum(ch.isalnum() for ch in s) < 2:
        return False
    return True


def parse_amount_to_float(raw: str) -> Optional[float]:
    """
    Acepta:
    - 9.261,00  (Uy)
    - 9261,00
    - 9,261.00  (US)
    - 9261.00
    Y devuelve float.
    """
    if not raw:
        return None
    s = raw.strip().replace(" ", "")

    # Quitar moneda/símbolos sueltos
    s = re.sub(r"[^\d\.,\-]", "", s)

    # Si hay ambos separadores, el último suele ser el decimal
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        if last_comma > last_dot:
            # decimal = ',' miles='.'
            s = s.replace(".", "").replace(",", ".")
        else:
            # decimal = '.' miles=','
            s = s.replace(",", "")
    else:
        # Solo uno: si hay coma, la tomamos como decimal
        if "," in s and "." not in s:
            # Caso OCR raro "3,129,00" -> miles con coma y decimal con coma
            if s.count(",") == 2:
                s = s.replace(",", "", 1).replace(",", ".")
            else:
                s = s.replace(",", ".")
        # Solo punto: ya es decimal o número entero "9261.00"
        # No tocamos.

    try:
        return float(s)
    except ValueError:
        return None


def format_amount_uy(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    # Formato con 2 decimales y separador decimal coma, miles con punto
    s = f"{value:,.2f}"  # 9,261.00
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # 9.261,00
    return s


# -------------------------
# Data model
# -------------------------

@dataclass
class InvoiceFields:
    fecha: Optional[str] = None
    serie: Optional[str] = None
    folio: Optional[str] = None
    razon_social: Optional[str] = None
    nombre_fantasia: Optional[str] = None
    rut_emisor: Optional[str] = None

    # Nuevos
    is_nota_credito: bool = False
    importe_sin_iva: Optional[str] = None
    importe_sin_iva_num: Optional[float] = None
    importe_total_con_iva: Optional[str] = None
    importe_total_con_iva_num: Optional[float] = None

    @property
    def serie_y_folio(self) -> Optional[str]:
        if self.serie and self.folio:
            return f"{self.serie} {self.folio}"
        if self.folio:
            return self.folio
        return None


# -------------------------
# Parser
# -------------------------

class InvoiceParser:
    DATE_RE = re.compile(r"([0-3]?\d/[01]?\d/\d{2,4})")
    # Montos con decimales (coma o punto). Acepta miles.
    AMOUNT_RE = re.compile(r"(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))")

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False) -> None:
        self.languages = languages or ["es"]
        self.gpu = gpu
        self._easyocr_reader = None

    # ---------- Input ----------
    def extract_lines(self, path: str) -> List[str]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            lines = self._extract_from_pdf_text(path)
            # Si el PDF no trae texto útil, caer a OCR (requiere convertir a imagen: no automático acá)
            if len("".join(lines)) > 50:
                return self._clean(lines)
            return self._clean(lines)

        # Img
        return self._clean(self._extract_from_image_ocr(path))

    def _get_reader(self):
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._easyocr_reader

    def _extract_from_image_ocr(self, image_path: str) -> List[str]:
        reader = self._get_reader()
        lines: List[str] = []
        lines.extend(reader.readtext(image_path, detail=0))

        # recorte arriba-derecha (mejora en many layouts tipo tabla)
        try:
            from PIL import Image
            import numpy as np

            im = Image.open(image_path)
            w, h = im.size
            roi = im.crop((int(w * 0.45), int(h * 0.08), w, int(h * 0.42)))
            lines.extend(reader.readtext(np.array(roi), detail=0))
        except Exception:
            pass

        return lines

    def _extract_from_pdf_text(self, pdf_path: str) -> List[str]:
        # PDF con texto embebido: súper útil (ej. AGUAPURA)
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        except Exception:
            return []

    def _clean(self, lines: List[str]) -> List[str]:
        out = []
        seen = set()
        for s in lines:
            s = norm_spaces(s)
            if not s:
                continue
            # cortar basura obvia
            if not looks_like_doc_id(s):
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    # ---------- Main parse ----------
    def parse(self, lines: List[str], filename: Optional[str] = None) -> InvoiceFields:
        joined = "\n".join(lines)
        joined_norm = strip_accents(joined).lower()

        is_nc = self._detect_credit_note(joined_norm, filename)

        rut = self._extract_rut_emisor(lines, joined_norm)
        fecha = self._extract_fecha(lines, joined_norm)
        serie, folio = self._extract_serie_folio(lines, joined_norm)

        razon_social, fantasia = self._extract_names(lines, joined_norm, rut=rut)

        total_num = self._extract_total(lines, joined_norm)
        neto_num = self._extract_neto(lines, joined_norm)

        sign = -1.0 if is_nc else 1.0
        if total_num is not None:
            total_num *= sign
        if neto_num is not None:
            neto_num *= sign

        return InvoiceFields(
            fecha=fecha,
            serie=serie,
            folio=folio,
            razon_social=razon_social,
            nombre_fantasia=fantasia,
            rut_emisor=rut,
            is_nota_credito=is_nc,
            importe_sin_iva=format_amount_uy(neto_num),
            importe_sin_iva_num=neto_num,
            importe_total_con_iva=format_amount_uy(total_num),
            importe_total_con_iva_num=total_num,
        )

    # ---------- Detectors ----------
    def _detect_credit_note(self, joined_norm: str, filename: Optional[str]) -> bool:
        # IMPORTANT: "Crédito" como forma de pago NO es nota de crédito.
        # Buscamos específicamente "nota de credito"
        if "nota de credito" in joined_norm:
            return True
        if filename:
            fn = strip_accents(filename).lower()
            if "nota de credito" in fn or "nota_de_credito" in fn:
                return True
        return False

    def _extract_fecha(self, lines: List[str], joined_norm: str) -> Optional[str]:
        # Preferir cerca de "fecha" y evitar vencimiento/vto
        for i, line in enumerate(lines[:80]):
            low = strip_accents(line).lower()
            if "fecha" in low and "venc" not in low and "vto" not in low:
                m = self.DATE_RE.search(line)
                if not m and i + 1 < len(lines):
                    m = self.DATE_RE.search(lines[i + 1])
                if m:
                    return m.group(1)

        # fallback: primera fecha razonable (ojo, puede agarrar vencimiento si no hay otra)
        m = self.DATE_RE.search("\n".join(lines[:120]))
        return m.group(1) if m else None

    def _extract_rut_emisor(self, lines: List[str], joined_norm: str) -> Optional[str]:
        # 1) Labels explícitos de emisor
        label_re = re.compile(r"\b(rut|ruc)\b", re.IGNORECASE)
        deny = ("comprador", "cliente", "cedula", "receptor")

        for i, line in enumerate(lines[:120]):
            low = strip_accents(line).lower()
            if label_re.search(low) and not any(d in low for d in deny):
                d = digits_only(line)
                if len(d) in (11, 12):
                    return d
                # a veces queda en la línea siguiente
                if i + 1 < len(lines):
                    d2 = digits_only(lines[i + 1])
                    if len(d2) in (11, 12):
                        return d2

        # 2) Fallback: primeras líneas con 11/12 dígitos (emisor suele estar arriba)
        candidates: List[Tuple[int, str]] = []
        for i, line in enumerate(lines[:60]):
            d = digits_only(line)
            if len(d) in (11, 12):
                candidates.append((i, d))

        return candidates[0][1] if candidates else None

    def _extract_serie_folio(self, lines: List[str], joined_norm: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Soporta formatos:
        - 'e-Factura A 41717'
        - 'Serie/Numero A-4489'
        - 'Serie A  Número 3519972'
        - DAFER: 'e-Factura A758443' y aparte 'C 61346' (preferimos el par cerca de Hora/Fecha si existe)
        """
        candidates: List[Tuple[int, str, str, int]] = []  # (idx, serie, folio, score)

        def add(idx: int, serie: str, folio: str, score: int):
            if not serie or not folio:
                return
            serie_u = serie.strip().upper()
            folio_u = folio.strip()
            # arreglo útil: OCR a veces lee A como e/E en cuadros
            if serie_u == "E":
                serie_u = "A"
            candidates.append((idx, serie_u, folio_u, score))

        # A) Patrones con etiqueta
        for i, line in enumerate(lines[:120]):
            low = strip_accents(line).lower()

            # "Serie/Numero A-4489" o "Serie,Numero A-4489"
            m = re.search(r"(serie\s*[/,]?\s*numero)\D*([A-Z])\D*[-,/ ]\D*(\d{3,})", strip_accents(line), re.IGNORECASE)
            if m:
                add(i, m.group(2), m.group(3), 110)
                continue

            # "Serie A  Numero 3519972"
            m = re.search(r"serie\D*([A-Z])\D*(numero|nro|n°|no)\D*(\d{3,})", strip_accents(line), re.IGNORECASE)
            if m:
                add(i, m.group(1), m.group(3), 120)
                continue

            # "e-Factura A 41717" (o pegado A41717)
            if "e-factura" in low or "efactura" in low:
                m = re.search(r"(?:e-?factura)\D*([A-Z])\D*([0-9]{3,})", strip_accents(line), re.IGNORECASE)
                if m:
                    add(i, m.group(1), m.group(2), 90)

        # B) Par "C" en una línea y "61346" en la siguiente (DAFER style)
        for i in range(0, min(len(lines) - 1, 120)):
            a = lines[i].strip()
            b = lines[i + 1].strip()
            if len(a) == 1 and a.isalpha() and re.fullmatch(r"\d{3,}", b):
                ctx = " ".join(strip_accents(x).lower() for x in lines[max(0, i - 3): i + 4])
                score = 80
                if "hora" in ctx:
                    score = 115
                if "fecha" in ctx:
                    score += 5
                add(i, a, b, score)

        # C) Buscar patrón general "A-4489" cerca de palabras clave
        for i, line in enumerate(lines[:120]):
            low = strip_accents(line).lower()
            if "comprobante" in low or "serie" in low or "numero" in low:
                m = re.search(r"\b([A-Z])\s*[-,/ ]\s*(\d{3,})\b", strip_accents(line))
                if m:
                    add(i, m.group(1), m.group(2), 85)

        if not candidates:
            return None, None

        # Elegir mejor por score, y si empata, el que aparece más arriba
        candidates.sort(key=lambda t: (-t[3], t[0]))
        _, serie, folio, _score = candidates[0]
        return serie, folio

    def _extract_total(self, lines: List[str], joined_norm: str) -> Optional[float]:
        # Buscamos TOTAL evitando falsos positivos tipo "Total Kg"
        for line in reversed(lines):
            low = strip_accents(line).lower()
            if "total" in low:
                if "kg" in low or "total kg" in low:
                    continue
                # Preferir "TOTAL $" / "TOTAL UYU" / "TOTAL :" etc
                m = self.AMOUNT_RE.search(line)
                if m:
                    val = parse_amount_to_float(m.group(1))
                    if val is not None:
                        return val

        # fallback: último monto razonable del documento
        for line in reversed(lines):
            m = self.AMOUNT_RE.search(line)
            if m:
                val = parse_amount_to_float(m.group(1))
                if val is not None:
                    return val
        return None

    def _extract_neto(self, lines: List[str], joined_norm: str) -> Optional[float]:
        """
        Neto (sin IVA) se intenta desde:
        - SUB TOTAL / SUBTOTAL
        - MONTO NETO IVA (tasa básica + mínima)
        - IMPONIBLE IVA xx%
        Si no está, devolvemos None.
        """
        # 1) SUBTOTAL
        for line in lines:
            low = strip_accents(line).lower()
            if ("sub" in low and "total" in low) and "iva" not in low:
                m = self.AMOUNT_RE.search(line)
                if m:
                    v = parse_amount_to_float(m.group(1))
                    if v is not None:
                        return v

        # 2) Monto Neto IVA (sumar si hay varias tasas)
        netos = []
        for line in lines:
            low = strip_accents(line).lower()
            if "monto neto" in low and "iva" in low:
                m = self.AMOUNT_RE.search(line)
                if m:
                    v = parse_amount_to_float(m.group(1))
                    if v is not None:
                        netos.append(v)
        if netos:
            # Muchos docs traen 1 o 2 líneas (10% y 22%), sumamos
            return sum(netos)

        # 3) IMPONIBLE IVA (sumar bases)
        imponibles = []
        for line in lines:
            low = strip_accents(line).lower()
            if "imponible" in low and "iva" in low:
                m = self.AMOUNT_RE.search(line)
                if m:
                    v = parse_amount_to_float(m.group(1))
                    if v is not None:
                        imponibles.append(v)
        if imponibles:
            return sum(imponibles)

        return None

    def _extract_names(self, lines: List[str], joined_norm: str, rut: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        - razon_social: intenta encontrar una línea "empresa" (S.A., S.R.L., SAS, etc) cerca del RUT o al inicio.
        - nombre_fantasia: intenta encontrar un nombre corto tipo logo ("BIMBO", "Propios", "AGUAPURA")
        """
        org_suffix = ("s.a", "srl", "s.r.l", "sas", "s.a.", "s.r.l.", "s.r.l", "s a", "s r l")
        deny = ("rut", "ruc", "comprador", "cliente", "direccion", "domicilio", "tel", "telefono", "fecha", "venc", "moneda", "iva", "total", "@", "www")

        # score lines in first block
        best_rs = (0, None)  # score, line

        first_block = lines[:40]

        # Preferir líneas cerca del RUT si lo tenemos
        rut_idx = None
        if rut:
            for i, ln in enumerate(lines[:80]):
                if rut in digits_only(ln):
                    rut_idx = i
                    break

        def score_rs(ln: str) -> int:
            low = strip_accents(ln).lower()
            if any(d in low for d in deny):
                return 0
            letters = sum(ch.isalpha() for ch in ln)
            digits = sum(ch.isdigit() for ch in ln)
            if letters < 6 or digits > 3:
                return 0
            score = letters
            if any(suf in low for suf in org_suffix):
                score += 50
            # Todo mayúsculas suele ser nombre
            if ln.upper() == ln and letters >= 8:
                score += 10
            return score

        # 1) Buscar razon social
        search_lines = first_block
        if rut_idx is not None:
            a = max(0, rut_idx - 10)
            b = min(len(lines), rut_idx + 5)
            search_lines = lines[a:b] + first_block

        for ln in search_lines:
            sc = score_rs(ln)
            if sc > best_rs[0]:
                best_rs = (sc, ln)

        razon_social = best_rs[1]

        # 2) Nombre fantasía: línea corta, mayormente letras, sin sufijos legales
        fantasia = None
        for ln in first_block[:15]:
            low = strip_accents(ln).lower()
            if any(d in low for d in deny):
                continue
            if any(suf in low for suf in org_suffix):
                continue
            # corta y "de marca"
            if 3 <= len(ln) <= 18 and sum(ch.isalpha() for ch in ln) >= 3 and sum(ch.isdigit() for ch in ln) == 0:
                # evitar "e" o cosas raras
                if ln.strip().lower() in ("e", "efactura", "e-factura"):
                    continue
                fantasia = ln.strip()
                break

        # Heurísticas por marcas conocidas en tu set (no obliga, pero ayuda)
        compact = re.sub(r"[^a-z0-9]", "", joined_norm)
        if "bimbo" in compact:
            fantasia = fantasia or "BIMBO"
        if "aguapura" in compact:
            fantasia = fantasia or "AGUAPURA"
        if "propios" in compact:
            fantasia = fantasia or "Propios"

        return razon_social, fantasia


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extrae datos clave desde facturas (UY) por OCR y/o PDF text.")
    ap.add_argument("path", help="Ruta a la imagen/PDF")
    ap.add_argument("--json", action="store_true", help="Salida en JSON")
    ap.add_argument("--debug", action="store_true", help="Imprime líneas detectadas")
    args = ap.parse_args()

    parser = InvoiceParser(languages=["es"], gpu=False)
    lines = parser.extract_lines(args.path)

    if args.debug:
        print("=== OCR/TEXT (lineas) ===")
        for ln in lines:
            print(ln)
        print("=== FIN ===\n")

    fields = parser.parse(lines, filename=os.path.basename(args.path))
    payload = asdict(fields)
    payload["serie_y_folio"] = fields.serie_y_folio

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for k, v in payload.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
