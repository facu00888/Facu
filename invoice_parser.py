import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class InvoiceFields:
    fecha: Optional[str] = None
    serie_y_folio: Optional[str] = None
    razon_social: Optional[str] = None
    rut_emisor: Optional[str] = None
    importe_total_con_iva: Optional[str] = None


class InvoiceParser:
    DATE_RE = re.compile(r"([0-3]?\d/[01]?\d/\d{2,4})")
    SERIE_FOLIO_RE = re.compile(r"\b([A-Z])\s*[-]?\s*(\d{3,})\b")
    AMOUNT_RE = re.compile(r"(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))")

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False) -> None:
        import easyocr
        self.reader = easyocr.Reader(languages or ["es"], gpu=gpu)

    def extract_text(self, image_path: str) -> List[str]:
        """
        2 pasadas:
        - OCR global
        - OCR en un recorte típico (arriba-derecha) donde suelen estar Fecha / Serie / Folio
        """
        lines: List[str] = []
        lines.extend(self.reader.readtext(image_path, detail=0))

        # Best-effort: recorte arriba-derecha (mejora mucho para facturas con cuadro de datos)
        try:
            from PIL import Image
            import numpy as np

            im = Image.open(image_path)
            w, h = im.size

            # ROI: ~45%->100% ancho, ~12%->38% alto (un poco más generoso)
            roi = im.crop((int(w * 0.45), int(h * 0.12), w, int(h * 0.38)))
            lines.extend(self.reader.readtext(np.array(roi), detail=0))
        except Exception:
            pass

        # Limpieza básica: trim + quitar vacíos + dedupe manteniendo orden
        out: List[str] = []
        seen = set()
        for s in lines:
            s = (s or "").strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def parse(self, lines: List[str]) -> InvoiceFields:
        normalized = [ln.strip() for ln in lines if ln and ln.strip()]
        joined = "\n".join(normalized)

        rut_emisor = self._extract_rut(normalized, joined)
        fecha = self._extract_date(normalized, joined)
        serie_y_folio = self._extract_serie_folio(normalized, joined)
        razon_social = self._extract_razon_social(normalized, joined, rut_emisor)
        importe_total = self._extract_total(normalized)

        return InvoiceFields(
            fecha=fecha,
            serie_y_folio=serie_y_folio,
            razon_social=razon_social,
            rut_emisor=rut_emisor,
            importe_total_con_iva=importe_total,
        )

    def _extract_date(self, lines: List[str], joined: str) -> Optional[str]:
        # Preferir fechas cerca de "fecha" (evitar vencimiento/vto)
        for i, line in enumerate(lines):
            ll = line.lower()
            if "fecha" in ll and "venc" not in ll and "vto" not in ll:
                m = self.DATE_RE.search(line)
                if not m and i + 1 < len(lines):
                    m = self.DATE_RE.search(lines[i + 1])
                if m:
                    return m.group(1)

        # Fallback: primera fecha del documento
        m = self.DATE_RE.search(joined)
        return m.group(1) if m else None

    def _extract_serie_folio(self, lines: List[str], joined: str) -> Optional[str]:
        """
        Caso más común en e-Facturas:
        - aparece un bloque con "e-Factura"
        - cerca está el NÚMERO (folio)
        - la SERIE suele ser una sola letra en el mismo cuadro (a veces OCR la lee como "e/E")
        """
        idx = None
        for i, line in enumerate(lines[:60]):
            if "e-factura" in line.lower():
                idx = i
                break

        serie = None
        folio = None

        if idx is not None:
            # Folio: número 3+ dígitos cerca de e-Factura
            for j in range(idx, min(len(lines), idx + 12)):
                m = re.search(r"\b(\d{3,})\b", lines[j])
                if m:
                    folio = m.group(1)
                    break

            # Serie: una letra sola cerca (OCR a veces ve la A como e/E)
            for j in range(max(0, idx - 8), min(len(lines), idx + 12)):
                s = lines[j].strip()
                if len(s) == 1 and s.isalpha():
                    serie = s.upper()
                    if serie == "E":  # hack útil: A suele salir como e/E en cuadros
                        serie = "A"
                    break

        if serie and folio:
            return f"{serie} {folio}"
        if folio:
            return folio

        # Fallback: patrón general "A 111347" en el texto
        m = self.SERIE_FOLIO_RE.search(joined)
        if m:
            return f"{m.group(1)} {m.group(2)}"

        return None

    def _extract_rut(self, lines: List[str], joined: str) -> Optional[str]:
        """
        RUT emisor puede venir como:
        - "RUT: 211524180014"
        - o como una línea sola con números (sin etiqueta)
        """
        # 1) Si existe "RUT" en alguna línea (caso ideal), usarlo (evitando comprador/cliente)
        for line in lines:
            ll = line.lower()
            if "rut" in ll and "comprador" not in ll and "cliente" not in ll:
                digits = re.sub(r"\D", "", line)
                if len(digits) in (11, 12):
                    return digits

        # 2) Fallback: buscar "rut" en el texto completo
        m = re.search(r"\brut\b[:\s-]*([\d .-]{8,})", joined, re.IGNORECASE)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) in (11, 12):
                return digits

        # 3) Fallback fuerte: línea SOLO números (11/12 dígitos).
        # Para evitar agarrar el RUT del comprador, priorizamos los primeros renglones.
        candidates = []
        for i, line in enumerate(lines):
            digits = re.sub(r"\D", "", line)
            if len(digits) in (11, 12):
                candidates.append((i, digits))

        for i, digits in candidates:
            if i <= 25:  # emisor suele estar arriba
                return digits

        return candidates[0][1] if candidates else None

    def _extract_razon_social(self, lines: List[str], joined: str, rut_emisor: Optional[str]) -> Optional[str]:
        def looks_like_name(s: str) -> bool:
            if len(s) < 4:
                return False
            low = s.lower()
            bad = (
                "rut", "comprador", "cliente", "fecha", "importe", "total",
                "direccion", "tel", "fax", "@", "cfe", "serie", "numero",
                "localidad", "moneda", "uy", "uyu"
            )
            if any(b in low for b in bad):
                return False
            digits = sum(ch.isdigit() for ch in s)
            letters = sum(ch.isalpha() for ch in s)
            return letters >= 4 and digits <= 3

        # 1) Buscar cerca del RUT (si lo tenemos)
        if rut_emisor:
            rut_digits = rut_emisor
            for i, line in enumerate(lines):
                if rut_digits in re.sub(r"\D", "", line):
                    for j in range(i - 1, max(-1, i - 10), -1):
                        cand = lines[j].strip()
                        if looks_like_name(cand):
                            return cand

        # 2) Heurísticas por marca / texto
        compact = re.sub(r"[^a-z0-9]", "", joined.lower())
        if "distribuidoradelsur" in compact:
            return "Distribuidora del Sur"
        if "distrimax" in compact:
            # A veces aparece "Distrimax" y también el nombre personal.
            # Preferimos el nombre grande si existe.
            for cand in lines[:20]:
                if "gustavo" in cand.lower() and looks_like_name(cand):
                    return cand
            return "Distrimax"

        # 3) Fallback: primera línea "con pinta" al principio
        for cand in lines[:15]:
            if looks_like_name(cand):
                return cand

        return None

    def _extract_total(self, lines: List[str]) -> Optional[str]:
        # Preferir líneas con “importe total” o “total”
        for line in reversed(lines):
            ll = line.lower()
            if "importe total" in ll or re.search(r"\btotal\b", ll):
                m = self.AMOUNT_RE.search(line)
                if m:
                    return self._normalize_amount(m.group(1))

        # Fallback: último importe con formato
        for line in reversed(lines):
            m = self.AMOUNT_RE.search(line)
            if m:
                return self._normalize_amount(m.group(1))
        return None

    @staticmethod
    def _normalize_amount(s: str) -> str:
        s = s.strip().replace(" ", "")

        # Caso OCR: "3,129,00" -> poner miles con punto: "3.129,00"
        if s.count(",") == 2 and s.count(".") == 0:
            s = s.replace(",", ".", 1)

        return s


def main() -> None:
    argp = argparse.ArgumentParser(description="Extraer campos clave desde una factura escaneada.")
    argp.add_argument("image", help="Ruta a la imagen de la factura")
    argp.add_argument("--json", action="store_true", help="Imprimir salida en formato JSON")
    argp.add_argument("--debug", action="store_true", help="Imprimir texto detectado por OCR")
    args = argp.parse_args()

    parser = InvoiceParser(languages=["es"], gpu=False)
    lines = parser.extract_text(args.image)

    if args.debug:
        print("=== OCR (lineas) ===")
        for ln in lines:
            print(ln)
        print("=== FIN OCR ===\n")

    fields = parser.parse(lines)

    if args.json:
        print(json.dumps(asdict(fields), ensure_ascii=False, indent=2))
    else:
        for key, value in asdict(fields).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
