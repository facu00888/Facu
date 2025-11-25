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

        # Best-effort: recorte arriba-derecha (mejora mucho para tu factura)
        try:
            from PIL import Image
            import numpy as np

            im = Image.open(image_path)
            w, h = im.size

            # ROI: ~45%->100% ancho, ~18%->35% alto (ajustable)
            roi = im.crop((int(w * 0.45), int(h * 0.18), w, int(h * 0.35)))
            lines.extend(self.reader.readtext(np.array(roi), detail=0))
        except Exception:
            pass

        # Limpieza básica
        out = []
        for s in lines:
            s = (s or "").strip()
            if s and s not in out:
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
        # Preferir fecha cerca de la palabra "fecha" (aunque no haya :)
        for i, line in enumerate(lines):
            ll = line.lower()
            if "fecha" in ll and "venc" not in ll and "vto" not in ll:
                m = self.DATE_RE.search(line)
                if not m and i + 1 < len(lines):
                    m = self.DATE_RE.search(lines[i + 1])
                if m:
                    return m.group(1)

        # Fallback: primera fecha del documento (siempre es medio peligroso, pero peor es nada)
        m = self.DATE_RE.search(joined)
        return m.group(1) if m else None

    def _extract_serie_folio(self, lines: List[str], joined: str) -> Optional[str]:
        # Caso típico: "A 111347 Crédito" o "Factura A 111347"
        keywords = ("serie", "folio", "factura", "crédito", "credito", "contado")
        for line in lines[:40]:
            if any(k in line.lower() for k in keywords):
                m = self.SERIE_FOLIO_RE.search(line)
                if m:
                    return f"{m.group(1)} {m.group(2)}"

        # Fallback: buscar patrón en primeras líneas (evita agarrar números random de abajo)
        for line in lines[:40]:
            m = self.SERIE_FOLIO_RE.search(line)
            if m:
                return f"{m.group(1)} {m.group(2)}"

        # Último fallback: el método viejo, por si alguna factura sí trae “Serie y Folio: …”
        m = re.search(r"serie\s*y\s*folio[:\s]*([\w-]+)", joined, re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_rut(self, lines: List[str], joined: str) -> Optional[str]:
        # Preferir líneas con "rut" pero NO "comprador"/"cliente"
        for line in lines:
            ll = line.lower()
            if "rut" in ll and "comprador" not in ll and "cliente" not in ll:
                digits = re.sub(r"\D", "", line)
                if len(digits) in (11, 12):
                    return digits

        # Fallback: buscar “RUT: 123”
        m = re.search(r"\brut\b[:\s-]*([\d .-]{8,})", joined, re.IGNORECASE)
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            return digits if len(digits) in (11, 12) else (digits or None)

        return None

    def _extract_razon_social(self, lines: List[str], joined: str, rut_emisor: Optional[str]) -> Optional[str]:
        def looks_like_name(s: str) -> bool:
            if len(s) < 4:
                return False
            low = s.lower()
            bad = ("rut", "comprador", "cliente", "fecha", "importe", "total", "dire", "tel", "fax", "@", "cfe")
            if any(b in low for b in bad):
                return False
            digits = sum(ch.isdigit() for ch in s)
            letters = sum(ch.isalpha() for ch in s)
            return letters >= 4 and digits <= 3

        # 1) Buscar cerca del RUT (líneas anteriores)
        if rut_emisor:
            rut_digits = rut_emisor
            for i, line in enumerate(lines):
                if rut_digits in re.sub(r"\D", "", line):
                    for j in range(i - 1, max(-1, i - 7), -1):
                        cand = lines[j].strip()
                        if looks_like_name(cand):
                            return cand

        # 2) Heurística por dominio (sirve mucho cuando el logo no se OCR-ea)
        compact = re.sub(r"[^a-z0-9]", "", joined.lower())
        if "distribuidoradelsur" in compact:
            return "Distribuidora del Sur"

        # 3) Fallback: primera línea “con pinta” en el arranque
        for cand in lines[:12]:
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

        # Caso típico OCR: "3,129,00" (dos comas, sin puntos) => "3.129,00"
        if s.count(",") == 2 and s.count(".") == 0:
            s = s.replace(",", ".", 1)

        return s


def parse_invoice(image_path: str, langs: Optional[List[str]] = None, gpu: bool = False) -> InvoiceFields:
    parser = InvoiceParser(languages=langs or ["es"], gpu=gpu)
    text_lines = parser.extract_text(image_path)
    return parser.parse(text_lines)


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
