import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import easyocr


@dataclass
class InvoiceFields:
    fecha: Optional[str] = None
    serie_y_folio: Optional[str] = None
    razon_social: Optional[str] = None
    rut_emisor: Optional[str] = None
    importe_total_con_iva: Optional[str] = None


class InvoiceParser:
    def __init__(self, languages: Optional[List[str]] = None) -> None:
        self.reader = easyocr.Reader(languages or ["es"], gpu=False)

    def extract_text(self, image_path: str) -> List[str]:
        return self.reader.readtext(image_path, detail=0)

    def parse(self, lines: List[str]) -> InvoiceFields:
        normalized_lines = [line.strip() for line in lines if line.strip()]
        joined = " \n".join(normalized_lines).lower()

        fecha = self._extract_date(joined)
        serie_y_folio = self._extract_serie_folio(joined)
        razon_social = self._extract_after_label(normalized_lines, "razon social")
        rut_emisor = self._extract_rut(joined)
        importe_total = self._extract_total(normalized_lines)

        return InvoiceFields(
            fecha=fecha,
            serie_y_folio=serie_y_folio,
            razon_social=razon_social,
            rut_emisor=rut_emisor,
            importe_total_con_iva=importe_total,
        )

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        match = re.search(r"fecha[:\s]*([0-3]?\d/[01]?\d/\d{2,4})", text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_serie_folio(text: str) -> Optional[str]:
        match = re.search(r"serie\s*y\s*folio[:\s]*([\w-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_after_label(lines: List[str], label: str) -> Optional[str]:
        label = label.lower()
        for line in lines:
            lower_line = line.lower()
            if label in lower_line:
                parts = re.split(r"[:\-]", line, maxsplit=1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    return value if value else None
        return None

    @staticmethod
    def _extract_rut(text: str) -> Optional[str]:
        match = re.search(r"rut[:\s]*([\d .-]{5,})", text, re.IGNORECASE)
        if match:
            rut = re.sub(r"[^\d-]", "", match.group(1))
            return rut if rut else None
        return None

    @staticmethod
    def _extract_total(lines: List[str]) -> Optional[str]:
        amount_pattern = re.compile(r"(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))")
        for line in reversed(lines):
            lower_line = line.lower()
            if "importe total" in lower_line or "total" in lower_line:
                amount_match = amount_pattern.search(line)
                if amount_match:
                    return amount_match.group(1)
        # Fallback: search anywhere
        for line in reversed(lines):
            amount_match = amount_pattern.search(line)
            if amount_match:
                return amount_match.group(1)
        return None


def parse_invoice(image_path: str) -> InvoiceFields:
    parser = InvoiceParser()
    text_lines = parser.extract_text(image_path)
    return parser.parse(text_lines)


def main() -> None:
    argp = argparse.ArgumentParser(description="Extraer campos clave desde una factura escaneada.")
    argp.add_argument("image", help="Ruta a la imagen de la factura")
    argp.add_argument("--json", action="store_true", help="Imprimir salida en formato JSON")
    args = argp.parse_args()

    fields = parse_invoice(args.image)
    if args.json:
        print(json.dumps(asdict(fields), ensure_ascii=False, indent=2))
    else:
        for key, value in asdict(fields).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
