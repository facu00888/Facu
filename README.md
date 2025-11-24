# Mi primer repo

Este repositorio incluye un script simple para extraer datos clave de facturas escaneadas mediante OCR.

## Requisitos

- Python 3.9+
- [Tesseract](https://github.com/tesseract-ocr/tesseract) no es necesario; se utiliza EasyOCR.
- Dependencias Python: `pip install -r requirements.txt` (EasyOCR descargará modelos la primera vez que se ejecute).

## Uso

```bash
python invoice_parser.py ruta/a/factura.jpg --json
```

La salida incluye los campos:

- Fecha
- Serie y Folio
- Razón Social
- RUT del Emisor
- Importe total con IVA

El script usa heurísticas basadas en etiquetas comunes (por ejemplo, `Fecha`, `Serie y Folio`, `Razón Social`, `RUT`, `Importe total`).
