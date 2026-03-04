"""
MedExpert - Text Anonymizer
Regex-based PII removal for medical consultation queries.
Strips personal identifiers before storing in database.
"""

import re


# Mexican CURP: 18 alphanumeric characters
_CURP_RE = re.compile(r'\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]\d\b', re.IGNORECASE)

# Mexican RFC: 12-13 alphanumeric characters
_RFC_RE = re.compile(r'\b[A-ZÑ&]{3,4}\d{6}[A-Z\d]{3}\b', re.IGNORECASE)

# Email addresses
_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Phone numbers (Mexican: 10 digits, with optional +52, country codes)
_PHONE_RE = re.compile(
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b'
)

# Mexican NSS (Social Security Number): 11 digits
_NSS_RE = re.compile(r'\b\d{11}\b')

# Credit card numbers (13-19 digits, possibly with spaces/dashes)
_CARD_RE = re.compile(r'\b(?:\d[-\s]?){13,19}\b')

# Addresses: common patterns (Calle, Av., Col., C.P., etc.)
_ADDRESS_RE = re.compile(
    r'(?:(?:calle|avenida|av\.?|blvd\.?|boulevard|col\.?|colonia|c\.?p\.?|codigo postal|'
    r'manzana|mza\.?|lote|lt\.?|numero|num\.?|no\.?|int\.?|ext\.?|depto\.?|departamento|'
    r'piso|edificio|entre|y\s+calle)\s*[:#]?\s*[\w\d.,\s#-]{3,40})',
    re.IGNORECASE
)

# Specific name patterns: "paciente [Name]", "Dr./Dra. [Name]", "nombre: [Name]"
_NAME_PATTERN_RE = re.compile(
    r'(?:(?:paciente|nombre|dr\.?a?|lic\.?|ing\.?)\s*:?\s+)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})',
    re.IGNORECASE
)


def anonymize_text(text: str) -> str:
    """Remove PII from text, replacing with generic tokens."""
    if not text:
        return text

    result = text

    # Replace specific patterns (order matters - most specific first)
    result = _CURP_RE.sub('[CURP]', result)
    result = _RFC_RE.sub('[RFC]', result)
    result = _EMAIL_RE.sub('[EMAIL]', result)
    result = _CARD_RE.sub('[TARJETA]', result)
    result = _NSS_RE.sub('[NSS]', result)
    result = _PHONE_RE.sub('[TELEFONO]', result)
    result = _ADDRESS_RE.sub('[DIRECCION]', result)

    # Replace name patterns (keep the prefix, replace the name)
    def _replace_name(match):
        prefix = match.group(0)[:match.start(1) - match.start(0)]
        return prefix + '[NOMBRE]'
    result = _NAME_PATTERN_RE.sub(_replace_name, result)

    return result
