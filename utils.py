"""
MedExpert - Shared Utilities
"""

import re
import unicodedata


def generate_slug(name: str) -> str:
    """Generate a filesystem-safe slug from a name.

    Examples:
        "Oncologia" -> "oncologia"
        "Cardiologia Pediatrica" -> "cardiologia-pediatrica"
        "Neurologia (adultos)" -> "neurologia-adultos"
    """
    # Normalize unicode and strip accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    # Lowercase, replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug)
    return slug or "expert"
