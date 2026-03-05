"""
Clinical Metadata Extraction — config-driven, zero LLM calls.

Public API:
    extract(query_text, response_text, specialty) -> dict

Patterns are loaded from metadata_patterns/{specialty}.json.
Adding a new specialty = creating one JSON file, zero code changes.
Diagnosis codes follow CIE-10 (ICD-10).
"""

import json
import os
import re
import unicodedata

_DIR = os.path.dirname(os.path.abspath(__file__))
_PATTERNS_DIR = os.path.join(_DIR, "metadata_patterns")


# ──────────────────────────────────────────────
# Text normalization
# ──────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase + strip accents for matching."""
    text = text.lower()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ──────────────────────────────────────────────
# Config loading & compilation (cached)
# ──────────────────────────────────────────────

_compiled_cache: dict[str, dict] = {}


def _load_config(specialty: str) -> dict:
    """Load and compile patterns for a specialty. Cached after first call."""
    if specialty in _compiled_cache:
        return _compiled_cache[specialty]

    config_path = os.path.join(_PATTERNS_DIR, f"{specialty}.json")
    if not os.path.exists(config_path):
        _compiled_cache[specialty] = {}
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    compiled = _compile_config(raw)
    _compiled_cache[specialty] = compiled
    return compiled


def _compile_config(raw: dict) -> dict:
    """Compile regex strings from JSON into re.Pattern objects."""
    # Diagnoses: list of (compiled_pattern, name, cie10, category)
    diagnoses = []
    for d in raw.get("diagnoses", []):
        combined = "|".join(d["patterns"])
        pat = re.compile(r"\b(?:" + combined + r")\b", re.IGNORECASE)
        diagnoses.append((pat, d["name"], d.get("cie10", ""), d.get("category", "")))

    # Subtypes: list of (compiled_pattern, name)
    # No trailing \b — patterns may end with non-word chars like +/-
    subtypes = []
    for s in raw.get("subtypes", []):
        pat = re.compile(r"\b(?:" + s["pattern"] + r")", re.IGNORECASE)
        subtypes.append((pat, s["name"]))

    # Intents: list of (compiled_pattern, name)
    intents = []
    for i in raw.get("intents", []):
        pat = re.compile(r"\b(?:" + i["pattern"] + r")\b", re.IGNORECASE)
        intents.append((pat, i["name"]))

    # Flags: dict of name -> compiled_pattern
    flags = {}
    for name, patterns in raw.get("flags", {}).items():
        combined = "|".join(patterns)
        flags[name] = re.compile(r"\b(?:" + combined + r")\b", re.IGNORECASE)

    return {
        "specialty": raw.get("specialty", ""),
        "cie10_chapter": raw.get("cie10_chapter", ""),
        "severity": raw.get("severity", {}),
        "treatment_line": raw.get("treatment_line", False),
        "diagnoses": diagnoses,
        "subtypes": subtypes,
        "intents": intents,
        "flags": flags,
    }


# ──────────────────────────────────────────────
# Severity / staging systems (built-in handlers)
# ──────────────────────────────────────────────

_ROMAN_MAP = {"i": "I", "ii": "II", "iii": "III", "iv": "IV"}

# Pre-compiled patterns for oncology staging
_STAGE_ROMAN = re.compile(
    r"(?:estadio|etapa|stage)\s*(iv|iii|ii|i)([abc])?", re.IGNORECASE,
)
_STAGE_NUMERIC = re.compile(
    r"(?:estadio|etapa|stage)\s+([1-4])([abc])?", re.IGNORECASE,
)
_STAGE_TNM = re.compile(
    r"\b(T[0-4x](?:is)?)\s*(N[0-3x])\s*(M[01x])\b", re.IGNORECASE,
)

# Pre-compiled patterns for NYHA classification
_NYHA_ROMAN = re.compile(
    r"(?:nyha|clase\s+funcional)\s*(iv|iii|ii|i)\b", re.IGNORECASE,
)
_NYHA_NUMERIC = re.compile(
    r"(?:nyha|clase\s+funcional)\s+([1-4])\b", re.IGNORECASE,
)

# Pre-compiled pattern for ejection fraction
_EF_PATTERN = re.compile(
    r"(?:fevi|fe|fraccion\s+(?:de\s+)?eyeccion)\s*(?:del?\s*)?(\d{1,2})\s*%", re.IGNORECASE,
)

# Pre-compiled patterns for Killip class
_KILLIP_ROMAN = re.compile(
    r"killip\s*(iv|iii|ii|i)\b", re.IGNORECASE,
)
_KILLIP_NUMERIC = re.compile(
    r"killip\s+([1-4])\b", re.IGNORECASE,
)


def _extract_severity(norm: str, severity_cfg: dict) -> dict:
    """Extract severity using the handler specified in config.

    Returns dict with severity-specific fields (e.g. stage, nyha_class, ef_pct).
    """
    stype = severity_cfg.get("type", "")

    if stype == "oncology_stage":
        return _severity_oncology(norm)
    elif stype == "cardiac_functional":
        return _severity_cardiac(norm)

    return {}


def _severity_oncology(norm: str) -> dict:
    """Oncology staging: Roman numeral, numeric, TNM."""
    stage = None
    m = _STAGE_ROMAN.search(norm)
    if m:
        base = _ROMAN_MAP.get(m.group(1).lower(), m.group(1).upper())
        suffix = m.group(2).upper() if m.group(2) else ""
        stage = base + suffix
    if not stage:
        m = _STAGE_NUMERIC.search(norm)
        if m:
            num = int(m.group(1))
            roman = {1: "I", 2: "II", 3: "III", 4: "IV"}.get(num, str(num))
            suffix = m.group(2).upper() if m.group(2) else ""
            stage = roman + suffix
    if not stage:
        m = _STAGE_TNM.search(norm)
        if m:
            stage = f"{m.group(1).upper()}{m.group(2).upper()}{m.group(3).upper()}"
    return {"stage": stage}


def _severity_cardiac(norm: str) -> dict:
    """Cardiac severity: NYHA class, ejection fraction, Killip."""
    result = {"nyha_class": None, "ef_pct": None, "killip": None}

    # NYHA
    m = _NYHA_ROMAN.search(norm)
    if m:
        result["nyha_class"] = _ROMAN_MAP.get(m.group(1).lower(), m.group(1).upper())
    else:
        m = _NYHA_NUMERIC.search(norm)
        if m:
            num = int(m.group(1))
            result["nyha_class"] = {1: "I", 2: "II", 3: "III", 4: "IV"}.get(num)

    # Ejection fraction
    m = _EF_PATTERN.search(norm)
    if m:
        result["ef_pct"] = int(m.group(1))

    # Killip
    m = _KILLIP_ROMAN.search(norm)
    if m:
        result["killip"] = _ROMAN_MAP.get(m.group(1).lower(), m.group(1).upper())
    else:
        m = _KILLIP_NUMERIC.search(norm)
        if m:
            num = int(m.group(1))
            result["killip"] = {1: "I", 2: "II", 3: "III", 4: "IV"}.get(num)

    return result


# ──────────────────────────────────────────────
# Treatment line (shared across specialties)
# ──────────────────────────────────────────────

_LINE_PATTERN = re.compile(
    r"\b(primera|segunda|tercera|cuarta|1[era]{0,3}|2[da]{0,2}|3[era]{0,3}|4[ta]{0,2})\s+linea\b",
    re.IGNORECASE,
)

_LINE_MAP = {
    "primera": "primera", "1": "primera", "1era": "primera", "1a": "primera",
    "segunda": "segunda", "2": "segunda", "2da": "segunda", "2a": "segunda",
    "tercera": "tercera", "3": "tercera", "3era": "tercera", "3a": "tercera",
    "cuarta": "cuarta", "4": "cuarta", "4ta": "cuarta", "4a": "cuarta",
}


def _extract_treatment_line(norm: str) -> str | None:
    m = _LINE_PATTERN.search(norm)
    if m:
        raw = m.group(1).lower().rstrip("adert")
        return _LINE_MAP.get(raw, _LINE_MAP.get(m.group(1).lower()))
    return None


# ──────────────────────────────────────────────
# Treatment extraction (from glossary DB)
# ──────────────────────────────────────────────

_drug_names_cache: dict[str, list[tuple[re.Pattern, str]]] = {}


def _load_drug_patterns(specialty: str) -> list[tuple[re.Pattern, str]]:
    """Lazy-load drug patterns from glossary DB."""
    if specialty in _drug_names_cache:
        return _drug_names_cache[specialty]

    patterns = []
    try:
        import database as db
        terms = db.get_glossary_terms_for_expert_by_slug(specialty)
        for t in terms:
            generic = t.get("term", "").strip()
            synonyms_str = t.get("synonyms", "").strip()
            # Only include terms that have brand-name synonyms (actual drugs)
            if not generic or len(generic) < 3 or not synonyms_str:
                continue
            # Generic name pattern
            escaped = re.escape(generic.lower())
            patterns.append((re.compile(r"\b" + escaped + r"\b", re.IGNORECASE), generic.lower()))
            # Synonyms (brand names)
            for syn in synonyms_str.split(","):
                syn = syn.strip()
                if syn and len(syn) >= 3:
                    escaped_syn = re.escape(syn.lower())
                    patterns.append((re.compile(r"\b" + escaped_syn + r"\b", re.IGNORECASE), generic.lower()))
    except Exception:
        pass

    _drug_names_cache[specialty] = patterns
    return patterns


def _extract_treatments(norm: str, original: str, specialty: str) -> list[str]:
    """Extract mentioned treatments, returning unique generic names."""
    patterns = _load_drug_patterns(specialty)
    found = set()
    combined = norm + " " + original.lower()
    for pat, generic in patterns:
        if pat.search(combined):
            found.add(generic)
    return sorted(found)


# ──────────────────────────────────────────────
# Main extraction
# ──────────────────────────────────────────────

def extract(query_text: str, response_text: str, specialty: str = "oncologia") -> dict:
    """
    Extract clinical metadata from query + response text.

    Loads patterns from metadata_patterns/{specialty}.json.
    Returns dict with: diagnosis, cie10, category, clinical_details,
    treatments_mentioned, intent, extraction_method.
    """
    cfg = _load_config(specialty)
    if not cfg:
        return _empty_result()

    combined = (query_text or "") + " " + (response_text or "")
    norm = _normalize(combined)
    query_norm = _normalize(query_text or "")

    # Diagnosis — prefer query text, fallback to combined
    diagnosis = None
    cie10 = None
    category = None
    for pat, name, code, cat in cfg["diagnoses"]:
        if pat.search(query_norm):
            diagnosis, cie10, category = name, code, cat
            break
    if not diagnosis:
        for pat, name, code, cat in cfg["diagnoses"]:
            if pat.search(norm):
                diagnosis, cie10, category = name, code, cat
                break

    # Severity (staging system depends on specialty config)
    severity_cfg = cfg.get("severity", {})
    severity = _extract_severity(norm, severity_cfg)

    # Subtypes
    subtype = None
    for pat, name in cfg["subtypes"]:
        if pat.search(combined) or pat.search(norm):
            subtype = name
            break

    # Treatment line (if specialty uses it)
    treatment_line = None
    if cfg.get("treatment_line"):
        treatment_line = _extract_treatment_line(norm)

    # Flags (e.g. metastatic)
    flag_results = {}
    for flag_name, flag_pat in cfg["flags"].items():
        flag_results[flag_name] = bool(flag_pat.search(norm))

    # Oncology-specific: stage IV implies metastatic
    stage = severity.get("stage")
    if stage and stage.startswith("IV") and "metastatic" in flag_results:
        flag_results["metastatic"] = True

    # Build clinical_details from severity + subtype + line + flags
    clinical_details = dict(severity)
    clinical_details["subtype"] = subtype
    clinical_details["treatment_line"] = treatment_line
    clinical_details.update(flag_results)

    # Intents
    intent = None
    for pat, name in cfg["intents"]:
        if pat.search(norm):
            intent = name
            break

    # Treatments from glossary
    treatments = _extract_treatments(norm, combined, specialty)

    return {
        "diagnosis": diagnosis,
        "cie10": cie10,
        "category": category,
        "clinical_details": clinical_details,
        "treatments_mentioned": treatments,
        "intent": intent,
        "extraction_method": "keyword",
    }


def _empty_result() -> dict:
    """Return empty metadata when no config is available."""
    return {
        "diagnosis": None,
        "cie10": None,
        "category": None,
        "clinical_details": {},
        "treatments_mentioned": [],
        "intent": None,
        "extraction_method": "keyword",
    }


def reload_config(specialty: str | None = None):
    """Clear cached config. Call after editing a JSON pattern file."""
    if specialty:
        _compiled_cache.pop(specialty, None)
    else:
        _compiled_cache.clear()


# ──────────────────────────────────────────────
# Built-in tests (run: python clinical_metadata.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    test_cases = [
        {
            "query": "Cancer de mama HER2+ estadio II, trastuzumab",
            "response": "El tratamiento con trastuzumab en cancer de mama HER2 positivo estadio II...",
            "expect_diagnosis": "cancer de mama",
            "expect_cie10": "C50",
            "expect_stage": "II",
            "expect_subtype": "HER2+",
        },
        {
            "query": "Opciones para cancer de pulmon etapa IIIA con EGFR mutado",
            "response": "En CPNM con mutacion EGFR, las opciones incluyen osimertinib como primera linea...",
            "expect_diagnosis": "cancer de pulmon",
            "expect_cie10": "C34",
            "expect_stage": "IIIA",
            "expect_subtype": "EGFR+",
        },
        {
            "query": "Linfoma no Hodgkin difuso de celulas B grandes",
            "response": "El esquema R-CHOP sigue siendo el estandar en primera linea para LDCBG...",
            "expect_diagnosis": "linfoma no hodgkin",
            "expect_cie10": "C82",
            "expect_category": "neoplasia_hematologica",
        },
        {
            "query": "Mieloma multiple recaido, que opciones paliativas hay",
            "response": "En mieloma refractario se pueden considerar daratumumab, lenalidomida...",
            "expect_diagnosis": "mieloma multiple",
            "expect_cie10": "C90",
            "expect_intent": "paliativo",
        },
        {
            "query": "Cancer colorrectal estadio IV metastasico MSI-H",
            "response": "En CCR metastasico con MSI-H, pembrolizumab ha demostrado beneficio...",
            "expect_diagnosis": "cancer colorrectal",
            "expect_cie10": "C18",
            "expect_stage": "IV",
            "expect_metastatic": True,
            "expect_subtype": "MSI-H",
        },
        {
            "query": "Seguimiento post quirurgico cancer de tiroides",
            "response": "El seguimiento incluye tiroglobulina serica y ecografia cervical...",
            "expect_diagnosis": "cancer de tiroides",
            "expect_cie10": "C73",
            "expect_intent": "seguimiento",
        },
        {
            "query": "Triple negativo mama neoadyuvancia",
            "response": "El esquema de neoadyuvancia con carboplatino y pembrolizumab ha mostrado...",
            "expect_diagnosis": "cancer de mama",
            "expect_subtype": "triple negativo",
            "expect_intent": "neoadyuvante",
        },
        {
            "query": "T2N1M0 cancer de prostata",
            "response": "Con clasificacion TNM T2N1M0 se considera enfermedad localmente avanzada...",
            "expect_diagnosis": "cancer de prostata",
            "expect_cie10": "C61",
            "expect_stage": "T2N1M0",
        },
        {
            "query": "Que es la anemia",
            "response": "La anemia es una condicion en la que no hay suficientes globulos rojos...",
            "expect_diagnosis": None,
        },
    ]

    print("=" * 60)
    print("Clinical Metadata Extraction — Test Suite")
    print(f"Config dir: {_PATTERNS_DIR}")
    print("=" * 60)

    passed = 0
    failed = 0
    for i, tc in enumerate(test_cases, 1):
        result = extract(tc["query"], tc["response"])
        ok = True
        errors = []

        if "expect_diagnosis" in tc and result["diagnosis"] != tc["expect_diagnosis"]:
            ok = False
            errors.append(f"diagnosis: got '{result['diagnosis']}', expected '{tc['expect_diagnosis']}'")
        if "expect_cie10" in tc and result["cie10"] != tc["expect_cie10"]:
            ok = False
            errors.append(f"cie10: got '{result['cie10']}', expected '{tc['expect_cie10']}'")
        if "expect_category" in tc and result["category"] != tc["expect_category"]:
            ok = False
            errors.append(f"category: got '{result['category']}', expected '{tc['expect_category']}'")
        if "expect_stage" in tc and result["clinical_details"].get("stage") != tc["expect_stage"]:
            ok = False
            errors.append(f"stage: got '{result['clinical_details'].get('stage')}', expected '{tc['expect_stage']}'")
        if "expect_subtype" in tc and result["clinical_details"].get("subtype") != tc["expect_subtype"]:
            ok = False
            errors.append(f"subtype: got '{result['clinical_details'].get('subtype')}', expected '{tc['expect_subtype']}'")
        if "expect_intent" in tc and result["intent"] != tc["expect_intent"]:
            ok = False
            errors.append(f"intent: got '{result['intent']}', expected '{tc['expect_intent']}'")
        if "expect_metastatic" in tc and result["clinical_details"].get("metastatic") != tc["expect_metastatic"]:
            ok = False
            errors.append(f"metastatic: got '{result['clinical_details'].get('metastatic')}', expected '{tc['expect_metastatic']}'")

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] Test {i}: {tc['query'][:60]}...")
        if not ok:
            for err in errors:
                print(f"  -> {err}")
        else:
            cie = result.get('cie10', '')
            diag = result['diagnosis']
            stg = result['clinical_details'].get('stage')
            sub = result['clinical_details'].get('subtype')
            intent = result['intent']
            print(f"  [{cie}] {diag}, stage={stg}, subtype={sub}, intent={intent}")

    # Performance test
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")

    print(f"\nPerformance test (1000 extractions)...")
    t0 = time.perf_counter()
    for _ in range(1000):
        extract("Cancer de mama HER2+ estadio IIIA primera linea trastuzumab",
                "Tratamiento con pertuzumab y trastuzumab en neoadyuvancia...")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  1000 extractions: {elapsed:.1f}ms ({elapsed/1000:.3f}ms each)")
    print(f"  {'PASS' if elapsed < 200 else 'SLOW'}: target < 200ms total")
