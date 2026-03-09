"""
MedExpert - PubMed E-utilities Integration
Search PubMed for systematic reviews, meta-analyses, and RCTs.
Returns structured paper references for clinical consultations.
"""

import os
import logging
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET

logger = logging.getLogger("medexpert.pubmed")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Filter for high-quality evidence
EVIDENCE_FILTER = "(systematic review[pt] OR meta-analysis[pt] OR randomized controlled trial[pt])"


def search_pubmed(query: str, max_results: int = 5, timeout: int = 10) -> list[dict]:
    """Search PubMed for relevant papers and return structured results.

    Args:
        query: Clinical search query (will be combined with evidence filter)
        max_results: Number of papers to return (3 for basic, 5 for premium)
        timeout: Request timeout in seconds

    Returns:
        List of dicts with: title, authors, year, abstract, pmid, doi
    """
    api_key = os.getenv("NCBI_API_KEY", "")

    try:
        # Step 1: Search for PMIDs
        pmids = _esearch(query, max_results, api_key, timeout)
        if not pmids:
            return []

        # Step 2: Fetch paper details
        papers = _efetch(pmids, api_key, timeout)
        return papers

    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []


def _esearch(query: str, max_results: int, api_key: str, timeout: int) -> list[str]:
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "date",
        "term": f"{query} AND {EVIDENCE_FILTER}",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "MedExpert/1.0"})

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())

    pmids = data.get("esearchresult", {}).get("idlist", [])
    logger.info(f"PubMed search: {data['esearchresult'].get('count', 0)} results, returning {len(pmids)}")
    return pmids


def _efetch(pmids: list[str], api_key: str, timeout: int) -> list[dict]:
    """Fetch paper details from PubMed by PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "MedExpert/1.0"})

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        xml_data = resp.read().decode()

    root = ET.fromstring(xml_data)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        paper = _parse_article(article)
        if paper:
            papers.append(paper)

    return papers


def _parse_article(article) -> dict | None:
    """Parse a PubmedArticle XML element into a structured dict."""
    title = article.findtext(".//ArticleTitle", "").strip()
    if not title:
        return None

    # Authors (first 3 + et al.)
    authors = []
    for author in article.findall(".//Author")[:3]:
        last = author.findtext("LastName", "")
        initials = author.findtext("Initials", "")
        if last:
            authors.append(f"{last} {initials}")
    total_authors = len(article.findall(".//Author"))
    author_str = ", ".join(authors)
    if total_authors > 3:
        author_str += " et al."

    # Year — try multiple locations
    year = (article.findtext(".//Journal//PubDate/Year", "")
            or article.findtext(".//ArticleDate/Year", "")
            or article.findtext(".//DateCompleted/Year", ""))
    if not year and article.findtext(".//PubDate/MedlineDate"):
        year = article.findtext(".//PubDate/MedlineDate", "")[:4]

    # PMID
    pmid = article.findtext(".//PMID", "")

    # DOI
    doi = ""
    for aid in article.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text or ""
            break

    # Abstract — join all AbstractText elements
    abstract_parts = []
    for at in article.findall(".//AbstractText"):
        label = at.get("Label", "")
        text = at.text or ""
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts).strip()

    # Journal
    journal = article.findtext(".//Journal/Title", "")

    return {
        "title": title,
        "authors": author_str,
        "year": year,
        "pmid": pmid,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}" if doi else "",
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "abstract": abstract,
        "journal": journal,
    }


def translate_abstracts(papers: list[dict]) -> list[dict]:
    """Translate English abstracts to Spanish using Groq (fast, free).
    Modifies papers in-place, adding 'abstract_es' field.
    Falls back to English if translation fails."""
    if not papers:
        return papers

    abstracts_to_translate = []
    for p in papers:
        short = p["abstract"][:400] if len(p["abstract"]) > 400 else p["abstract"]
        abstracts_to_translate.append(short)

    if not abstracts_to_translate:
        return papers

    try:
        from openai import OpenAI
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            logger.warning("No GROQ_API_KEY for abstract translation")
            return papers

        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

        # Build batch prompt
        numbered = "\n\n".join(f"[{i+1}] {a}" for i, a in enumerate(abstracts_to_translate))
        prompt = (
            "Traduce los siguientes abstracts medicos del ingles al español. "
            "Mantén la terminologia medica precisa. "
            "Responde SOLO con las traducciones numeradas en el mismo formato [1], [2], etc. "
            "No agregues explicaciones.\n\n"
            f"{numbered}"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,
        )
        translated_text = response.choices[0].message.content or ""

        # Parse numbered translations
        import re
        translations = {}
        for match in re.finditer(r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)', translated_text, re.DOTALL):
            idx = int(match.group(1)) - 1
            translations[idx] = match.group(2).strip()

        for i, p in enumerate(papers):
            p["abstract_es"] = translations.get(i, p["abstract"])

        logger.info(f"Translated {len(translations)}/{len(papers)} abstracts to Spanish")

    except Exception as e:
        logger.warning(f"Abstract translation failed (using English): {e}")
        for p in papers:
            p["abstract_es"] = p["abstract"]

    return papers


def format_papers_telegram(papers: list[dict]) -> str:
    """Format papers for Telegram message (plain text, no markdown)."""
    if not papers:
        return ""

    lines = ["\nLITERATURA CIENTIFICA RECIENTE (PubMed):"]
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract_es", p["abstract"])
        abstract_short = abstract[:250] + "..." if len(abstract) > 250 else abstract
        lines.append(f"\n{i}. {p['title']}")
        lines.append(f"   {p['authors']} ({p['year']})")
        if p["journal"]:
            lines.append(f"   {p['journal']}")
        lines.append(f"   {abstract_short}")
        if p["doi_url"]:
            lines.append(f"   DOI: {p['doi_url']}")
        else:
            lines.append(f"   PMID: {p['pmid']}")

    return "\n".join(lines)
