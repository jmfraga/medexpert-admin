"""
MedExpert - Web Scraper Module
Hybrid 3-tier system for fetching clinical guidelines from the web.

Tier 1 (public):       Fetch and index open-access guidelines
Tier 2 (monitor):      Detect new versions on public pages of auth-required sites
Tier 3 (authenticated): Fetch with user-provided session cookie
"""

import re
import time
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

# Rate limiting: track last request time per domain
_domain_last_request: dict[str, float] = {}
_domain_request_count: dict[str, int] = {}
RATE_LIMIT_SECONDS = 2.0
RATE_LIMIT_EXTERNAL_SECONDS = 5.0  # More conservative for external domains

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
}

# ─────────────────────────────────────────────
# Known Sources Catalog
# ─────────────────────────────────────────────

KNOWN_SOURCES = [
    # ── Tier 1: Public sources ──
    {
        "name": "NCI PDQ - Breast Cancer Treatment",
        "url": "https://www.cancer.gov/types/breast/hp/breast-treatment-pdq",
        "source_type": "public",
        "category": "NCI",
        "css_selector_content": "#cgvBody .pdq-sections",
        "css_selector_version": "",
        "version_regex": "",
        "notes": "NCI Physician Data Query - open access",
    },
    {
        "name": "NCI PDQ - Colon Cancer Treatment",
        "url": "https://www.cancer.gov/types/colorectal/hp/colon-treatment-pdq",
        "source_type": "public",
        "category": "NCI",
        "css_selector_content": "#cgvBody .pdq-sections",
        "css_selector_version": "",
        "version_regex": "",
        "notes": "NCI Physician Data Query - open access",
    },
    {
        "name": "NCI PDQ - Lung Cancer Treatment",
        "url": "https://www.cancer.gov/types/lung/hp/non-small-cell-lung-treatment-pdq",
        "source_type": "public",
        "category": "NCI",
        "css_selector_content": "#cgvBody .pdq-sections",
        "css_selector_version": "",
        "version_regex": "",
        "notes": "NCI Physician Data Query - open access",
    },
    # ── Tier 2: Version monitors ──
    {
        "name": "NCCN Breast Cancer (monitor)",
        "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1419",
        "source_type": "monitor",
        "category": "NCCN",
        "css_selector_content": "",
        "css_selector_version": ".guideline-version, .version-info, h2, .guideline-detail",
        "version_regex": r"[Vv]ersion\s+(\d+\.\d{4})",
        "notes": "Requires login for full PDF. Monitor detects new versions.",
    },
    {
        "name": "NCCN Colon Cancer (monitor)",
        "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1428",
        "source_type": "monitor",
        "category": "NCCN",
        "css_selector_content": "",
        "css_selector_version": ".guideline-version, .version-info, h2, .guideline-detail",
        "version_regex": r"[Vv]ersion\s+(\d+\.\d{4})",
        "notes": "Requires login for full PDF. Monitor detects new versions.",
    },
    {
        "name": "NCCN NSCLC (monitor)",
        "url": "https://www.nccn.org/guidelines/guidelines-detail?category=1&id=1450",
        "source_type": "monitor",
        "category": "NCCN",
        "css_selector_content": "",
        "css_selector_version": ".guideline-version, .version-info, h2, .guideline-detail",
        "version_regex": r"[Vv]ersion\s+(\d+\.\d{4})",
        "notes": "Requires login for full PDF. Monitor detects new versions.",
    },
    # ── Tier 1: ESMO ──
    {
        "name": "ESMO Breast Cancer Guidelines",
        "url": "https://www.esmo.org/guidelines/guidelines-by-topic/breast-cancer",
        "source_type": "public",
        "category": "ESMO",
        "css_selector_content": ".field--name-body, .content-area, article",
        "css_selector_version": "",
        "version_regex": "",
        "notes": "ESMO clinical practice guidelines - summaries are public",
    },
    # ── Tier 1: CENETEC / GPC Mexico ──
    {
        "name": "GPC Mexico - Cancer de Mama",
        "url": "http://www.cenetec-difusion.com/CMGPC/GPC-SS-586-19/ER.pdf",
        "source_type": "public",
        "category": "CENETEC",
        "css_selector_content": "",
        "css_selector_version": "",
        "version_regex": "",
        "notes": "Guia de Practica Clinica Mexico - PDF directo",
    },
]


def _content_hash(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rate_limit(domain: str):
    """Enforce rate limiting per domain."""
    now = time.time()
    last = _domain_last_request.get(domain, 0)
    wait = RATE_LIMIT_SECONDS - (now - last)
    if wait > 0:
        time.sleep(wait)
    _domain_last_request[domain] = time.time()


def _extract_text_from_html(html: str, css_selector: str = "") -> str:
    """Extract text from HTML with fallback chain:
    1. CSS selector if provided
    2. All <p> tags
    3. All visible text
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try CSS selector first
    if css_selector:
        elements = soup.select(css_selector)
        if elements:
            text = "\n\n".join(el.get_text(separator="\n", strip=True) for el in elements)
            if text.strip():
                return text.strip()

    # Fallback: all <p> tags
    paragraphs = soup.find_all("p")
    if paragraphs:
        text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        if text.strip():
            return text.strip()

    # Final fallback: all visible text
    return soup.get_text(separator="\n", strip=True)


def _extract_version(html: str, css_selector_version: str, version_regex: str) -> str:
    """Extract version string from HTML using CSS selector + regex."""
    if not version_regex:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Get text to search from
    search_text = ""
    if css_selector_version:
        elements = soup.select(css_selector_version)
        if elements:
            search_text = " ".join(el.get_text(separator=" ", strip=True) for el in elements)

    if not search_text:
        search_text = soup.get_text(separator=" ", strip=True)

    match = re.search(version_regex, search_text)
    if match:
        return match.group(1) if match.lastindex else match.group(0)

    return ""


def _extract_links(html: str, base_url: str, url_pattern: str = "",
                    url_exclude: str = "", allowed_domains: list[str] = None) -> list[str]:
    """Extract and filter links from HTML page.
    By default only follows links on the same domain.
    allowed_domains: extra domains to follow (e.g. ['annalsofoncology.org']).
    Optionally filters by url_pattern regex and excludes by url_exclude regex.
    """
    soup = BeautifulSoup(html, "lxml")
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    extra_domains = set(allowed_domains or [])
    seen = set()
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        # Skip non-navigable links
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        href = urljoin(base_url, href)

        # Remove fragment
        href = href.split("#")[0]
        if not href:
            continue

        # Same domain or allowed external domains
        link_domain = urlparse(href).netloc
        if link_domain != base_domain and not any(d in link_domain for d in extra_domains):
            continue

        # Optional URL pattern filter (include)
        if url_pattern and not re.search(url_pattern, href):
            continue

        # Optional URL exclude filter
        if url_exclude and re.search(url_exclude, href):
            continue

        # Deduplicate
        if href not in seen:
            seen.add(href)
            links.append(href)

    return links


async def _fetch_with_browser(url: str, wait_selector: str = "", wait_ms: int = 5000) -> str:
    """Fetch a page using Playwright async headless browser (for SPAs).
    Returns the fully rendered HTML after JavaScript execution.
    Uses domcontentloaded + fixed wait to avoid hanging on analytics-heavy pages.
    Total timeout: 45 seconds.
    """
    import asyncio as _asyncio
    from playwright.async_api import async_playwright

    async def _do_fetch():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(
                    user_agent=HEADERS["User-Agent"],
                    viewport={"width": 1280, "height": 800},
                )
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                # Wait for JS to render content
                await page.wait_for_timeout(wait_ms)
                if wait_selector:
                    try:
                        await page.wait_for_selector(wait_selector, timeout=5000)
                    except Exception:
                        pass
                html = await page.content()
                return html
            finally:
                await browser.close()

    return await _asyncio.wait_for(_do_fetch(), timeout=60)


class GuidelineScraper:
    """Web scraper for clinical guidelines with 3-tier support."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    async def fetch_with_browser(self, url: str, css_selector: str = "") -> dict:
        """Fetch a page using headless browser for SPA sites. Returns {ok, text, html, error}."""
        try:
            # Handle PDF links (browser can't render PDFs, download instead)
            if url.lower().endswith(".pdf"):
                return self.fetch_public_source(url, css_selector)

            html = await _fetch_with_browser(url)
            text = _extract_text_from_html(html, css_selector)
            if not text or len(text) < 50:
                return {"ok": False, "text": "", "html": html, "error": "No se pudo extraer contenido significativo"}

            return {"ok": True, "text": text, "html": html, "error": ""}
        except Exception as e:
            return {"ok": False, "text": "", "html": "", "error": f"Browser error: {e}"}

    def fetch_public_source(self, url: str, css_selector: str = "") -> dict:
        """Tier 1: Fetch and parse a public source. Returns {ok, text, error}."""
        domain = urlparse(url).netloc
        _rate_limit(domain)

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")

            # Handle PDF downloads
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                return self._handle_pdf(resp.content, url)

            text = _extract_text_from_html(resp.text, css_selector)
            if not text or len(text) < 50:
                return {"ok": False, "text": "", "error": "No se pudo extraer contenido significativo"}

            return {"ok": True, "text": text, "error": ""}

        except requests.exceptions.Timeout:
            return {"ok": False, "text": "", "error": "Timeout (30s)"}
        except requests.exceptions.HTTPError as e:
            return {"ok": False, "text": "", "error": f"HTTP {e.response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "text": "", "error": str(e)}

    def _handle_pdf(self, pdf_bytes: bytes, url: str) -> dict:
        """Extract text from a downloaded PDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            if not text.strip():
                return {"ok": False, "text": "", "error": "PDF sin texto extraible"}
            return {"ok": True, "text": text.strip(), "error": ""}
        except ImportError:
            return {"ok": False, "text": "", "error": "PyMuPDF no instalado (pip install PyMuPDF)"}
        except Exception as e:
            return {"ok": False, "text": "", "error": f"Error procesando PDF: {e}"}

    def check_version(self, url: str, css_selector_version: str,
                      version_regex: str) -> dict:
        """Tier 2: Check version on a public page. Returns {ok, version, error}."""
        domain = urlparse(url).netloc
        _rate_limit(domain)

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()

            version = _extract_version(resp.text, css_selector_version, version_regex)
            if not version:
                return {"ok": True, "version": "", "error": "No se detecto version"}

            return {"ok": True, "version": version, "error": ""}

        except requests.exceptions.HTTPError as e:
            return {"ok": False, "version": "", "error": f"HTTP {e.response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "version": "", "error": str(e)}

    def fetch_authenticated(self, url: str, cookie_string: str,
                            css_selector: str = "") -> dict:
        """Tier 3: Fetch with session cookie. Returns {ok, text, error}."""
        domain = urlparse(url).netloc
        _rate_limit(domain)

        try:
            headers = {**HEADERS, "Cookie": cookie_string}
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code in (401, 403):
                return {
                    "ok": False, "text": "",
                    "error": f"Acceso denegado (HTTP {resp.status_code}). Cookie expirada o invalida."
                }

            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                return self._handle_pdf(resp.content, url)

            text = _extract_text_from_html(resp.text, css_selector)
            if not text or len(text) < 50:
                return {"ok": False, "text": "", "error": "No se pudo extraer contenido significativo"}

            return {"ok": True, "text": text, "error": ""}

        except requests.exceptions.HTTPError as e:
            return {"ok": False, "text": "", "error": f"HTTP {e.response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "text": "", "error": str(e)}

    def login(self, login_url: str, username: str, password: str) -> dict:
        """Login to a site using form POST. Returns {ok, error, cookies_count}.
        Automatically detects form fields and CSRF tokens.
        """
        domain = urlparse(login_url).netloc
        _rate_limit(domain)

        try:
            # GET the login page to find form and CSRF tokens
            resp = self.session.get(login_url, timeout=60)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Find login form (prefer form with password input)
            form = None
            for f in soup.find_all("form"):
                if f.find("input", {"type": "password"}):
                    form = f
                    break
            if not form:
                form = soup.find("form")
            if not form:
                return {"ok": False, "error": "No se encontro formulario de login"}

            # Resolve form action URL
            action = form.get("action", login_url)
            action = urljoin(login_url, action)
            method = form.get("method", "post").upper()

            # Collect all form fields (hidden inputs = CSRF tokens, etc.)
            form_data = {}
            username_field = None
            password_field = None

            for inp in form.find_all(["input", "select"]):
                name = inp.get("name")
                if not name:
                    continue
                input_type = inp.get("type", "text").lower()
                value = inp.get("value", "")

                if input_type == "password":
                    password_field = name
                elif input_type in ("text", "email") or any(
                    kw in name.lower() for kw in ("user", "email", "login", "account")
                ):
                    if not username_field:
                        username_field = name
                elif input_type in ("hidden", "submit"):
                    form_data[name] = value

            if not username_field or not password_field:
                return {"ok": False, "error": f"No se identificaron campos del formulario. "
                        f"Campos encontrados: {[inp.get('name') for inp in form.find_all('input')]}"}

            form_data[username_field] = username
            form_data[password_field] = password

            console.print(f"[cyan]Login: POST {action} (user_field={username_field})[/cyan]")

            _rate_limit(domain)
            resp = self.session.post(action, data=form_data, timeout=60, allow_redirects=True)

            # Heuristic: login succeeded if we got cookies or no login form in response
            cookies_count = len(self.session.cookies)
            response_soup = BeautifulSoup(resp.text, "lxml")
            still_has_login = bool(response_soup.find("input", {"type": "password"}))

            if cookies_count > 0 and not still_has_login:
                console.print(f"[green]Login exitoso ({cookies_count} cookies)[/green]")
                return {"ok": True, "error": "", "cookies_count": cookies_count}
            elif still_has_login:
                return {"ok": False, "error": "Login parece haber fallado (formulario sigue presente)"}
            else:
                # Ambiguous - assume success if we got a 200
                console.print(f"[yellow]Login ambiguo (HTTP {resp.status_code}, {cookies_count} cookies)[/yellow]")
                return {"ok": True, "error": "", "cookies_count": cookies_count}

        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": f"Error de conexion: {e}"}
        except Exception as e:
            return {"ok": False, "error": f"Error en login: {e}"}

    async def crawl(self, seed_url: str, max_depth: int = 1, url_pattern: str = "",
              url_exclude: str = "", css_selector: str = "",
              max_pages: int = 200, use_browser: bool = False,
              allowed_domains: list[str] = None,
              min_content_length: int = 2000) -> list[dict]:
        """Crawl from seed URL following links up to max_depth levels.
        Returns list of {url, title, text} for pages with extractable content.
        depth=0 means seed page only (same as single fetch).
        url_exclude: regex pattern to exclude links (e.g. translations, variants).
        url_pattern: regex filter for links. Use '||' to specify per-depth patterns:
            'pattern0||pattern1||pattern2' applies pattern0 at depth 0, pattern1 at depth 1, etc.
            A single pattern (no '||') applies at all depths.
        use_browser: if True, use Playwright headless browser for SPA sites.
        """
        visited = set()
        results = []

        # Parse depth-specific patterns (split by '||')
        if "||" in url_pattern:
            depth_patterns = url_pattern.split("||")
        else:
            depth_patterns = None  # same pattern at all depths

        def _get_pattern(depth: int) -> str:
            if depth_patterns is None:
                return url_pattern
            if depth < len(depth_patterns):
                return depth_patterns[depth].strip()
            return depth_patterns[-1].strip() if depth_patterns else ""

        seed_domain = urlparse(seed_url).netloc

        async def _fetch_page(url):
            """Fetch page HTML, using browser or requests based on setting.
            External domains (articles) always use requests (not browser).
            """
            if url.lower().endswith(".pdf"):
                # Always use requests for PDFs
                domain = urlparse(url).netloc
                _rate_limit(domain)
                resp = self.session.get(url, timeout=60)
                resp.raise_for_status()
                return {"html": None, "pdf_bytes": resp.content}

            if use_browser:
                domain = urlparse(url).netloc
                _rate_limit(domain)
                html = await _fetch_with_browser(url, wait_ms=8000)
                return {"html": html, "pdf_bytes": None}
            else:
                domain = urlparse(url).netloc
                _rate_limit(domain)
                resp = self.session.get(url, timeout=60)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "application/pdf" in content_type:
                    return {"html": None, "pdf_bytes": resp.content}
                return {"html": resp.text, "pdf_bytes": None}

        async def _crawl(url, depth):
            if url in visited or len(visited) >= max_pages:
                return
            visited.add(url)

            try:
                page_data = await _fetch_page(url)
            except Exception as e:
                console.print(f"[red]Crawl error {url}: {e}[/red]")
                return

            # Handle PDF
            if page_data["pdf_bytes"]:
                result = self._handle_pdf(page_data["pdf_bytes"], url)
                if result["ok"]:
                    title = url.split("/")[-1].replace(".pdf", "").replace("%20", " ")
                    results.append({"url": url, "title": title, "text": result["text"]})
                    console.print(f"  [green]PDF: {title} ({len(result['text'])} chars)[/green]")
                return

            html = page_data["html"]

            # Follow links if we haven't reached max depth
            if depth < max_depth:
                pattern_for_depth = _get_pattern(depth)
                links = _extract_links(html, seed_url, pattern_for_depth, url_exclude, allowed_domains)
                # Prioritize PDF/document links over navigation links
                links.sort(key=lambda l: (0 if l.lower().endswith('.pdf') else 1))
                console.print(f"  [cyan]Nivel {depth}: {url[:80]} → {len(links)} links[/cyan]")
                for link in links:
                    await _crawl(link, depth + 1)

            # Extract content from pages with enough text to be useful for RAG
            # Skip listing/navigation pages with minimal content
            text = _extract_text_from_html(html, css_selector)
            if text and len(text) >= min_content_length:
                soup = BeautifulSoup(html, "lxml")
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]
                results.append({"url": url, "title": title, "text": text})
                console.print(f"  [green]Contenido: {title[:60]} ({len(text)} chars)[/green]")

        mode = "browser" if use_browser else "requests"
        pattern_desc = " → ".join(depth_patterns) if depth_patterns else (url_pattern or "sin filtro")
        console.print(f"[bold cyan]Crawling {seed_url} (profundidad: {max_depth}, modo: {mode}, patron: {pattern_desc}, excluir: {url_exclude or 'nada'})[/bold cyan]")
        await _crawl(seed_url, 0)
        console.print(f"[bold green]Crawl completado: {len(results)} paginas con contenido, {len(visited)} visitadas[/bold green]")
        return results

    def index_content(self, text: str, source_name: str, category: str,
                      rag, expert_slug: str) -> int:
        """Save text to guides/web/ and index in RAG. Returns chunk count."""
        # Save to disk
        web_dir = Path(f"data/experts/{expert_slug}/guides/web")
        web_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_name = re.sub(r'[^\w\s-]', '', source_name).strip()
        safe_name = re.sub(r'[\s]+', '_', safe_name)
        filepath = web_dir / f"{safe_name}.txt"
        filepath.write_text(text, encoding="utf-8")

        # Delete existing chunks for this source, then re-index
        rag.delete_guideline(safe_name)
        rag.load_text(text, source=safe_name, category=category)

        console.print(f"[green]Indexed web source: {source_name} ({len(text)} chars)[/green]")
        return rag.get_total_count()

    async def fetch_and_index(self, source: dict, rag, expert_slug: str) -> dict:
        """Full flow: optional login + fetch/crawl + detect changes + index + update DB.
        source: dict from web_sources table.
        Returns {ok, message, new_version, content_changed, pages_found}.
        """
        import database as db

        source_type = source["source_type"]
        url = source["url"]
        now = datetime.now().isoformat()
        crawl_depth = source.get("crawl_depth", 0) or 0
        use_browser = bool(source.get("use_browser", 0))

        # ── Tier 2: Monitor only ──
        if source_type == "monitor":
            result = self.check_version(
                url,
                source.get("css_selector_version", ""),
                source.get("version_regex", ""),
            )
            if not result["ok"]:
                db.update_web_source_status(
                    source["id"], "error", error_message=result["error"],
                    last_checked=now,
                )
                return {"ok": False, "message": result["error"],
                        "new_version": "", "content_changed": False}

            new_version = result["version"]
            old_version = source.get("current_version", "")
            version_changed = bool(new_version and new_version != old_version)

            status = "update_available" if version_changed else "active"
            db.update_web_source_status(
                source["id"], status,
                current_version=new_version if new_version else old_version,
                last_checked=now,
            )
            msg = (f"Nueva version detectada: {new_version}" if version_changed
                   else f"Version actual: {new_version or 'no detectada'}")
            return {"ok": True, "message": msg,
                    "new_version": new_version, "content_changed": version_changed}

        # ── Auto-login if credentials are configured ──
        login_url = source.get("login_url", "")
        login_user = source.get("login_username", "")
        login_pass = source.get("login_password", "")

        if login_url and login_user and login_pass:
            login_result = self.login(login_url, login_user, login_pass)
            if not login_result["ok"]:
                db.update_web_source_status(
                    source["id"], "error",
                    error_message=f"Login fallido: {login_result['error']}",
                    last_checked=now,
                )
                return {"ok": False, "message": f"Login fallido: {login_result['error']}",
                        "new_version": "", "content_changed": False}

        # ── Multi-level crawl ──
        if crawl_depth > 0:
            # Parse allowed_domains from comma-separated string
            domains_str = source.get("allowed_domains", "")
            extra_domains = [d.strip() for d in domains_str.split(",") if d.strip()] if domains_str else None

            min_len = source.get("min_content_length", 2000) or 2000

            pages = await self.crawl(
                url,
                max_depth=crawl_depth,
                url_pattern=source.get("url_pattern", ""),
                url_exclude=source.get("url_exclude", ""),
                css_selector=source.get("css_selector_content", ""),
                use_browser=use_browser,
                allowed_domains=extra_domains,
                min_content_length=min_len,
            )

            if not pages:
                db.update_web_source_status(
                    source["id"], "error",
                    error_message="Crawl no encontro contenido",
                    last_checked=now,
                )
                return {"ok": False, "message": "Crawl no encontro contenido",
                        "new_version": "", "content_changed": False}

            # Index each page separately for better RAG retrieval
            all_hashes = []
            category = source.get("category", "")
            for page in pages:
                page_name = f"{source['name']} - {page['title']}"
                self.index_content(page["text"], page_name, category, rag, expert_slug)
                all_hashes.append(_content_hash(page["text"]))

            new_hash = _content_hash("".join(all_hashes))
            old_hash = source.get("content_hash", "")
            content_changed = new_hash != old_hash

            db.update_web_source_status(
                source["id"], "active",
                content_hash=new_hash,
                last_fetched=now,
                last_checked=now,
            )

            msg = f"Crawl completado: {len(pages)} paginas indexadas"
            if not content_changed and old_hash:
                msg += " (sin cambios)"
            return {"ok": True, "message": msg, "pages_found": len(pages),
                    "new_version": "", "content_changed": content_changed}

        # ── Single page fetch (Tier 1 / Tier 3) ──
        if source_type == "authenticated":
            cookie = source.get("session_cookie", "")
            if not cookie and not (login_url and login_user):
                db.update_web_source_status(
                    source["id"], "error",
                    error_message="Cookie de sesion no configurada y sin credenciales de login",
                    last_checked=now,
                )
                return {"ok": False, "message": "Cookie de sesion no configurada y sin credenciales de login",
                        "new_version": "", "content_changed": False}
            if cookie:
                result = self.fetch_authenticated(
                    url, cookie, source.get("css_selector_content", ""),
                )
            else:
                # Already logged in via session, use regular fetch
                result = self.fetch_public_source(url, source.get("css_selector_content", ""))
        elif use_browser:
            result = await self.fetch_with_browser(url, source.get("css_selector_content", ""))
        else:
            result = self.fetch_public_source(url, source.get("css_selector_content", ""))

        if not result["ok"]:
            db.update_web_source_status(
                source["id"], "error", error_message=result["error"],
                last_checked=now,
            )
            return {"ok": False, "message": result["error"],
                    "new_version": "", "content_changed": False}

        text = result["text"]
        new_hash = _content_hash(text)
        old_hash = source.get("content_hash", "")
        content_changed = new_hash != old_hash

        self.index_content(text, source["name"], source.get("category", ""),
                           rag, expert_slug)

        db.update_web_source_status(
            source["id"], "active",
            content_hash=new_hash,
            last_fetched=now,
            last_checked=now,
        )

        msg = "Contenido actualizado e indexado"
        if not content_changed and old_hash:
            msg = "Contenido sin cambios, reindexado"

        return {"ok": True, "message": msg,
                "new_version": "", "content_changed": content_changed}
