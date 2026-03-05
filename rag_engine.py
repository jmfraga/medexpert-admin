"""
MedExpert Admin - RAG Engine
Full read/write RAG engine for clinical guidelines using ChromaDB.
Manages indexing, searching, and guideline lifecycle per expert.
"""

import re
from pathlib import Path
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────
# Clinical Chunking
# ─────────────────────────────────────────────

# Patterns that indicate a section header in clinical documents
_HEADER_PATTERNS = [
    re.compile(r"^#{1,4}\s+.+"),                         # Markdown headers
    re.compile(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s\-/&:,()]{4,}$"),  # ALL-CAPS lines (5+ chars)
    re.compile(r"^\d+\.\d*\s+[A-Z]"),                    # Numbered sections: "1.2 Treatment"
    re.compile(r"^(?:ESTADIO|STAGE|SECTION|CHAPTER|TABLE|FIGURE)\b", re.IGNORECASE),
    re.compile(r"^(?:Introduction|Background|Methods|Results|Discussion|Recommendations|References|Conclusiones|Tratamiento|Diagnostico|Evaluacion|Seguimiento)\b", re.IGNORECASE),
]


def _is_header(line: str) -> bool:
    """Detect if a line looks like a section header."""
    line = line.strip()
    if not line or len(line) > 200:
        return False
    return any(p.match(line) for p in _HEADER_PATTERNS)


def _extract_metadata_from_source(source: str, category: str = "") -> dict:
    """Extract structured metadata from source name."""
    meta = {"source": source, "category": category}

    # Detect society/organization
    source_upper = source.upper()
    for society in ["NCCN", "ESMO", "ASCO", "NCI", "IMSS", "CENETEC", "WHO", "AHA", "ACC", "ESC"]:
        if society in source_upper:
            meta["society"] = society
            break

    # Detect year (4 digits, 2019-2029)
    year_match = re.search(r"\b(20[12]\d)\b", source)
    if year_match:
        meta["year"] = year_match.group(1)

    # Detect doc type from source name
    for dtype, keywords in [
        ("guideline", ["guideline", "guia", "gpc"]),
        ("consensus", ["consensus", "consenso"]),
        ("review", ["review", "revision"]),
        ("article", ["article", "articulo", "annals"]),
    ]:
        if any(kw in source.lower() for kw in keywords):
            meta["doc_type"] = dtype
            break

    return meta


def chunk_clinical_text(text: str, source: str, category: str = "",
                        chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split clinical text into chunks with overlap and section awareness.

    Returns list of {id, text, metadata} dicts ready for ChromaDB.
    - Splits on section headers first, then by size within sections
    - Preserves section_path in metadata for traceability
    - Overlap between chunks to avoid losing context at boundaries
    """
    base_meta = _extract_metadata_from_source(source, category)
    chunks = []
    current_section = ""
    current_text = ""
    section_stack = []  # track nested sections

    lines = text.split("\n")

    def _flush(section_path: str):
        nonlocal current_text
        txt = current_text.strip()
        if not txt:
            return
        # Split large sections into overlapping chunks
        if len(txt) <= chunk_size:
            meta = {**base_meta, "section_path": section_path}
            chunks.append({"text": txt, "metadata": meta})
        else:
            # Overlap-aware splitting within section
            pos = 0
            while pos < len(txt):
                end = pos + chunk_size
                chunk_text = txt[pos:end].strip()
                if chunk_text:
                    meta = {**base_meta, "section_path": section_path}
                    chunks.append({"text": chunk_text, "metadata": meta})
                pos = end - overlap  # step back by overlap amount
                if pos <= (end - chunk_size):  # safety: avoid infinite loop
                    break
        current_text = ""

    for line in lines:
        if _is_header(line):
            # Flush previous section
            _flush(current_section)
            # Update section tracking
            current_section = line.strip()
            # Add header as prefix to next chunk's text
            current_text = line + "\n"
        else:
            current_text += line + "\n"
            # Flush if accumulated text is large enough (2x chunk_size)
            if len(current_text) > chunk_size * 2:
                _flush(current_section)

    # Flush remaining
    _flush(current_section)

    # Assign IDs
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"{source}_{i}"

    return chunks


class RAGEngine:
    """RAG engine for clinical guidelines using ChromaDB, scoped per expert."""

    def __init__(self, persist_dir: str = "./data/chromadb", guides_dir: str = "./data/guides"):
        self.persist_dir = persist_dir
        self.guides_dir = guides_dir
        self.collection = None
        self._initialized = False

    def initialize(self):
        """Initialize ChromaDB collection."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = client.get_or_create_collection(
                name="clinical_guidelines",
                metadata={"hnsw:space": "cosine"}
            )
            self._initialized = True

            count = self.collection.count()
            console.print(f"[green]RAG ready ({count} chunks) [dir: {self.persist_dir}][/green]")

            if count == 0:
                console.print("[yellow]No guidelines loaded yet.[/yellow]")

        except Exception as e:
            console.print(f"[yellow]RAG not available: {e}[/yellow]")

    def search(self, query: str, n_results: int = 3) -> str:
        """Search guidelines for relevant content. Returns formatted context string."""
        if not self._initialized or self.collection is None:
            return "(Guias clinicas no disponibles)"

        if self.collection.count() == 0:
            return "(No hay guias indexadas)"

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"][0]:
                return "(Sin resultados relevantes)"

            context_parts = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                source = meta.get("source", "Guia clinica")
                section = meta.get("section_path", "")
                society = meta.get("society", "")
                label = source
                if section:
                    label += f" > {section}"
                if society:
                    label = f"[{society}] {label}"
                context_parts.append(f"[{label}]: {doc[:500]}")

            return "\n\n".join(context_parts)

        except Exception as e:
            return f"(Error consultando guias: {e})"

    def search_detailed(self, query: str, n_results: int = 5,
                         where: dict | None = None) -> list[dict]:
        """Search with full metadata and scores. For admin/debugging."""
        if not self._initialized or self.collection is None:
            return []

        try:
            query_kwargs = dict(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            if where:
                query_kwargs["where"] = where
            results = self.collection.query(**query_kwargs)

            hits = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "text": doc[:500],
                    "source": meta.get("source", ""),
                    "section_path": meta.get("section_path", ""),
                    "society": meta.get("society", ""),
                    "year": meta.get("year", ""),
                    "distance": round(dist, 4),
                })
            return hits

        except Exception:
            return []

    def load_text(self, text: str, source: str, chunk_size: int = 500,
                  overlap: int = 100, category: str = ""):
        """Load a text document into the RAG store with clinical chunking."""
        if not self._initialized:
            self.initialize()

        if self.collection is None:
            console.print("[red]Cannot load: RAG not initialized[/red]")
            return

        chunk_dicts = chunk_clinical_text(
            text, source, category=category,
            chunk_size=chunk_size, overlap=overlap,
        )

        if not chunk_dicts:
            return

        ids = [c["id"] for c in chunk_dicts]
        documents = [c["text"] for c in chunk_dicts]
        metadatas = [c["metadata"] for c in chunk_dicts]

        # ChromaDB batch limit is ~5000
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                ids=ids[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        console.print(f"[green]Loaded '{source}': {len(chunk_dicts)} chunks[/green]")

    def list_guidelines(self) -> list[dict]:
        """List all unique guideline sources with metadata."""
        if not self._initialized or self.collection is None:
            return []
        total = self.collection.count()
        if total == 0:
            return []
        sources = {}
        offset = 0
        batch_size = 5000
        while offset < total:
            batch = self.collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            for meta in batch["metadatas"]:
                src = meta.get("source", "unknown")
                cat = meta.get("category", "")
                if src not in sources:
                    sources[src] = {"source": src, "category": cat, "chunks": 0, "has_file": False}
                sources[src]["chunks"] += 1
            fetched = len(batch["metadatas"])
            if fetched == 0:
                break
            offset += fetched
        guides_path = Path(self.guides_dir)
        if guides_path.exists():
            all_files = {f.stem: True for f in guides_path.rglob("*") if f.is_file()}
            for entry in sources.values():
                entry["has_file"] = entry["source"] in all_files
        return sorted(sources.values(), key=lambda x: x["source"])

    def delete_guideline(self, source: str) -> int:
        """Delete all chunks for a given guideline source. Returns count deleted."""
        if not self._initialized or self.collection is None:
            return 0
        try:
            matching = self.collection.get(
                where={"source": source},
                include=[],
            )
            ids_to_delete = matching["ids"]
        except Exception:
            matching = self.collection.get(
                where={"source": source},
                include=[],
            )
            ids_to_delete = matching.get("ids", [])
        if ids_to_delete:
            batch_size = 500
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                self.collection.delete(ids=batch)
        return len(ids_to_delete)

    def get_total_count(self) -> int:
        """Get total number of document chunks."""
        if not self._initialized or self.collection is None:
            return 0
        return self.collection.count()

    def reload_all(self):
        """Delete all documents and reload from the expert's guides directory."""
        if not self._initialized or self.collection is None:
            return
        total = self.collection.count()
        while total > 0:
            batch = self.collection.get(limit=500, include=[])
            if not batch["ids"]:
                break
            self.collection.delete(ids=batch["ids"])
            total = self.collection.count()
        from load_guidelines import load_from_directory
        load_from_directory(self.guides_dir, self)


# ─────────────────────────────────────────────
# Per-expert RAG cache
# ─────────────────────────────────────────────

_rag_cache: dict[str, RAGEngine] = {}


def get_rag_for_expert(slug: str) -> RAGEngine:
    """Get or create a RAGEngine for a specific expert."""
    if slug in _rag_cache:
        return _rag_cache[slug]

    expert_dir = Path(f"data/experts/{slug}")
    persist_dir = str(expert_dir / "chromadb")
    guides_dir = str(expert_dir / "guides")

    (expert_dir / "chromadb").mkdir(parents=True, exist_ok=True)
    (expert_dir / "guides").mkdir(parents=True, exist_ok=True)

    rag = RAGEngine(persist_dir=persist_dir, guides_dir=guides_dir)
    rag.initialize()
    _rag_cache[slug] = rag
    return rag
