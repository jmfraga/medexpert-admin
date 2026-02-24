"""
MedExpert Admin - RAG Engine
Full read/write RAG engine for clinical guidelines using ChromaDB.
Manages indexing, searching, and guideline lifecycle per expert.
"""

from pathlib import Path
from rich.console import Console

console = Console()


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
        """Search guidelines for relevant content."""
        if not self._initialized or self.collection is None:
            return "(Guias clinicas no disponibles)"

        if self.collection.count() == 0:
            return "(No hay guias indexadas)"

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
            )

            if not results["documents"][0]:
                return "(Sin resultados relevantes)"

            context_parts = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                source = meta.get("source", "Guia clinica")
                context_parts.append(f"[{source}]: {doc[:500]}")

            return "\n\n".join(context_parts)

        except Exception as e:
            return f"(Error consultando guias: {e})"

    def load_text(self, text: str, source: str, chunk_size: int = 500, category: str = ""):
        """Load a text document into the RAG store, chunked."""
        if not self._initialized:
            self.initialize()

        if self.collection is None:
            console.print("[red]Cannot load: RAG not initialized[/red]")
            return

        chunks = []
        current_chunk = ""

        for paragraph in text.split("\n"):
            if len(current_chunk) + len(paragraph) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n" + paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "category": category} for _ in chunks]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        console.print(f"[green]Loaded '{source}': {len(chunks)} chunks[/green]")

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
