# MedExpert Admin - Expert Management & Distribution

Service provider tool for managing medical experts, clinical guidelines (RAG), client licensing, and distribution. Runs on a Mac or server and pushes configuration, guidelines, and licenses to client devices via Tailscale.

Part of the [MedExpert](https://github.com/jmfraga/MedExpert) ecosystem, split into:
- **[medexpert-client](https://github.com/jmfraga/medexpert-client)** — Doctor-facing app with live transcription + AI consultation
- **[medexpert-admin](https://github.com/jmfraga/medexpert-admin)** (this repo) — Admin service for expert management, RAG indexing, and client distribution

## Features

- **Expert management** — Create and configure medical specialties with custom system prompts and icons
- **Guidelines management** — Upload PDFs/text, index into ChromaDB (RAG), manage per expert
- **Web scraping** — 3-tier scraper (public, monitor, authenticated) with multi-level crawler
  - NCCN auto-login and guideline crawling
  - PDF priority downloading
  - URL pattern matching with depth control
- **Client management** — Register client devices, assign experts, manage plans
- **License generation** — Create license.json and config.json for client devices
- **Distribution** — Push ChromaDB, config, and licenses to clients via rsync/scp over Tailscale
- **LLM configuration** — Set default provider and model for client licenses
- **Dashboard** — Overview of experts, clients, guidelines, and distribution status

## Available LLM Models

| Provider | Models |
|---|---|
| Anthropic | Opus 4.6, Sonnet 4.6, Sonnet 4, Haiku 4.5 |
| OpenAI | GPT-5.1, GPT-4.1, GPT-4.1 Mini, GPT-4.1 Nano |

## Quick Start

```bash
git clone https://github.com/jmfraga/medexpert-admin.git
cd medexpert-admin
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # Add your API keys
python app.py
```

Open http://localhost:8080

## Structure

```
medexpert-admin/
├── app.py                 # FastAPI main app (dashboard, experts, clients, config)
├── database.py            # SQLite: experts, clients, web_sources, settings
├── rag_engine.py          # RAG engine (read/write) with ChromaDB
├── license_server.py      # License & config generation for clients
├── distributor.py         # Push ChromaDB/config/license via rsync/scp
├── web_scraper.py         # 3-tier web scraper + multi-level crawler
├── load_guidelines.py     # Load PDF/TXT guidelines into ChromaDB
├── utils.py               # Shared utilities
├── templates/
│   ├── base.html          # Admin theme (amber accent)
│   ├── dashboard.html     # Overview with expert/client cards
│   ├── experts.html       # Expert CRUD
│   ├── guidelines.html    # Per-expert guidelines + web sources
│   ├── clients.html       # Client management + distribution
│   └── config.html        # API keys, LLM model config, system info
└── data/
    ├── medexpert_admin.db # SQLite database
    ├── clients/           # Generated license/config per client
    └── experts/
        └── <specialty>/
            ├── chromadb/  # Vector DB (indexed here, pushed to clients)
            └── guides/    # Source guideline documents
```

## Configuration

Environment variables (`.env`):

| Variable | Default | Description |
|---|---|---|
| `ADMIN_PORT` | `8080` | Web server port |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (injected into client licenses) |
| `OPENAI_API_KEY` | — | OpenAI API key (injected into client licenses) |

Additional settings (default LLM provider/model) are configured from the web UI and persisted in the database.

## Client Distribution Workflow

1. **Create expert** with system prompt and guidelines
2. **Index guidelines** into ChromaDB (upload PDFs or fetch from web sources)
3. **Register client** device with hostname and Tailscale IP
4. **Assign experts** to the client
5. **Push** ChromaDB, config, and license to the client via Tailscale

```
Admin (Mac/Server)                    Client (Raspberry Pi)
┌─────────────┐                      ┌─────────────┐
│ ChromaDB    │──── rsync/scp ──────>│ ChromaDB    │
│ (read/write)│    (Tailscale)       │ (read-only) │
│             │                      │             │
│ config.json │──── scp ────────────>│ config.json │
│ license.json│──── scp ────────────>│ license.json│
└─────────────┘                      └─────────────┘
```

## Disclaimer

Clinical decision **support** tool. Does NOT replace clinical judgment.

## License

This project is licensed under the **Business Source License 1.1**.

- **Free to use** for development, testing, and internal purposes
- **Source code is publicly available**
- **Commercial use** as a competing service requires a commercial license
- **Converts to MIT License** on February 23, 2030

For commercial licensing inquiries: [jmfraga@emergencias.com.mx](mailto:jmfraga@emergencias.com.mx)

Full license text: [LICENSE](./LICENSE)
