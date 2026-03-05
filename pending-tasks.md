# Pending Tasks

## Completed
- [x] Split MedExpert into client + admin (2026-02-24)
- [x] Add MedExpert logo to both projects (2026-02-25)
- [x] Web scraper: Playwright browser support for SPAs (2026-02-25)
- [x] Web scraper: per-depth URL patterns, external domains, min_content_length (2026-02-25)
- [x] IMSS GPC oncologia: 30 GER PDFs indexed (15K chunks) (2026-02-25)
- [x] Clinical chunking: section-aware + overlap + rich metadata (2026-02-25)
- [x] Knowledge Packs: manifest.json, versioning, integrity verification (2026-02-25)
- [x] ESMO: 171 guidelines scraped from Annals of Oncology (120K total chunks) (2026-02-25)
- [x] Authenticated Playwright crawling with Elsevier OAuth2 + CAPTCHA support (2026-02-25)
- [x] VAD auto-calibration + tuning (2026-02-26)
- [x] Glossary system: admin CRUD + push to clients + Whisper initial_prompt (2026-02-26)
- [x] Ticket system: client corrections/bugs/features + admin review (2026-02-26)
- [x] Telegram Bot Sprint 1: bot.py, bot_brain.py, DB tables, handlers (2026-02-26)
- [x] Bilingual RAG search: Spanish + English translation dict, merge & dedup (2026-02-27)
- [x] Source diversification: NCCN/ESMO priority over IMSS in results (2026-02-27)
- [x] Society detection from source strings (_detect_society) (2026-02-27)
- [x] Citations stored in DB (citations_json column) + shown in PDF (2026-02-27)
- [x] LLM benchmark script (llm_benchmark.py): 5 models, 5 cases, PDF output (2026-02-27)
- [x] Tiered Profundizar feature: GPT-OSS 120B (free/basic) / Opus (premium) (2026-02-27)
- [x] Groq provider integration (GPT-OSS 20B as default base model) (2026-02-27)
- [x] Markdown cleanup for Telegram messages (_clean_markdown) (2026-02-27)
- [x] Two-plan pricing: Basico $14.99 / Premium $24.99 USD (2026-02-27)
- [x] PDF export per consultation with full citations (2026-02-27)
- [x] User feedback post-response with 5 options (2026-02-27)
- [x] Disclaimers: legal/medical disclaimers in PDF and Telegram messages (2026-02-27)
- [x] Stripe integration with Checkout Sessions (4 plans) (2026-03-01)
- [x] Stripe webhook at api.medexpert.mx — auto-activates subscriptions (2026-03-01)
- [x] Mercado Pago preapproval (subscriptions) with payer_email (2026-03-01)
- [x] PayPal subscription plans created via API (4 plans) (2026-03-01)
- [x] Email collection: ask before /suscribir, store in bot_users (2026-03-01)
- [x] Annual plans: $149.90/$249.90 USD (10 months = 2 months free) (2026-03-01)
- [x] Bot dashboard in admin: stats, user table, plan management (2026-03-01)
- [x] Cloudflare Tunnel: api.medexpert.mx -> M1:8081 (2026-03-01)
- [x] 3-provider payment: Stripe (USD), MP (MXN), PayPal (USD) (2026-03-01)
- [x] Glossary synonyms: brand→generic drug mapping (137 drugs loaded) (2026-03-02)
- [x] Synonym expansion in RAG search + Whisper initial_prompt (2026-03-02)
- [x] Feedback survey after profundizar (bug fix) (2026-03-02)
- [x] Support tickets: /soporte → DB → admin responds → Telegram notification (2026-03-02)
- [x] Bot commands menu visible on "/" (set_my_commands) (2026-03-02)
- [x] Admin subscription management: cancel, change, notify, send messages (2026-03-02)
- [x] deploy.sh: restart admin + bot, stop overwriting production DB (2026-03-02)
- [x] /cancelar command to abort pending actions (2026-03-02)
- [x] WHISPER_MODEL configurable via .env in client (2026-03-02)

## Telegram Bot

### Sprint 2: Remaining Items
- [x] Clip payment integration (MXN, tarjetas + OXXO) — api.payclip.com/v2/checkout (2026-03-04)
- [x] Smart payment routing: Mexico (MP/Clip MXN) vs Internacional (Stripe/PayPal USD) (2026-03-04)
- [x] Test payment flows end-to-end (Clip sandbox OK, MP/PayPal production-ready) (2026-03-04)
- [x] Admin: manage prices/promotions from UI (2026-03-03)
- [x] Medical verification flow (cedula profesional + INE upload, manual review) (2026-03-03)
- [x] Referral program (processing logic + admin dashboard) (2026-03-03)
- [x] Production switch: test/live toggle in admin config (2026-03-03)
- [x] Make services permanent on M1 (launchd for cloudflared, app.py, bot.py) (2026-03-03)

### Sprint 3: Engagement & Scale
- [x] Broadcast system: admin compose + send to target groups via Telegram (2026-03-04)
- [x] Congress calendar: 5 events seeded, admin CRUD, /congresos bot command, auto-alerts (2026-03-04)
- [x] Text anonymization: regex PII removal (CURP, RFC, email, phone, NSS, cards, addresses, names) integrated into consultation storage (2026-03-04)
- [ ] Migrate GPT-OSS 20B from Groq to local Ollama on Mac Mini M4 Pro
- [x] Analytics dashboard: Chart.js, KPIs, API costs, clinical stats, feedback analysis (2026-03-04)
- [x] LLM tiering: base=GPT-OSS-120B(Groq), deepen=Sonnet(basic,5/d)/Opus(premium,3/d) (2026-03-04)
- [x] Admin auth: cookie-based login + role-based middleware (admin/soporte) (2026-03-04)
- [ ] Admin user management UI: create/edit users, change password, role assignment
- [ ] Clinical metadata extraction: diagnóstico, etapa, detalles clínicos from queries/responses
  - Store as structured fields in bot_consultations (diagnosis, stage, cancer_type, etc.)
  - Use LLM post-processing or keyword extraction to classify
  - Enable segmentation in analytics (profundización by diagnosis, not just specialty)
- [ ] Source preference per user: filter/boost guidelines by society (NCCN, ESMO, IMSS, etc.)
  - ChromaDB where filter by society metadata, or score multiplier
  - Bot: /fuentes command to select preferred sources
  - Client: checkboxes in UI
- [ ] Analytics export: CSV/Excel download of analytics data
- [ ] 30-day data retention/cleanup
- [ ] Rate limiting per user

## Client (B2B / Hospital)
- [ ] Persist transcription on page refresh (don't lose state)
- [ ] Fallback LLM (Groq backup if Anthropic fails)
- [ ] Install torch + Silero VAD (better voice detection)
- [x] RPi 5 + Freenove case assembly + GPIO I2C fix (2026-03-03)
- [ ] Continue piloting in clinical sessions, collect feedback
- [ ] Refine prompts based on clinical feedback
- [ ] Separate audio: browser -> WebSocket -> RPi
- [ ] Black box packaging for enterprise

## Infrastructure
- [ ] Mac Mini M4 Pro setup (arriving ~15 days)
- [ ] Ollama + GPT-OSS 20B local (replace Groq)
- [ ] Monitoring (Prometheus + Grafana)

## Expand Specialties
- [ ] Cardiologia (ACC/AHA, ESC guidelines)
- [ ] IMSS GPC for all specialties
- [ ] Second Telegram bot (@MedExpertCardioBot)

## Research
- [ ] 15 casos clinicos, 3 brazos evaluation
- [ ] Paper: "AI-Powered Clinical Decision Support in LATAM"
- [ ] SMEO 2026 presentation
