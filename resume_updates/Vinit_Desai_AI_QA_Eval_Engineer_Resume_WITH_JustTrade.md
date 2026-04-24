# VINIT DESAI, MS

Los Gatos, CA (relocating) / Norwalk, CT • (617) 697-0941 • desaivinit8@gmail.com
linkedin.com/in/vinitdesai • github.com/vdesai • www.vamika.net

## AI QA / EVAL ENGINEER | LLM OBSERVABILITY | AI PRODUCT ENGINEER

15 years of software QA automation + 2 years shipping production LLM products — a rare combination. I've led regression systems that turned 1.5-month QA cycles into 3 hours, and I've also felt the pain of testing my own AI products: silent model drift between Claude/GPT releases, prompts that pass demos but fail edge cases, RAG answers that cite the wrong chunk. I want to build the eval, observability, and testing infrastructure that AI products actually need — with the firsthand perspective of someone who's had to ship without it.

## CORE SKILLS

- **LLM Evaluation & Testing:** offline evals, LLM-as-judge, regression testing for prompts, golden datasets, hallucination detection, RAG grounding checks, citation validation, structured-output schema validation, A/B prompt experiments, walk-forward validation, out-of-sample testing, survivorship-bias audits
- **QA Automation (15+ years):** Python, TypeScript/Node.js, pytest, Cypress, Playwright, Selenium, Postman, jMeter, CI/CD (CircleCI, GitHub Actions), multi-language regression, load & functional testing
- **AI/ML:** RAG, embeddings, vector DBs (FAISS, Chroma, pgvector), OCR, multimodal (Claude Vision, GPT-4o), agents, tool-use, MCP, LangChain, LangGraph
- **LLM APIs:** OpenAI, Anthropic Claude, Replicate, Google Gemini, Polygon.io
- **Full-Stack:** FastAPI, Next.js 15, React, Streamlit, Supabase, PostgreSQL, SQLite, Stripe
- **Infra / DevOps:** Docker, Render, Vercel, GitHub Actions, CircleCI, OAuth2, REST APIs

## SHIPPED AI PROJECTS (testing & quality angle)

### JustTrade – LLM Classification Pipeline with Rigorous Eval Methodology ⭐ NEW
End-to-end LLM pipeline for SEC 8-K filings with production-grade evaluation practices: prompt anonymization to eliminate look-ahead bias from LLM training data, time-split out-of-sample validation, walk-forward backtesting across rolling windows, market-regime adjustment (SPY-neutralized returns) to isolate event signal from market beta. Classified 9,000+ real filings across Q1 2026; caught a "phantom edge" that turned out to be survivorship bias from unshortable micro-caps — exactly the silent failure mode LLM eval infrastructure exists to catch. Paper-trading executor with risk controls (kill switch, shortability pre-check, ATR-based position sizing) deployed via launchd to run daily against live filings.
*Stack: Python, Anthropic Claude API (Haiku), SEC EDGAR, Alpaca, yfinance, pandas, walk-forward validation*

### FamPilot – Multimodal Event Assistant (Claude Vision → Google Calendar)
Production multimodal pipeline: flyer image → OCR → Claude Sonnet 4.5 → structured event → Google Calendar. Adaptive OCR routing cuts per-image cost from ~$0.048 to ~$0.003 (a real production optimization I had to measure, not a synthetic benchmark). Built test suites for OCR fallback paths, JSON parsing robustness, and time-range edge cases. Used daily by my own family — a tight feedback loop that caught real failure modes.
*Stack: FastAPI, Claude Sonnet 4.5, Claude Vision, Tesseract, Google Calendar API, pytest, SQLite*

### Earnings Redline Copilot – RAG with Citation Validation
RAG over SEC 10-K/10-Q filings with strict citation requirements. Every answer must ground back to a specific source paragraph — the exact failure mode LLM eval platforms exist to catch. Good testbed for ground-truth comparison and retrieval precision metrics.
*Stack: FastAPI, FAISS, LangChain, OpenAI API, Next.js*

### Snap2Steps – Multi-Provider OCR with Playwright E2E Tests
Converts product manuals into step-by-step instructions with optional TTS audio. Monorepo with swappable OCR providers (Google Cloud Vision / Tesseract) and Playwright end-to-end tests covering the full upload → OCR → LLM → audio pipeline.
*Stack: Next.js 15, FastAPI, Google Cloud Vision, OpenAI, Playwright, Docker*

### Fusion Meals – AI Meal Planning Platform
Full-stack AI cooking platform generating 7-day meal plans, fusion recipes, and grocery lists. End-to-end Stripe subscription flow built and tested — the exact kind of long-running, stateful LLM product where silent output quality issues (bad substitutions, missing allergens, inconsistent units) are hard to catch without production evals.
*Stack: FastAPI, Streamlit, OpenAI API, Replicate API, Stripe*

### SweepScout – Real-Time Options Flow with AI Summaries
Real-time web app summarizing unusual options activity with AI-generated insights. Required building guardrails against hallucinated tickers and made-up price levels — a concrete case study in why output validation matters for AI financial products.
*Stack: FastAPI, Next.js, Supabase, Polygon.io API, Resend API*

### Gmail AI Search – Local-First RAG
Local-first RAG over Gmail Takeout mbox using ChromaDB. Built to test whether on-device embeddings could match cloud quality for personal search — privacy-preserving by design.
*Stack: Python, Streamlit, OpenAI Embeddings, ChromaDB, Docker*

## PROFESSIONAL EXPERIENCE

### Sr. Software QA Engineer – Buoy Software | Remote | Nov 2025 – Present
- Testing digital healthcare platform features with focus on stability and privacy compliance.
- Built a Node.js/TypeScript CLI that connects to Jira and Qase via **MCP**, reads ticket details and acceptance criteria, and auto-generates structured test cases (happy path, negative, edge cases) organized by sprint suite — then links the Qase cases back to the Jira ticket.
- Built a custom Cursor Agent Skill using Vercel Labs' agent-browser (headless Chrome) to **execute** generated tests against the live UI — login, navigation, form filling, screenshot capture, and bug filing back to Jira. **Found and filed a real production bug** on its second exploratory run.
- Also ran agent-driven security testing (XSS, SQL injection, IDOR) across search fields and form inputs as part of the same workflow.

### Independent AI Product Development | Remote | Jan 2024 – Oct 2025
- Shipped 7+ AI SaaS products end-to-end — see Shipped AI Projects above.
- Built my own eval harnesses, prompt regression tests, and output validators for each product.
- Went deep on RAG grounding, multimodal OCR failure modes, and model-version drift across Claude and OpenAI releases.

### Sr. Software QA Engineer – Procore | Remote | Mar 2023 – Dec 2023
- Led multi-language regression strategy, cutting cycle time from **1.5 months to 3 hours** — the exact kind of speedup that LLM eval platforms are trying to deliver for prompt/model changes.
- Built QA pipelines with Cypress, CircleCI, Tugboat, Backstage, and Sumo Logic.

### Sr. Software QA Engineer – One Door | Boston, MA | Feb 2015 – Feb 2023
- Led QA automation for SaaS platforms over 8 years with Python, Selenium, and Cypress, progressively owning larger portions of the regression suite as I deepened into automation.
- Contributed to reducing regression test time from **1 month to 2 hours** via end-to-end automation.

### Software QA Engineer – Ingenico Mobile Solutions | Boston, MA | Jul 2010 – Feb 2015
- Payment API testing across web, Android, iOS — high-stakes domain where wrong answers had financial consequences (similar to AI products in regulated domains).
- Contributed to load and functional automation frameworks with jMeter and Selenium.

### Software QA Intern – Isobar | Watertown, MA | Jan 2008 – Jan 2010
- Automated web app testing and reporting.

## EDUCATION

**M.S. Computer Systems Engineering (Software Design)** — Northeastern University, Boston, MA
**B.S. Computer Science** — Visveswaraiah Technological University, India
