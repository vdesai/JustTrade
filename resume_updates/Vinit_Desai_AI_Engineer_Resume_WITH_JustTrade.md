# VINIT DESAI, MS

Los Gatos, CA (relocating) / Norwalk, CT • (617) 697-0941 • desaivinit8@gmail.com
linkedin.com/in/vinitdesai • github.com/vdesai • www.vamika.net

## AI ENGINEER | PRODUCT BUILDER | QA AUTOMATION LEADER

AI Engineer with 15+ years in software quality automation, now shipping AI products end-to-end. Built and launched multiple AI SaaS apps using FastAPI, Next.js, Supabase, OpenAI, Anthropic Claude, and Replicate APIs — spanning RAG, multimodal (Vision + OCR), agents (MCP), and real-time data. Strong testing mindset, fast iteration, clean UX, and a track record of taking ideas from zero to live users.

## CORE SKILLS

- **AI/ML:** RAG, LLM integration, embeddings, vector DBs (FAISS, Chroma, pgvector), OCR, multimodal (Claude Vision, GPT-4o), agents, tool-use, MCP, LangChain, LangGraph
- **LLM APIs:** OpenAI, Anthropic Claude, Replicate, Google Gemini, Polygon.io
- **Full-Stack:** FastAPI, Next.js 15, React, Streamlit, Supabase, PostgreSQL, SQLite, Stripe, Resend, Tailwind, shadcn/ui
- **Infra / DevOps:** Docker, Render, Vercel, GitHub Actions, CircleCI, OAuth2, REST APIs
- **Automation & QA:** Python, TypeScript/Node.js, Cypress, Playwright, Selenium, Postman, jMeter, CI/CD

## SELECTED AI PROJECTS

### JustTrade – LLM-Driven Event-Driven Trading System ⭐ NEW
End-to-end LLM-powered trading pipeline: real-time SEC EDGAR ingestion, Claude-based 8-K classification across 30+ event types (earnings, M&A, going concern, insider transactions, FDA approvals), shortability-aware backtesting with walk-forward validation, and live Alpaca paper-trading executor with automated risk controls. Classified 9,000+ SEC filings across Q1 2026; caught a "phantom edge" that was actually survivorship bias from unshortable micro-caps, then pivoted to contrarian signals that survived market-regime adjustment.
*Stack: Python, Anthropic Claude API, SEC EDGAR, Alpaca, yfinance, pandas, launchd*

### FamPilot – AI Event Assistant (Image → Google Calendar)
Built a multimodal assistant that extracts event details from flyers/screenshots using Claude Vision (with Tesseract fallback for local dev), lets users confirm/edit fields, and auto-creates Google Calendar events via OAuth2. Used daily by my own family to manage school events. Adaptive OCR routing cuts per-image cost from ~$0.048 to ~$0.003 when Tesseract is available — a real production cost optimization.
*Stack: FastAPI, Claude Sonnet 4.5, Claude Vision, Tesseract, Google Calendar API, SQLite, Render*

### Snap2Steps – Manuals to Step-by-Step Guides (with Audio)
Converts photos/PDFs of product manuals into clear, numbered step-by-step instructions with optional text-to-speech audio. Monorepo with FastAPI backend and Next.js 15 frontend, swappable OCR providers (Google Cloud Vision / Tesseract), Playwright E2E tests.
*Stack: Next.js 15, FastAPI, Google Cloud Vision, OpenAI, Tailwind, shadcn/ui, Docker, Playwright*

### Fusion Meals – AI Meal Planning Platform
Full-stack AI cooking platform generating 7-day meal plans, fusion recipes, grocery lists, and premium AI-chef features. Complete Stripe checkout and subscription flow wired up — payment, entitlement, and tier-gated features all built end-to-end.
*Stack: FastAPI, Streamlit, OpenAI API, Replicate API, Stripe*

### SweepScout – AI Unusual Options Tracker
Real-time web app for tracking and summarizing unusual options activity with AI-generated insights and custom filters for retail traders. Email digests via Resend.
*Stack: FastAPI, Next.js, Supabase, Polygon.io API, Resend API*

### Earnings Redline Copilot – RAG System for SEC Filings
RAG pipeline that ingests, chunks, embeds, and queries SEC 10-K / 10-Q filings with GPT-powered answers and strict citations back to source paragraphs.
*Stack: FastAPI, FAISS, LangChain, OpenAI API, Next.js*

### Gmail AI Search – Personal Email RAG
Local-first RAG over Gmail Takeout mbox: parses, embeds, and indexes email into Chroma for natural-language search and summarization — keeps data on-device.
*Stack: Python, Streamlit, OpenAI Embeddings, ChromaDB, Docker*

### ReceiptZen – AI Expense Tracker MVP
MVP for freelancers to scan receipts with OCR + AI, extract structured data, and simplify tax prep.
*Stack: Python, OCR, React, Streamlit*

## PROFESSIONAL EXPERIENCE

### Sr. Software QA Engineer – Buoy Software | Remote | Nov 2025 – Present
- Testing digital healthcare platform features with focus on stability and privacy compliance.
- Built a Node.js/TypeScript CLI that connects to Jira and Qase via **MCP**, reads ticket details and acceptance criteria, and auto-generates structured test cases (happy path, negative, edge cases) organized by sprint suite — then links the Qase cases back to the Jira ticket.
- Built a custom Cursor Agent Skill using Vercel Labs' agent-browser (headless Chrome) to **execute** generated tests against the live UI — login, navigation, form filling, screenshot capture, and bug filing back to Jira. **Found and filed a real production bug** on its second exploratory run.

### Independent AI Product Development | Remote | Jan 2024 – Oct 2025
- Self-directed two-year sprint shipping AI SaaS products end-to-end — see Selected AI Projects above.
- Full-stack ownership: product scoping, backend (FastAPI), frontend (Next.js/Streamlit), LLM integration (OpenAI, Anthropic Claude), payments (Stripe), deployment (Render, Vercel), and user feedback loops.
- Went deep on RAG, multimodal (Vision + OCR), agents / MCP, and evaluation patterns for production LLM apps.

### Sr. Software QA Engineer – Procore | Remote | Mar 2023 – Dec 2023
- Led automated multi-language regression testing strategy, cutting cycle time from 1.5 months to 3 hours.
- Built robust QA pipelines with Cypress, CircleCI, Tugboat, Backstage, and Sumo Logic.

### Sr. Software QA Engineer – One Door | Boston, MA | Feb 2015 – Feb 2023
- Led QA automation for SaaS platforms over 8 years using Python, Selenium, and Cypress, progressively owning larger portions of the regression suite.
- Contributed to reducing regression test time from 1 month to 2 hours via end-to-end automation.

### Software QA Engineer – Ingenico Mobile Solutions | Boston, MA | Jul 2010 – Feb 2015
- Performed Payment API testing across web, Android, and iOS clients.
- Contributed to load and functional automation frameworks with jMeter and Selenium.

### Software QA Intern – Isobar | Watertown, MA | Jan 2008 – Jan 2010
- Generated and reported test results, collaborating to fix bugs and enhance web app quality.

## EDUCATION

**M.S. Computer Systems Engineering (Software Design)** — Northeastern University, Boston, MA
**B.S. Computer Science** — Visveswaraiah Technological University, India
