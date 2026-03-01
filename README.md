# SHL Assessment Recommendation Engine

AI-powered recommendation system that maps hiring queries to relevant SHL assessments using scraping, embeddings, vector search, and LLM-assisted ranking.

## What This Project Delivers

- Catalog ingestion from SHL Individual Test Solutions (`type=1`)
- Data cleaning, deduplication, and validation pipeline
- Embedding + Pinecone vector search retrieval
- LLM-assisted RAG recommendation engine
- FastAPI backend (`/`, `/health`, `/recommend`)
- React frontend (Vite) for interactive recommendations
- Evaluation framework (Mean Recall@K, Precision@K, F1@K)
- CSV prediction generation for submission format

## Repository Structure

- `src/data_pipeline/`: scraping, schema, processing, validation
- `src/recommendation/`: embeddings, vector DB, retrieval, RAG, recommender
- `src/api/`: FastAPI service
- `src/evaluation/`: metrics and evaluator
- `frontend/`: React app (Vite)
- `scripts/`: runnable workflows (scrape, evaluate, API, frontend, checks)
- `docs/`: setup guides and submission docs
- `tests/`: unit and integration tests
- `data/`: scraped/processed datasets and backups

## Prerequisites

- Python 3.9+
- Node.js 18+
- `npm`
- Chrome + ChromeDriver (for Selenium scraping)
- API keys:
  - `PINECONE_API_KEY`
  - `GROQ_API_KEY`

## Setup

```bash
source myenv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set required values in `.env`:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME=shl-assessments`
- `GROQ_API_KEY`
- `LLM_MODEL=llama-3.3-70b-versatile`
- `API_PORT=8000`
- `FRONTEND_PORT=8501`
- `API_BASE_URL=http://localhost:8000`
- Optional low-memory mode for small cloud instances:
  - `EMBEDDING_BACKEND=hf_inference`
  - `HF_API_KEY=<your_huggingface_token>`
  - `EMBEDDING_DIMENSION=384`

## How To Use (End-to-End)

### 1. Validate data and index state

```bash
source myenv/bin/activate
python scripts/validate_catalog_quality.py --catalog data/processed_catalog.json
```

Expected:
- `Overall valid: True`
- `count >= 377`

### 2. Run API

```bash
API_RELOAD=false python scripts/run_api.py
```

API endpoints:
- Root: `http://localhost:8000/`
- Health: `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

### 3. Run React frontend

Open a new terminal:

```bash
source myenv/bin/activate
python scripts/run_frontend.py
```

Frontend URL:
- `http://localhost:8501`

### 4. Verify API behavior

Open another terminal:

```bash
source myenv/bin/activate
python scripts/test_api.py
```

Expected:
- Root endpoint: passed
- Health check: passed
- Recommendations: passed
- Error handling: passed

### 5. Use the application

In frontend:
- Enter/pick a hiring query
- Submit recommendation request
- Review recommendations table
- Open assessment URLs directly
- Use Developer Controls only for diagnostics (`/`, `/health`, config)

## Evaluation Commands

```bash
# Train-set evaluation
LLM_MODEL=llama-3.3-70b-versatile python scripts/evaluate.py --iteration baseline

# Generate test-set submission CSV
LLM_MODEL=llama-3.3-70b-versatile python scripts/generate_test_predictions.py --output test_predictions.csv
```

## Testing

```bash
pytest -q
```

## Deployment

### Docker

```bash
docker compose up --build
```

### Render

`render.yaml` is included for API + frontend services.

For Render free tier (512MB RAM), use remote embeddings to avoid loading local transformer weights in API memory:

- Set `EMBEDDING_BACKEND=hf_inference`
- Set `HF_API_KEY` in Render secret environment variables
- Keep `EMBEDDING_DIMENSION=384` for `sentence-transformers/all-MiniLM-L6-v2`

The API now initializes the recommendation engine lazily on first `/recommend` request to reduce cold-start memory spikes.

## Final Submission Artifacts

Prepare and verify:

- Public API URL
- Public frontend URL
- Repository URL
- `test_predictions.csv` (`query,assessment_url`)
- 2-page approach document PDF (from `docs/APPROACH_DOCUMENT.md`)

Run local readiness check:

```bash
python scripts/final_readiness_check.py --catalog data/processed_catalog.json --predictions test_predictions.csv
```

## Notes

- External deployment/public URL checks are manual.
- Frontend is React (not Streamlit).
