# Cultural Curator

A Python API that turns cultural memories into event recommendations and answers
questions about past experiences with citations. Built with FastAPI, Mistral and
MongoDB, using a small synthetic event catalogue.

For example, “I loved documentary photography of everyday urban life” becomes an
explicit preference linked to the original quote. Recommendations show how each
preference contributes to the score; answers cite the memories used as context.

## Run with Docker

Requires Docker with Compose. Run these commands from the repository root:

```bash
docker compose up --build -d --wait
docker compose exec api python -m cultural_curator.maintenance --seed-catalogue --apply
```

Open [localhost:8000/docs](http://localhost:8000/docs) to try the API. The default
mode is `demo`: no API key or model calls. It uses deterministic lexical rules
and source excerpts to demonstrate the flow, not model quality.

```bash
curl http://localhost:8000/memories -H 'Content-Type: application/json' \
  -d '{"text":"I loved documentary photography of everyday urban life."}'
curl http://localhost:8000/recommendations -H 'Content-Type: application/json' \
  -d '{"top_k":2}'
curl http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"What documentary photography did I enjoy?"}'
```

On a fresh demo database, **City Witness** ranks first and the answer cites the
saved memory. Each `POST /memories` creates a new record. Seeding the catalogue
again skips identical events.

`docker compose down` stops the services and keeps the MongoDB volume.

## Use Mistral

Copy `.env.example` to `.env` and set:

```dotenv
CULTURAL_CURATOR_MODE=mistral
MISTRAL_API_KEY=your-api-key
MONGODB_DATABASE=cultural_curator
```

Then run the two Docker commands above again to apply the configuration and seed
the separate database. Compose reads `.env` automatically. The chat model defaults
to `mistral-small-latest` (override with `MISTRAL_MODEL`); embeddings use
`mistral-embed`. This mode sends text to Mistral and requires API access.

## How it works

```text
Memory → stored text → preference extraction → quote validation → stored preferences
Preferences + catalogue → embeddings → signed cosine scores → explained recommendations
Question → memory retrieval → optional BM25 reranking → LLM answer → citation validation
Assistant → bounded tool loop → search_memories / recommend_events
```

- Preferences retain their source quotes. Recommendation scores combine positive
  and negative similarities, with an explicit contribution for each preference.
- RAG validates source IDs and verbatim quotes, and abstains when no relevant
  memory is found. Checking a quote does not prove the answer follows from it.
- Mistral and MongoDB calls are asynchronous. The application uses the SDK directly,
  with validated outputs, timeouts and bounded retries for transient model errors.

The API exposes `POST /memories`, `/recommendations`, `/ask` and `/assistant`, plus
`GET /health`. Request and response schemas are available in `/docs`.

## Evaluation

Requires Python 3.12+. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m evals.run
```

The evaluation runs without MongoDB, an API key or network calls. It compares
retrieval with and without BM25 using Recall@k, measures ranking with NDCG@k, and
checks RAG citation and abstention behavior. It uses synthetic fixtures and model
doubles; it does not measure live Mistral quality.

## Scope and limits

The catalogue is fictional. There is no authentication or user isolation; Docker
publishes the API and database on localhost for local use. Embeddings are computed
per request, with no persistent vector index or cache. Memory and preference
writes are not transactional. The assistant's final prose does not have the same
citation validation as `/ask`.
