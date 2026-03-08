from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.embedder import embedder
    from app.cache import SemanticCache

    app.state.embedder = embedder
    app.state.cache = SemanticCache(
        threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
    )

    # these two can fail if scripts haven't been run yet
    try:
        from app.cluster import ClusterPredictor
        app.state.clusters = ClusterPredictor()
    except RuntimeError as e:
        app.state.clusters = None
        app.state.cluster_err = str(e)

    try:
        from app.vector_store import VectorStore
        app.state.vs = VectorStore()
    except RuntimeError as e:
        app.state.vs = None
        app.state.vs_err = str(e)

    yield


app = FastAPI(title="FuzzyFind", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query(body: QueryRequest, req: Request):
    if not body.query.strip():
        raise HTTPException(400, "query is empty")

    if req.app.state.clusters is None:
        raise HTTPException(503, f"models not loaded: {req.app.state.cluster_err}")
    if req.app.state.vs is None:
        raise HTTPException(503, f"vector store not loaded: {req.app.state.vs_err}")

    emb = req.app.state.embedder.encode(body.query)
    membership, cluster = req.app.state.clusters.predict(emb)
    cached, score = req.app.state.cache.lookup(emb, cluster)

    if cached:
        return {
            "query": body.query,
            "cache_hit": True,
            "matched_query": cached.query,
            "similarity_score": round(score, 4),
            "result": cached.result,
            "dominant_cluster": cluster,
            "cluster_keywords": req.app.state.clusters.keywords_for(cluster)
        }

    result = req.app.state.vs.search(emb, n=5)
    req.app.state.cache.store(body.query, emb, result, cluster, membership)

    return {
        "query": body.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster,
        "cluster_keywords": req.app.state.clusters.keywords_for(cluster)
    }


@app.get("/cache/stats")
async def cache_stats(req: Request):
    return req.app.state.cache.stats()


@app.delete("/cache")
async def clear_cache(req: Request):
    req.app.state.cache.flush()
    return {"status": "cleared"}


@app.get("/health")
async def health(req: Request):
    return {
        "models_loaded": req.app.state.clusters is not None,
        "vectordb_loaded": req.app.state.vs is not None,
        "threshold": req.app.state.cache.threshold
    }
