"""
main.py
-------
FastAPI application entry point.

Endpoints
─────────
GET /                         → health check
GET /recommend?user_id=1      → top-5 item recommendations for a user
GET /similar_items?item_id=101 → top-5 items similar to a given item

The RecommendationEngine is instantiated once at startup (lifespan event)
so expensive matrix computations are performed only once, not per request.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app.recommender import RecommendationEngine


# ── Application state ──────────────────────────────────────────────────────
# Stored in a plain dict so it is accessible inside route handlers without
# using global variables (cleaner for testing / dependency injection).
app_state: Dict[str, Any] = {}


# ── Lifespan: startup / shutdown ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the recommendation engine once at startup.
    FastAPI guarantees this runs before the first request is served.
    """
    print("🚀 Starting up — loading recommendation engine …")
    app_state["engine"] = RecommendationEngine()
    print("✅ Recommendation engine ready.")
    yield
    # Shutdown cleanup (nothing needed here, but the hook is available)
    app_state.clear()
    print("🛑 Shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Based Recommendation System",
    description=(
        "A production-style recommendation API built with FastAPI and "
        "scikit-learn. Implements user-based collaborative filtering and "
        "item-item cosine similarity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper ─────────────────────────────────────────────────────────────────
def get_engine() -> RecommendationEngine:
    """Retrieve the shared RecommendationEngine from application state."""
    engine = app_state.get("engine")
    if engine is None:  # should never happen in normal operation
        raise HTTPException(status_code=503, detail="Recommendation engine not initialised.")
    return engine


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check() -> Dict[str, str]:
    """
    Basic health-check endpoint.
    Returns 200 OK when the service is running.
    """
    return {"status": "ok", "service": "AI Recommendation System"}


@app.get("/recommend", tags=["Recommendations"])
def recommend(
    user_id: int = Query(
        ...,
        description="Integer user ID for which to generate recommendations.",
        example=1,
    )
) -> JSONResponse:
    """
    **User-Based Collaborative Filtering**

    Returns the top-5 items predicted to interest *user_id*, based on the
    preferences of similar users (cosine similarity on the user-item matrix).

    - Items the user has already rated are excluded from results.
    - Predictions are weighted sums of neighbour ratings.

    **Errors**
    - `404` if `user_id` does not exist in the dataset.
    """
    engine = get_engine()
    try:
        recommendations = engine.get_recommendations(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not recommendations:
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "recommendations": [],
                "message": "No new recommendations available (user may have rated all items).",
            },
        )

    return JSONResponse(
        content={
            "user_id": user_id,
            "top_n": len(recommendations),
            "recommendations": recommendations,
        }
    )


@app.get("/similar_items", tags=["Recommendations"])
def similar_items(
    item_id: int = Query(
        ...,
        description="Integer item ID for which to find similar items.",
        example=101,
    )
) -> JSONResponse:
    """
    **Item-Item Cosine Similarity**

    Returns the top-5 items most similar to *item_id*, computed via cosine
    similarity on item rating vectors (columns of the user-item matrix).

    **Errors**
    - `404` if `item_id` does not exist in the dataset.
    """
    engine = get_engine()
    try:
        similars = engine.get_similar_items(item_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return JSONResponse(
        content={
            "item_id": item_id,
            "top_n": len(similars),
            "similar_items": similars,
        }
    )
