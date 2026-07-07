from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import inference

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR.parent / "static"

app = FastAPI(title="Used Car Price Predictor")

# Not strictly needed since frontend is served from the same app, but kept
# in case you ever want to call the API from a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    variant_name: str
    km_driven: float = Field(..., ge=0, le=1_000_000)
    model_year: int = Field(..., ge=1985, le=2026)
    owner_type: str
    city_name: str
    color: str


class PredictResponse(BaseModel):
    predicted_price: float
    predicted_price_display: str


def _format_inr(amount: float) -> str:
    amount = max(0, round(amount))
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Crore"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} Lakh"
    return f"₹{amount:,.0f}"


@app.get("/api/brands")
def api_brands():
    return {"brands": inference.get_brands()}


@app.get("/api/models")
def api_models(brand: str):
    models = inference.get_models_for_brand(brand)
    if not models:
        raise HTTPException(status_code=404, detail="No models found for this brand")
    return {"models": models}


@app.get("/api/variants")
def api_variants(model: str):
    variants = inference.get_variants_for_model(model)
    if not variants:
        raise HTTPException(status_code=404, detail="No variants found for this model")
    return {"variants": variants}


@app.get("/api/variant-details")
def api_variant_details(variant_name: str):
    details = inference.get_variant_details(variant_name)
    if details is None:
        raise HTTPException(status_code=404, detail="Unknown variant")
    return details


@app.get("/api/cities")
def api_cities():
    return {"cities": inference.get_cities()}


@app.get("/api/colors")
def api_colors():
    return {"colors": inference.get_colors()}


@app.get("/api/owner-types")
def api_owner_types():
    return {"owner_types": inference.get_owner_types()}


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    if inference.VARIANT_BY_NAME.get(req.variant_name) is None:
        raise HTTPException(status_code=400, detail="Unknown variant_name")
    if inference.CITY_BY_NAME.get(req.city_name) is None:
        raise HTTPException(status_code=400, detail="Unknown city_name")

    try:
        price = inference.predict_price(
            variant_name=req.variant_name,
            km_driven=req.km_driven,
            model_year=req.model_year,
            owner_type=req.owner_type,
            city_name=req.city_name,
            color=req.color,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return PredictResponse(
        predicted_price=price,
        predicted_price_display=_format_inr(price),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Serve the frontend (static files) ----
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))
