# Used Car Price Prediction

Predicts the resale price of a used car in India, trained on real CarDekho
listings. Data cleaning, EDA, feature engineering and model selection live
in `notebooks/`; the trained Random Forest model is served through a small
FastAPI backend with a built-in web UI.

**Live app:** _add your deployed link here once live_

## How it works

1. Pick Brand → Model → Variant. The app already knows that variant's
   engine, dimensions, fuel type, etc. from historical listings.
2. Enter the details that make *this* car different from another one of
   the same variant: registration year, kilometers driven, ownership,
   city, and colour.
3. The backend reconstructs the same 69-feature vector the model was
   trained on (frequency encoding for high-cardinality columns, one-hot
   encoding for low-cardinality columns) and returns a predicted price.

## Project structure

```
notebooks/          data cleaning, EDA, model training (source of truth)
data/                raw + processed CSVs used by the notebooks
models/              original full-size model artifacts from training
backend/
  app/
    main.py          FastAPI app + API routes
    inference.py      rebuilds the feature vector and calls the model
    data/            frequency maps + lookup tables generated from data/processed
    model/           compressed model + feature column order used in production
  static/            the frontend (plain HTML/CSS/JS, served by FastAPI)
  requirements.txt   production dependencies
render.yaml          one-click Render deployment config
```

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Deploy

See the deployment guide (Render) — connect this GitHub repo, root
directory `backend`, build command `pip install -r requirements.txt`,
start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
