# Frontend

For deployment simplicity, the frontend lives inside `backend/static/`
(`index.html`, `style.css`, `script.js`) and is served directly by the
FastAPI app in `backend/app/main.py`. This means one deploy = one working
link, with no separate frontend hosting or CORS setup needed.

If you ever want to split it into its own service, just move
`backend/static/*` here and point it at the backend's `/api/*` endpoints.
