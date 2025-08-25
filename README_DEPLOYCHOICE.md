Which backend to deploy?

- `backend_api.py` (recommended): Full-featured universal detector with robust file readers and multiple ML models. Heavier CPU/memory usage but provides best functionality.
- `simple_backend.py`: Lightweight version optimized for faster startup and smaller memory footprint; good for free-host trials or low-traffic demos.

Recommendation: Start with `simple_backend.py` on free hosts (Railway/Fly) to confirm connectivity. If you need full features, scale up to `backend_api.py` on a paid tier or VPS.

To switch Procfile to use `simple_backend.py`, replace its contents with:

```
web: gunicorn simple_backend:app --timeout 120 --worker-class=gthread --workers 2
```

Or when using Docker, set CMD to use `simple_backend:app` instead of `backend_api:app`.
