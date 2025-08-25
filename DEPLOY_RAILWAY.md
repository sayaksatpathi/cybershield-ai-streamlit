Deploying CyberShield AI Backend to Railway (quick guide)

Why Railway
- Railway supports quick GitHub-connected deployments and accepts a `Procfile`/Python app out of the box.
- Good for small demo apps and free-tier testing (small usage limits).

Prerequisites
- A GitHub account and the project pushed to a GitHub repository.
- Railway account (https://railway.app). You can sign up with GitHub.

Quick steps (GitHub -> Railway)
1. Push your local repo to GitHub (if not already):

```bash
git init
git add .
git commit -m "Prepare backend for Railway deploy: Procfile, runtime, requirements"
git remote add origin git@github.com:<your-username>/<repo>.git
git push -u origin main
```

2. On Railway:
- Create a new Project -> Deploy from GitHub.
- Select your repository and the branch (main).
- Railway will detect a Python app (Procfile) and use `gunicorn backend_api:app` to run.
- Set the `PORT` environment variable if asked (Railway auto-creates one).

3. After deploy completes, copy the service URL shown by Railway, e.g. `https://my-app.up.railway.app`.

4. Update the frontend `API_BASE` in `frontend/cybershield_working.html` to point to the deployed API base, e.g.: 

```js
const API_BASE = 'https://my-app.up.railway.app/api';
```

5. Open the frontend page (GitHub Pages or local file) and confirm health endpoint responds: 

```bash
curl -sS https://my-app.up.railway.app/api/health | jq
```

Local testing (before deploying)
- Create and activate a virtualenv, install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run locally with Flask dev server (port 5000):

```bash
python backend_api.py
```

- Or run with gunicorn (closer to production):

```bash
gunicorn backend_api:app --bind 0.0.0.0:5000 --workers 2
```

- Set `API_BASE` in `frontend/cybershield_working.html` to `http://localhost:5000/api` and open the HTML in your browser.

Notes & troubleshooting
- If deployment fails because of memory/time limits, try `simple_backend.py` for lighter resource usage and update Procfile to run `gunicorn simple_backend:app`.
- If you prefer Docker/Fly.io instead, I can add a Dockerfile.
- I added `Procfile` and pinned `runtime.txt` and updated `requirements.txt` to include `Flask`, `flask-cors`, and `gunicorn` so Railway/Gunicorn can run the app.

If you want, I can:
- Add a minimal `Dockerfile` for Fly.io or general Docker-based deploy.
- Create a GitHub Actions workflow to auto-deploy on push.
- Help you push the repo to GitHub from here if you provide the remote URL (or I can show exact commands to run locally).
