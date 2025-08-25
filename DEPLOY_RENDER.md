Render deployment (free tier)

1) Create a new Web Service on Render:
   - Provider: Render
   - Service type: Web Service
   - Environment: Python
   - Name: cybershield-backend
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn backend_api:app --bind 0.0.0.0:$PORT --worker-class=gthread --workers 2 --timeout 120
   - Health check path: /api/health
   - Auto-deploy: ON (connect your GitHub repo)

2) Or use the `render.yaml` manifest in the repo. When you link the repo, Render will auto-detect `render.yaml` and create the service with the above configuration.

3) Set required GitHub repository secrets (if you want CI to trigger deploys):
   - RENDER_API_KEY: (your Render API key)
   - RENDER_SERVICE_ID: (service id from Render. Optional if using API key trigger.)

4) After deploy, set `API_BASE` in your frontends to the service URL (e.g., https://cybershield-backend.onrender.com) or update the `frontend/cybershield_working.html` `API_BASE` constant.

Notes:
- Render has a free tier suitable for testing. It will sleep after inactivity; ensure health checks or paid plan if you need always-on.
- If you prefer I can connect and deploy using your Render API key — provide it as a repo secret and I will trigger a deploy via the workflow.
