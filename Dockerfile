FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

# Use backend_api by default; to use simple_backend, set CMD accordingly
CMD ["gunicorn", "backend_api:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
