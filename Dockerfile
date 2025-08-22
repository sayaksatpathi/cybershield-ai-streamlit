FROM python:3.9-slim

WORKDIR /app

COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

COPY . .

EXPOSE 8501 5000

CMD ["streamlit", "run", "simple_model_trainer.py", "--server.port=8501", "--server.address=0.0.0.0"]
