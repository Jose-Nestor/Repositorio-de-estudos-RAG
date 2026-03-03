FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scrape_ufcg_dspace.py .

# O CSV será salvo em /app; monte seu diretório com -v $(pwd):/app
CMD ["python", "scrape_ufcg_dspace.py"]
