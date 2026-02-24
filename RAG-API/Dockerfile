# RAG-API - Imagem Docker
# Para GPU: use nvidia/cuda como base e adicione --gpus all no docker run

FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências
COPY requirements.txt .

# Instalar PyTorch CPU (menor tamanho). Para GPU, substitua por: pip install torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Instalar demais dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY api/ ./api/
COPY documents/ ./documents/

# Variáveis de ambiente
ENV DOCUMENTS_PATH=/app/documents
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
