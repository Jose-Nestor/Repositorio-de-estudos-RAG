# RAG-API

API de **Retrieval Augmented Generation (RAG)** para consulta em documentos, especializada em Verilog e engenharia eletrônica.

Baseado no guia: [Building RAG From Scratch](https://medium.com/red-buffer/building-retrieval-augmented-generation-rag-from-scratch-74c1cd7ae2c0)

## Estrutura do Projeto

```
RAG-API/
├── api/
│   ├── __init__.py
│   ├── main.py      # FastAPI + endpoints
│   └── rag.py       # Lógica RAG (chunking, embeddings, retrieval)
├── documents/       # Arquivos .txt temporários (upload aqui)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (para rodar em container)

## Uso

### 1. Adicionar documentos

Coloque arquivos `.txt` na pasta `documents/`:

```bash
cp seus_docs.txt RAG-API/documents/
```

### 2. Rodar com Docker (recomendado)

```bash
cd RAG-API
docker compose up --build
```

A API ficará em **http://localhost:8000**

- Documentação interativa: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Rodar localmente (sem Docker)

```bash
cd RAG-API
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Endpoints

| Método | Endpoint  | Descrição                           |
|--------|-----------|-------------------------------------|
| GET    | /         | Info da API                         |
| GET    | /health   | Status e quantidade de chunks       |
| POST   | /query    | Envia pergunta e recebe resposta    |
| POST   | /ingest   | Reindexa documentos (após novos .txt) |

**Exemplo de consulta:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Como descrever uma porta XOR em Verilog?", "top_k": 3}'
```

## Sugestões e Melhorias Futuras

1. **Persistência do banco vetorial** – Salvar embeddings em disco (ex.: FAISS, Chroma) em vez de recriar a cada restart.
2. **Pasta `database/`** – Pode ser usada para armazenar o índice vetorial persistido.
3. **Upload via API** – Endpoint `POST /upload` para receber arquivos diretamente na API.
4. **GPU no Docker** – Descomente as linhas de `deploy` no `docker-compose.yml` e use `nvidia-docker` para acelerar.
5. **Variáveis de ambiente** – Criar `.env.example` com `DOCUMENTS_PATH`, `MODEL_ID`, etc.
6. **Suporte a PDF** – Adicionar `pypdf` ou `unstructured` para processar PDFs além de TXT.

## Observação

O primeiro start pode demorar alguns minutos devido ao download dos modelos (MiniLM + Qwen2.5-1.5B). A imagem Docker usa CPU por padrão; para GPU, ajuste o Dockerfile e o docker-compose conforme indicado acima.
