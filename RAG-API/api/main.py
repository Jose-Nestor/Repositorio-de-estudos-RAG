# -*- coding: utf-8 -*-
"""
API FastAPI para o sistema RAG.
Endpoints para consulta e ingestão de documentos.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import rag


# Caminho dos documentos (configurável via variável de ambiente)
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "/app/documents")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelos e indexa documentos ao iniciar a API."""
    print("Iniciando RAG-API...")
    rag.carregar_modelos()
    if rag.indexar_documentos(DOCUMENTS_PATH):
        print(f"Documentos indexados: {rag.obter_estado_db()} chunks")
    else:
        print("Nenhum documento .txt encontrado em documents/")
    yield
    print("Encerrando RAG-API...")


app = FastAPI(
    title="RAG-API",
    description="API de Retrieval Augmented Generation para consulta em documentos",
    version="1.0.0",
    lifespan=lifespan,
)


# Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., description="Pergunta a ser respondida pelo RAG")
    top_k: int = Field(default=3, ge=1, le=10, description="Quantidade de chunks para contexto")


class QueryResponse(BaseModel):
    resposta: str
    contexto: str | None = None


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_indexados: int


class HealthResponse(BaseModel):
    status: str
    chunks_no_banco: int


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check para Docker/Kubernetes."""
    return {
        "status": "ok",
        "chunks_no_banco": rag.obter_estado_db(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Consulta o RAG com uma pergunta.
    Retorna a resposta gerada e o contexto usado (trecho).
    """
    try:
        resultado = rag.consultar(req.question, top_k=req.top_k)
        return QueryResponse(
            resposta=resultado["resposta"],
            contexto=resultado.get("contexto"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Reprocessa os documentos da pasta documents/ e reindexa o banco vetorial.
    Útil após adicionar novos arquivos .txt.
    """
    try:
        sucesso = rag.indexar_documentos(DOCUMENTS_PATH)
        chunks = rag.obter_estado_db()
        return IngestResponse(
            success=sucesso,
            message="Documentos reindexados." if sucesso else "Nenhum documento .txt encontrado.",
            chunks_indexados=chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Informações básicas da API."""
    return {
        "api": "RAG-API",
        "docs": "/docs",
        "health": "/health",
    }
