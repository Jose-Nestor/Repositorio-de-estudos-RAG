# -*- coding: utf-8 -*-
"""
Módulo RAG - Retrieval Augmented Generation.
Adaptado do Colab: Building RAG From Scratch.
Referência: https://medium.com/red-buffer/building-retrieval-augmented-generation-rag-from-scratch-74c1cd7ae2c0
"""

import os
import uuid
import threading
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai


# Modelos carregados uma única vez (singleton)
_embed_tokenizer = None
_embed_model = None
_device = torch.device("cpu")

_embed_lock = threading.Lock()

# FAISS index (banco vetorial) e textos associados
_faiss_index: faiss.IndexFlatIP | None = None
_faiss_texts: list[str] = []
_faiss_lock = threading.Lock()

# Gemini (LLM em nuvem)
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_gemini_configured = False
_gemini_lock = threading.Lock()

_SUPPORTED_DOC_EXTENSIONS = {
    ".txt",
}


def _configure_gemini() -> None:
    """Configura o cliente Gemini usando variável de ambiente."""
    global _gemini_configured

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY não definido. Configure a chave da API do Gemini no ambiente."
        )
    genai.configure(api_key=api_key)
    _gemini_configured = True


def _chunk_text_by_tokens(text: str, max_tokens: int = 400) -> list[str]:
    """
    Divide um texto em chunks baseados em tokens do modelo de embedding.
    Usa o tokenizer para mapear tokens para offsets de caracteres.
    """
    global _embed_tokenizer

    if not text.strip():
        return []

    # Garante que o tokenizer está carregado
    if _embed_tokenizer is None:
        # Em prática, carregar_modelos já foi chamado no startup.
        # Este fallback evita falhas se for chamado isoladamente.
        carregar_modelos()

    encoding = _embed_tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    offsets = encoding["offset_mapping"]
    if not offsets:
        return []

    chunks: list[str] = []
    start_idx = 0
    n_tokens = len(offsets)

    while start_idx < n_tokens:
        end_idx = min(start_idx + max_tokens, n_tokens)
        char_start = offsets[start_idx][0]
        char_end = offsets[end_idx - 1][1]
        chunk_txt = text[char_start:char_end].strip()
        if chunk_txt:
            chunks.append(chunk_txt)
        start_idx = end_idx

    return chunks


def carregar_modelos():
    """Carrega os modelos de embedding e LLM. Executa uma vez no startup."""
    global _embed_tokenizer, _embed_model, _device

    if _embed_model is not None:
        return

    print("Carregando modelo de Embeddings...")
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    _embed_model = AutoModel.from_pretrained(embed_model_name)
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        dtype = torch.float16
        device_map = "auto"
    else:
        _device = torch.device("cpu")
        dtype = torch.float32
        device_map = None

    _embed_model.to(_device)

    # Configura cliente Gemini uma única vez
    with _gemini_lock:
        if not _gemini_configured:
            print(f"Configurando LLM Gemini: modelo {_GEMINI_MODEL}...")
            _configure_gemini()
            print("Gemini configurado com sucesso!")


def processar_documentos(directory_path: str, chunk_size: int = 400) -> dict:
    """
    Processa arquivos .txt do diretório e gera chunks.
    O chunking é feito por número de tokens (aprox. max `chunk_size` tokens).
    """
    documents: dict[str, dict] = {}
    if not os.path.exists(directory_path):
        return documents

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in _SUPPORTED_DOC_EXTENSIONS:
            # Ignora formatos não suportados
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            continue

        # Normaliza quebras de linha
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Gera chunks baseados em tokens
        chunks = _chunk_text_by_tokens(text, max_tokens=chunk_size)
        if not chunks:
            continue

        doc_id = str(uuid.uuid4())
        chunk_dict: dict[str, dict] = {}
        for chunk_txt in chunks:
            if len(chunk_txt) > 20:
                chunk_id = str(uuid.uuid4())
                chunk_dict[chunk_id] = {
                    "text": chunk_txt,
                    "metadata": {
                        "file": filename,
                        "extension": ext,
                    },
                }

        if chunk_dict:
            documents[doc_id] = chunk_dict

    return documents


def gerar_embeddings(text_list: list) -> np.ndarray:
    """Converte lista de textos em vetores (embeddings)."""
    global _embed_tokenizer, _embed_model, _device

    inputs = _embed_tokenizer(
        text_list, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    if _device.type == "cuda":
        inputs = {k: v.to(_device) for k, v in inputs.items()}

    with _embed_lock:
        with torch.no_grad():
            outputs = _embed_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def criar_banco_vetorial(documents: dict) -> None:
    """
    Converte documentos em vetores e cria o banco de similaridade usando FAISS.
    Usa similaridade de cosseno (via produto interno em vetores normalizados).
    """
    global _faiss_index, _faiss_texts

    text_buffer: list[str] = []
    for doc_id, chunks in documents.items():
        for chunk_id, content in chunks.items():
            text_buffer.append(content["text"])

    if not text_buffer:
        with _faiss_lock:
            _faiss_index = None
            _faiss_texts = []
        return

    # Gera embeddings e normaliza para norma 1 (cosine similarity)
    embeddings = gerar_embeddings(text_buffer).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    with _faiss_lock:
        _faiss_index = index
        _faiss_texts = text_buffer


def buscar_contexto(query: str, top_k: int = 3) -> str:
    """Recupera os top_k chunks mais similares à pergunta usando FAISS."""
    global _faiss_index, _faiss_texts

    with _faiss_lock:
        if _faiss_index is None or not _faiss_texts:
            return ""

        # Embedding da query, normalizado
        query_emb = gerar_embeddings([query]).astype("float32")
        faiss.normalize_L2(query_emb)

        k = min(top_k, len(_faiss_texts))
        scores, indices = _faiss_index.search(query_emb, k)

        idxs = indices[0]
        context_parts: list[str] = []
        for idx in idxs:
            if 0 <= idx < len(_faiss_texts):
                context_parts.append(_faiss_texts[idx])

    return "\n\n".join(context_parts)


def gerar_resposta(query: str, context: str) -> str:
    """Gera resposta usando o contexto recuperado e o LLM."""
    system_prompt = (
        "Você é um Engenheiro Eletrônico sênior e especialista em Verilog. "
        "Use APENAS o contexto fornecido para responder. "
        "Se a resposta não estiver no contexto, responda exatamente: "
        "\"Não encontrei essa informação nos documentos fornecidos.\" "
        "Quando gerar código Verilog, use módulos bem estruturados e 'assign' corretamente."
    )

    context_text = context or "Nenhum contexto disponível."

    prompt = (
        f"{system_prompt}\n\n"
        f"Contexto:\n{context_text}\n\n"
        f"Pergunta do usuário:\n{query}\n\n"
        "Resposta:"
    )

    with _gemini_lock:
        model = genai.GenerativeModel(_GEMINI_MODEL)
        response = model.generate_content(prompt)

    # Normaliza saída em string simples
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    if isinstance(response, str):
        return response.strip()
    return "Não encontrei essa informação nos documentos fornecidos."


def indexar_documentos(documents_path: str) -> bool:
    """
    Processa documentos e atualiza o banco vetorial (FAISS) em memória.
    Retorna True se houve documentos processados.
    """
    docs = processar_documentos(documents_path)
    if docs:
        criar_banco_vetorial(docs)
        return True

    # Nenhum documento -> limpa o índice
    global _faiss_index, _faiss_texts
    with _faiss_lock:
        _faiss_index = None
        _faiss_texts = []
    return False


def consultar(pergunta: str, top_k: int = 3) -> dict:
    """
    Consulta o RAG: busca contexto e gera resposta.
    Retorna dict com 'contexto' e 'resposta'.
    """
    contexto = buscar_contexto(pergunta, top_k=top_k)
    resposta = gerar_resposta(pergunta, contexto)
    return {"contexto": contexto[:500] + "..." if len(contexto) > 500 else contexto, "resposta": resposta}


def obter_estado_db() -> int:
    """Retorna a quantidade de chunks no banco vetorial."""
    global _faiss_texts, _faiss_index
    with _faiss_lock:
        if _faiss_index is None:
            return 0
        # Mantém compatibilidade com a ideia de "quantos chunks estão indexados"
        return len(_faiss_texts)
