# -*- coding: utf-8 -*-
"""
Módulo RAG - Retrieval Augmented Generation.
Adaptado do Colab: Building RAG From Scratch.
Referência: https://medium.com/red-buffer/building-retrieval-augmented-generation-rag-from-scratch-74c1cd7ae2c0
"""

import os
import uuid
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


# Modelos carregados uma única vez (singleton)
_embed_tokenizer = None
_embed_model = None
_llm = None
_vector_db = []


def carregar_modelos():
    """Carrega os modelos de embedding e LLM. Executa uma vez no startup."""
    global _embed_tokenizer, _embed_model, _llm

    if _embed_model is not None:
        return

    print("Carregando modelo de Embeddings...")
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    _embed_model = AutoModel.from_pretrained(embed_model_name)

    print("Carregando LLM (Qwen2.5-1.5B)...")
    llm_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )
    _llm = HuggingFacePipeline(pipeline=pipe)
    print("Modelos carregados com sucesso!")


def processar_documentos(directory_path: str, chunk_size: int = 400) -> dict:
    """
    Processa arquivos .txt do diretório e gera chunks.
    Chunking: divide o texto em pedaços para não sobrecarregar o contexto da IA.
    """
    documents = {}
    if not os.path.exists(directory_path):
        return documents

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Erro ao ler {filename}: {e}")
                continue

            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

            doc_id = str(uuid.uuid4())
            chunk_dict = {}
            for chunk_txt in chunks:
                if len(chunk_txt) > 20:
                    chunk_id = str(uuid.uuid4())
                    chunk_dict[chunk_id] = {"text": chunk_txt, "metadata": {"file": filename}}
            documents[doc_id] = chunk_dict

    return documents


def gerar_embeddings(text_list: list) -> np.ndarray:
    """Converte lista de textos em vetores (embeddings)."""
    global _embed_tokenizer, _embed_model

    inputs = _embed_tokenizer(
        text_list, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    if torch.cuda.is_available():
        _embed_model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def criar_banco_vetorial(documents: dict) -> list:
    """
    Converte documentos em vetores e cria o banco de similaridade.
    Usa similaridade de cosseno para recuperação.
    """
    vector_db = []
    text_buffer = []

    for doc_id, chunks in documents.items():
        for chunk_id, content in chunks.items():
            text_buffer.append(content["text"])

    if text_buffer:
        embeddings = gerar_embeddings(text_buffer)
        for i, emb in enumerate(embeddings):
            vector_db.append({"embedding": emb, "text": text_buffer[i]})

    return vector_db


def buscar_contexto(query: str, vector_db: list, top_k: int = 3) -> str:
    """Recupera os top_k chunks mais similares à pergunta (similaridade de cosseno)."""
    if not vector_db:
        return ""

    query_emb = gerar_embeddings([query])[0]
    scores = []
    for item in vector_db:
        score = np.dot(query_emb, item["embedding"]) / (
            np.linalg.norm(query_emb) * np.linalg.norm(item["embedding"])
        )
        scores.append((score, item["text"]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([item[1] for item in scores[:top_k]])


def gerar_resposta(query: str, context: str) -> str:
    """Gera resposta usando o contexto recuperado e o LLM."""
    global _llm

    template = """<|im_start|>system
Você é um Engenheiro Eletrônico sênior e especialista em Verilog.
Use o contexto abaixo para responder.
Se o usuário pedir código Verilog, use módulos e 'assign' corretamente.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos fornecidos."

Contexto:
{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = prompt | _llm
    return chain.invoke({"context": context or "Nenhum contexto disponível.", "question": query})


def indexar_documentos(documents_path: str) -> bool:
    """
    Processa documentos e atualiza o banco vetorial em memória.
    Retorna True se houve documentos processados.
    """
    global _vector_db

    docs = processar_documentos(documents_path)
    if docs:
        _vector_db = criar_banco_vetorial(docs)
        return True
    _vector_db = []
    return False


def consultar(pergunta: str, top_k: int = 3) -> dict:
    """
    Consulta o RAG: busca contexto e gera resposta.
    Retorna dict com 'contexto' e 'resposta'.
    """
    global _vector_db

    contexto = buscar_contexto(pergunta, _vector_db, top_k=top_k)
    resposta = gerar_resposta(pergunta, contexto)
    return {"contexto": contexto[:500] + "..." if len(contexto) > 500 else contexto, "resposta": resposta}


def obter_estado_db() -> int:
    """Retorna a quantidade de chunks no banco vetorial."""
    global _vector_db
    return len(_vector_db)
