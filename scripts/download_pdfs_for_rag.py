#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baixa PDFs do DSpace UFCG e extrai texto para o RAG.
Usa o CSV ufcg_ceei_2020_2026.csv e salva .txt em RAG-API/documents/

Variáveis de ambiente:
  LIMIT   - quantidade de itens (default: todos / 1164)
  OFFSET  - índice inicial no CSV (default: 0)
  RESUME  - 1 para pular arquivos já existentes (default: 1)
  DELAY   - segundos entre requisições (default: 2)
"""

import csv
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import io

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://dspace.sti.ufcg.edu.br"
LIMIT = int(os.getenv("LIMIT", "0")) or 0  # 0 = processar todos
OFFSET = int(os.getenv("OFFSET", "0"))
RESUME = os.getenv("RESUME", "1").lower() in ("1", "true", "yes")
DELAY = float(os.getenv("DELAY", "2"))
_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", _SCRIPT_DIR.parent / "RAG-API" / "documents"))
CSV_PATH = Path(os.getenv("CSV_PATH", _SCRIPT_DIR / "ufcg_ceei_2020_2026.csv"))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def extract_handle_id(link: str) -> str | None:
    """Extrai o ID do handle da URL (ex: 36596 de .../handle/riufcg/36596)."""
    match = re.search(r"/handle/[\w-]+/(\d+)(?:$|[?\s/])", link)
    return match.group(1) if match else None


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Remove caracteres inválidos para nome de arquivo."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")[:max_len]
    return name or "documento"


def output_path_for_item(handle_id: str, titulo: str, output_dir: Path) -> Path:
    """Retorna o caminho do .txt para o item (nome único por handle_id)."""
    base = sanitize_filename(titulo)
    path = output_dir / f"{handle_id}_{base}.txt"
    idx = 1
    while path.exists():
        path = output_dir / f"{handle_id}_{base}_{idx}.txt"
        idx += 1
    return path


def already_processed(
    handle_id: str, link: str, output_dir: Path, min_size: int = 100
) -> bool:
    """Verifica se já existe .txt para esse item (para resume)."""
    if not output_dir.exists():
        return False
    for f in output_dir.glob(f"{handle_id}_*.txt"):
        if f.stat().st_size >= min_size:
            return True
    for f in output_dir.glob("*.txt"):
        if f.stat().st_size < min_size:
            continue
        try:
            with open(f, encoding="utf-8", errors="ignore") as fp:
                head = fp.read(2048)
            if f"Link: {link}" in head or f"Link:{link}" in head:
                return True
        except Exception:
            pass
    return False


def find_pdf_url(item_url: str, session: requests.Session) -> str | None:
    """Acessa a página do item e localiza o link do PDF (bitstream)."""
    try:
        r = session.get(item_url, timeout=30, verify=False)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # DSpace: link para bitstream ou arquivo PDF
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # bitstream: /bitstream/riufcg/36596/1/arquivo.pdf
            if "/bitstream/" in href and (
                ".pdf" in href.lower() or "application/pdf" in str(a).lower()
            ):
                if href.startswith("/"):
                    return BASE_URL + href
                return href

        # Fallback: qualquer link .pdf
        for a in soup.find_all("a", href=re.compile(r"\.pdf", re.I)):
            href = a["href"]
            if href.startswith("/"):
                return BASE_URL + href
            if href.startswith("http"):
                return href

    except Exception as e:
        print(f"    Erro ao buscar PDF: {e}")
    return None


def download_pdf(url: str, session: requests.Session) -> bytes | None:
    """Baixa o PDF e retorna o conteúdo."""
    try:
        r = session.get(url, timeout=60, verify=False, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"    Erro ao baixar PDF: {e}")
    return None


def sanitize_text(text: str) -> str:
    """Remove caracteres surrogate e outros inválidos que quebram UTF-8."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    """Extrai texto do PDF."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(sanitize_text(text))
        return "\n\n".join(parts).strip() if parts else None
    except Exception as e:
        print(f"    Erro ao extrair texto: {e}")
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"CSV não encontrado: {CSV_PATH}")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = all_rows[OFFSET:]
    if LIMIT > 0:
        rows = rows[:LIMIT]

    total_csv = len(all_rows)
    print(f"CSV: {total_csv} itens | Processando: {len(rows)} (offset={OFFSET})")
    print(f"Resume: {'sim' if RESUME else 'não'} | Delay: {DELAY}s")
    print(f"Saída: {OUTPUT_DIR}\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    ok = 0
    fail = 0
    skipped = 0

    for i, row in enumerate(rows, OFFSET + 1):
        titulo = row.get("titulo", "").strip()
        link = row.get("link", "").strip()
        autor = row.get("autor", "")

        if not link:
            print(f"[{i}] Sem link")
            fail += 1
            continue

        handle_id = extract_handle_id(link)
        if not handle_id:
            print(f"[{i}] Link inválido (sem handle)")
            fail += 1
            continue

        if RESUME and already_processed(handle_id, link, OUTPUT_DIR):
            print(f"[{i}] Já processado (pulando)")
            skipped += 1
            continue

        print(f"[{i}] {titulo[:60]}...")

        pdf_url = find_pdf_url(link, session)
        if not pdf_url:
            print(f"    Sem PDF encontrado")
            fail += 1
            time.sleep(DELAY)
            continue

        pdf_bytes = download_pdf(pdf_url, session)
        if not pdf_bytes or len(pdf_bytes) < 100:
            print(f"    PDF vazio ou não baixado")
            fail += 1
            time.sleep(DELAY)
            continue

        text = extract_text_from_pdf(pdf_bytes)
        if not text or len(text) < 50:
            print(f"    Texto extraído vazio ou muito curto")
            fail += 1
            time.sleep(DELAY)
            continue

        out_path = output_path_for_item(handle_id, titulo, OUTPUT_DIR)
        header = f"Título: {titulo}\nAutor: {autor}\nLink: {link}\n\n---\n\n"
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(sanitize_text(header) + text)

        print(f"    OK: {out_path.name}")
        ok += 1
        time.sleep(DELAY)

    print(f"\nConcluído: {ok} sucesso, {fail} falha, {skipped} pulados (já existiam)")
    if ok:
        print(f"Arquivos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
