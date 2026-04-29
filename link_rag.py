from uuid import uuid4
import tempfile
import os

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import requests

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resource/vectorstore"
COLLECTION_NAME = "real_estate"

LLM_CONTEXT_TOKENS   = 4096
MAX_ANSWER_TOKENS    = 500
SYSTEM_PROMPT_TOKENS = 200
CONTEXT_BUDGET       = LLM_CONTEXT_TOKENS - MAX_ANSWER_TOKENS - SYSTEM_PROMPT_TOKENS

CHARS_PER_TOKEN = 4
RETRIEVAL_K = 10

llm = None
vector_store = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def select_chunks_within_budget(docs_and_distances: list, token_budget: int) -> list:
    sorted_pairs = sorted(docs_and_distances, key=lambda x: x[1])
    selected = []
    used_tokens = 0
    for doc, distance in sorted_pairs:
        chunk_tokens = estimate_tokens(doc.page_content)
        if used_tokens + chunk_tokens > token_budget:
            continue
        selected.append((doc, distance))
        used_tokens += chunk_tokens
    return selected


def build_context_from_chunks(selected_pairs: list) -> str:
    parts = []
    for i, (doc, distance) in enumerate(selected_pairs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Chunk {i} | distance={distance:.4f} | source={source}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=ef,
        )


def process_urls(urls: list):
    # Plain function — no 'yield', not a generator
    print("Initializing components …")
    initialize_components()
    vector_store.reset_collection()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (Chrome/120.0.0.0 Safari/537.36)"
        )
    }

    print("Loading data …")
    all_docs = []
    for url in urls:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(response.text)
            tmp_path = f.name
        try:
            loader = UnstructuredHTMLLoader(tmp_path)
            docs = loader.load()
        finally:
            os.unlink(tmp_path)

        for doc in docs:
            doc.metadata["source"] = url
        all_docs.extend(docs)

    print("Splitting text …")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = text_splitter.split_documents(all_docs)

    print(f"Adding {len(docs)} chunks to vector store …")
    uuids = [str(uuid4()) for _ in docs]
    vector_store.add_documents(docs, ids=uuids)
    print("Done.")


def generate_answer(query: str) -> tuple:
    if vector_store is None:
        raise RuntimeError("Vector DB not initialized. Call process_urls() first.")

    docs_and_distances = vector_store.similarity_search_with_score(query, k=RETRIEVAL_K)

    if not docs_and_distances:
        return "No relevant information found.", ""

    selected = select_chunks_within_budget(docs_and_distances, CONTEXT_BUDGET)

    if not selected:
        best_doc, best_dist = min(docs_and_distances, key=lambda x: x[1])
        selected = [(best_doc, best_dist)]

    total_tokens = sum(estimate_tokens(d.page_content) for d, _ in selected)
    print(
        f"\n[Chunk prioritization] "
        f"retrieved={len(docs_and_distances)}, "
        f"selected={len(selected)}, "
        f"estimated_tokens={total_tokens}/{CONTEXT_BUDGET}"
    )
    for i, (doc, dist) in enumerate(selected, 1):
        src = doc.metadata.get("source", "?")
        tok = estimate_tokens(doc.page_content)
        print(f"  [{i}] distance={dist:.4f}  tokens≈{tok:4d}  source={src}")

    context = build_context_from_chunks(selected)
    sources = ", ".join(
        {doc.metadata.get("source", "unknown") for doc, _ in selected}
    )

    prompt = (
        "You are a helpful assistant. Answer the question below using ONLY the "
        "context provided. If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer.strip(), sources


if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Mortgage_loan",
        "https://en.wikipedia.org/wiki/Federal_Reserve",
        "https://www.freddiemac.com/pmms",
    ]

    process_urls(urls)

    answer, sources = generate_answer(
        "Tell me what was the 30 year fixed mortgage rate along with date?"
    )
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources:\n{sources}")



