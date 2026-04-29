from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resource/vectorstore_pdf"
COLLECTION_NAME = "pdf_docs"

LLM_CONTEXT_TOKENS = 4096
MAX_ANSWER_TOKENS = 500
SYSTEM_PROMPT_TOKENS = 200
CONTEXT_BUDGET = LLM_CONTEXT_TOKENS - MAX_ANSWER_TOKENS - SYSTEM_PROMPT_TOKENS

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
        page = doc.metadata.get("page", "")
        page_info = f" | page={page}" if page != "" else ""
        parts.append(
            f"[Chunk {i} | distance={distance:.4f} | source={source}{page_info}]\n"
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


def process_files(file_paths: list):
    """
    Process a list of uploaded PDF or DOCX file paths.
    file_paths: list of local file paths (e.g. from st.file_uploader saved to disk)
    """
    print("Initializing components …")
    initialize_components()
    vector_store.reset_collection()

    print("Loading documents …")
    all_docs = []
    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(path))
        else:
            print(f"Unsupported file type: {suffix}, skipping {path.name}")
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No valid documents were loaded. Check file types.")

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
        raise RuntimeError("Vector DB not initialized. Call process_files() first.")

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
