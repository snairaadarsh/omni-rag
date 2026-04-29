from uuid import uuid4
from pathlib import Path
import base64

from dotenv import load_dotenv
from groq import Groq
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resource/vectorstore_image"
COLLECTION_NAME = "image_content"

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MEDIA_TYPE_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}

LLM_CONTEXT_TOKENS   = 4096
MAX_ANSWER_TOKENS    = 500
SYSTEM_PROMPT_TOKENS = 200
CONTEXT_BUDGET       = LLM_CONTEXT_TOKENS - MAX_ANSWER_TOKENS - SYSTEM_PROMPT_TOKENS

CHARS_PER_TOKEN = 4
RETRIEVAL_K = 10

llm           = None
vision_client = None
vector_store  = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def select_chunks_within_budget(docs_and_distances: list, token_budget: int) -> list:
    sorted_pairs = sorted(docs_and_distances, key=lambda x: x[1])
    selected, used_tokens = [], 0
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
    global llm, vision_client, vector_store
    if vision_client is None:
        # Reuses GROQ_API_KEY — no extra key needed
        vision_client = Groq()
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


def extract_image_content(image_path: str) -> str:
    """
    Use Groq Llama 4 Scout vision (free) to extract all content from an image.
    Returns a rich textual description suitable for RAG indexing.
    """
    path       = Path(image_path)
    suffix     = path.suffix.lower()
    media_type = MEDIA_TYPE_MAP.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = vision_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Please extract and describe all content from this image in detail. "
                            "Include:\n"
                            "1. All visible text (exact transcription)\n"
                            "2. Any data, numbers, statistics, or figures\n"
                            "3. Charts, tables, or graphs (describe structure and values)\n"
                            "4. Key visual elements, objects, people, or scenes\n"
                            "5. Any labels, captions, or annotations\n\n"
                            "Be thorough and specific so this content can be searched and queried later."
                        ),
                    },
                ],
            }
        ],
        max_tokens=2048,
    )
    return response.choices[0].message.content


def process_images(image_paths: list):
    """
    Extract content from images using Groq Vision and index into vector store.
    image_paths: list of local image file paths
    """
    print("Initializing components …")
    initialize_components()
    vector_store.reset_collection()

    print("Extracting image content with Groq vision model …")
    all_docs = []
    for image_path in image_paths:
        path   = Path(image_path)
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            print(f"  Unsupported image type: {suffix}, skipping {path.name}")
            continue

        try:
            print(f"  Processing: {path.name}")
            extracted_text = extract_image_content(image_path)
            doc = Document(
                page_content=extracted_text,
                metadata={"source": path.name, "type": "image"},
            )
            all_docs.append(doc)
        except Exception as e:
            print(f"  Warning: Could not process {path.name}: {e}")
            continue

    if not all_docs:
        raise ValueError("No valid images could be processed.")

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
        raise RuntimeError("Vector DB not initialized. Call process_images() first.")

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
        "image content context provided. If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer.strip(), sources
