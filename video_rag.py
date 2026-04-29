from uuid import uuid4
from pathlib import Path
import re

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resource/vectorstore_video"
COLLECTION_NAME = "video_transcripts"

LLM_CONTEXT_TOKENS   = 4096
MAX_ANSWER_TOKENS    = 500
SYSTEM_PROMPT_TOKENS = 200
CONTEXT_BUDGET       = LLM_CONTEXT_TOKENS - MAX_ANSWER_TOKENS - SYSTEM_PROMPT_TOKENS

CHARS_PER_TOKEN = 4
RETRIEVAL_K = 10

llm          = None
vector_store = None


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


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript using the new youtube-transcript-api v1.x API.
    Tries English first, then falls back to any available language.
    """
    ytt_api = YouTubeTranscriptApi()

    try:
        # Try fetching English transcript directly
        fetched = ytt_api.fetch(video_id)
    except Exception:
        # Fall back to listing all available transcripts and picking the first
        transcript_list = ytt_api.list(video_id)
        transcript = next(iter(transcript_list))  # get first available
        fetched = transcript.fetch()

    # New API returns a FetchedTranscript object — join snippet texts
    full_text = " ".join(snippet.text for snippet in fetched.snippets)
    return full_text


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


def process_videos(urls: list):
    """
    Fetch transcripts from YouTube URLs and index them into the vector store.
    """
    print("Initializing components …")
    initialize_components()
    vector_store.reset_collection()

    print("Fetching transcripts …")
    all_docs = []
    for url in urls:
        try:
            video_id = extract_video_id(url)
            print(f"  Fetching transcript for video ID: {video_id}")
            transcript_text = fetch_transcript(video_id)
            print(f"  Got {len(transcript_text)} characters of transcript")

            doc = Document(
                page_content=transcript_text,
                metadata={"source": url, "video_id": video_id},
            )
            all_docs.append(doc)
        except Exception as e:
            print(f"  Warning: Could not fetch transcript for {url}: {e}")
            continue

    if not all_docs:
        raise ValueError(
            "No transcripts could be fetched. "
            "Ensure the videos have captions enabled and the URLs are valid."
        )

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
        raise RuntimeError("Vector DB not initialized. Call process_videos() first.")

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
        "video transcript context provided. If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer.strip(), sources