import streamlit as st
import tempfile
import os
from pathlib import Path

# Import all RAG modules
import link_rag
import pdf_rag
import video_rag
import image_rag

st.set_page_config(page_title="Multi-Source Research Tool", layout="wide")
st.title("🔍 OmniRAG — Multi-Modal Retrieval-Augmented Generation     Research Assistant")
st.caption("Research across web links, PDFs/DOCX, YouTube videos, and images.")

# ── Mode Tabs ──────────────────────────────────────────────────────────────────
tab_link, tab_pdf, tab_video, tab_image = st.tabs([
    "🌐 Web Links",
    "📄 PDF / DOCX",
    "🎬 YouTube Video",
    "🖼️ Image",
])


# ── Helper: render answer + sources ───────────────────────────────────────────
def render_answer(generate_fn, query: str):
    """Call a generate_answer function and display the result."""
    try:
        answer, sources = generate_fn(query)
        st.header("Answer:")
        st.write(answer)
        if sources:
            st.subheader("Sources:")
            for source in sources.split(","):
                st.write(source.strip())
    except RuntimeError as e:
        st.error(str(e))


# ── Tab 1: Web Links ───────────────────────────────────────────────────────────
with tab_link:
    st.subheader("Process Web URLs")
    col1, col2, col3 = st.columns(3)
    url1 = col1.text_input("URL 1", key="link_url1")
    url2 = col2.text_input("URL 2", key="link_url2")
    url3 = col3.text_input("URL 3", key="link_url3")

    link_placeholder = st.empty()
    if st.button("Process URLs", key="btn_links"):
        urls = [u for u in (url1, url2, url3) if u.strip()]
        if not urls:
            link_placeholder.warning("Provide at least one URL.")
        else:
            with st.spinner("Fetching and indexing URLs…"):
                link_rag.process_urls(urls)
            link_placeholder.success("URLs processed! Ask a question below.")

    link_query = st.text_input("Question about the web pages", key="link_query")
    if link_query:
        render_answer(link_rag.generate_answer, link_query)


# ── Tab 2: PDF / DOCX ─────────────────────────────────────────────────────────
with tab_pdf:
    st.subheader("Upload PDF or DOCX Files")
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    pdf_placeholder = st.empty()
    if st.button("Process Files", key="btn_pdf"):
        if not uploaded_files:
            pdf_placeholder.warning("Upload at least one file.")
        else:
            # Save uploaded files to a temp directory
            tmp_paths = []
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                tmp_path = os.path.join(tmp_dir, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.read())
                tmp_paths.append(tmp_path)

            with st.spinner("Processing documents…"):
                pdf_rag.process_files(tmp_paths)
            pdf_placeholder.success(
                f"Processed {len(tmp_paths)} file(s)! Ask a question below."
            )

    pdf_query = st.text_input("Question about the documents", key="pdf_query")
    if pdf_query:
        render_answer(pdf_rag.generate_answer, pdf_query)


# ── Tab 3: YouTube Video ──────────────────────────────────────────────────────
with tab_video:
    st.subheader("Process YouTube Videos")
    st.caption(
        "Transcripts are fetched automatically. Videos must have captions enabled."
    )
    col1, col2, col3 = st.columns(3)
    vid_url1 = col1.text_input("YouTube URL 1", key="vid_url1")
    vid_url2 = col2.text_input("YouTube URL 2", key="vid_url2")
    vid_url3 = col3.text_input("YouTube URL 3", key="vid_url3")

    video_placeholder = st.empty()
    if st.button("Process Videos", key="btn_videos"):
        vid_urls = [u for u in (vid_url1, vid_url2, vid_url3) if u.strip()]
        if not vid_urls:
            video_placeholder.warning("Provide at least one YouTube URL.")
        else:
            with st.spinner("Fetching transcripts and indexing…"):
                try:
                    video_rag.process_videos(vid_urls)
                    video_placeholder.success(
                        "Videos processed! Ask a question below."
                    )
                except ValueError as e:
                    video_placeholder.error(str(e))

    video_query = st.text_input("Question about the videos", key="video_query")
    if video_query:
        render_answer(video_rag.generate_answer, video_query)


# ── Tab 4: Image ──────────────────────────────────────────────────────────────
with tab_image:
    st.subheader("Upload Images")
    st.caption(
        "Claude Vision extracts all text, data, and visual content from your images."
    )
    uploaded_images = st.file_uploader(
        "Choose image files (JPG, PNG, WEBP, GIF)",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        accept_multiple_files=True,
        key="img_uploader",
    )

    # Show previews
    if uploaded_images:
        preview_cols = st.columns(min(len(uploaded_images), 4))
        for i, img_file in enumerate(uploaded_images):
            preview_cols[i % 4].image(img_file, caption=img_file.name, use_column_width=True)

    image_placeholder = st.empty()
    if st.button("Process Images", key="btn_images"):
        if not uploaded_images:
            image_placeholder.warning("Upload at least one image.")
        else:
            tmp_paths = []
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_images:
                tmp_path = os.path.join(tmp_dir, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.read())
                tmp_paths.append(tmp_path)

            with st.spinner("Extracting image content with Vision model…"):
                try:
                    image_rag.process_images(tmp_paths)
                    image_placeholder.success(
                        f"Processed {len(tmp_paths)} image(s)! Ask a question below."
                    )
                except ValueError as e:
                    image_placeholder.error(str(e))

    image_query = st.text_input("Question about the images", key="image_query")
    if image_query:
        render_answer(image_rag.generate_answer, image_query)
