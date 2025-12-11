import os
import io
import json
from typing import List, Optional
import numpy as np
import requests
import streamlit as st
from huggingface_hub import InferenceClient, HfApi
from PIL import Image

# --- Setup API clients ---
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ky3ow/lab7-images"

client = InferenceClient(api_key=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# --- RAG settings ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, efficient
RAG_STORE_FILE = "rag_store.jsonl"  # JSONL persisted in HF dataset
TOP_K = 5  # number of retrieved snippets to include in prompt

# --- Helpers (core) ---
def upload_image_file(file) -> str:
    """Upload a Streamlit UploadedFile to HF dataset and return public URL."""
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file.name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file.name}"

def upload_bytes_as_file(data: bytes, file_name: str) -> str:
    """Upload bytes to HF dataset and return public URL."""
    api.upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=file_name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file_name}"

def list_dataset_image_paths() -> List[str]:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return [p for p in files if p.lower().endswith(exts)]

def to_public_url(path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path_in_repo}"

def run_vqa(image_url: str, prompt: str) -> Optional[str]:
    completion = client.chat.completions.create(
        model="zai-org/GLM-4.1V-9B-Thinking",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    if not completion.choices[0].message.content:
        return None
    return completion.choices[0].message.content.strip()

def generate_image(prompt: str) -> Image.Image:
    return client.text_to_image(
        prompt, model="stabilityai/stable-diffusion-xl-base-1.0"
    )

# --- Embeddings + simple in-memory vector store for RAG ---
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Use HF InferenceClient's feature extraction to get sentence embeddings.
    Returns L2-normalized vectors for cosine similarity via dot product.
    """
    outs = client.feature_extraction(texts, model=EMBED_MODEL)
    arr = np.array(outs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms

def cosine_top_k(query_vec: np.ndarray, doc_mat: np.ndarray, k: int) -> List[int]:
    sims = doc_mat @ query_vec  # cosine if normalized
    if k >= len(sims):
        idx = np.argsort(-sims)
        return idx.tolist()
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()

def ensure_rag_session():
    if "rag_docs" not in st.session_state:
        st.session_state.rag_docs = []  # list of dicts: {id, text, meta}
    if "rag_mat" not in st.session_state:
        st.session_state.rag_mat = None

def _rebuild_matrix():
    ensure_rag_session()
    if not st.session_state.rag_docs:
        st.session_state.rag_mat = None
        return
    texts = [d["text"] for d in st.session_state.rag_docs]
    st.session_state.rag_mat = embed_texts(texts)

def add_doc(text: str, meta: dict):
    ensure_rag_session()
    st.session_state.rag_docs.append(
        {"id": f"d{len(st.session_state.rag_docs)+1}", "text": text, "meta": meta}
    )

def retrieve(query: str, k: int = TOP_K, scope_image: Optional[str] = None) -> List[dict]:
    ensure_rag_session()
    if not st.session_state.rag_docs or st.session_state.rag_mat is None:
        return []
    qv = embed_texts([query])[0]
    M = st.session_state.rag_mat
    # Over-fetch to allow filtering by image scope
    idxs = cosine_top_k(qv, M, min(len(st.session_state.rag_docs), k * 3))
    picked = []
    for i in idxs:
        d = st.session_state.rag_docs[i]
        if scope_image and d["meta"].get("image_path") != scope_image:
            continue
        picked.append(d)
        if len(picked) >= k:
            break
    return picked

def load_rag_corpus():
    """
    Build the in-memory RAG corpus from:
    - Sidecar .txt files stored alongside images (same stem)
    - Persisted rag_store.jsonl lines (notes and prior Q&A)
    """
    ensure_rag_session()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    # 1) Sidecar .txt files
    txts = [p for p in files if p.lower().endswith(".txt")]
    image_paths = list_dataset_image_paths()
    for t in txts:
        try:
            url = to_public_url(t)
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                continue
            stem = os.path.splitext(os.path.basename(t))[0]
            related_img = next(
                (
                    p
                    for p in image_paths
                    if os.path.splitext(os.path.basename(p))[0] == stem
                ),
                None,
            )
            add_doc(text, {"type": "sidecar", "image_path": related_img})
        except Exception:
            pass

    # 2) Persisted JSONL notes/Q&A
    if RAG_STORE_FILE in files:
        try:
            r = requests.get(to_public_url(RAG_STORE_FILE), timeout=10)
            r.raise_for_status()
            for line in r.text.splitlines():
                try:
                    rec = json.loads(line)
                    add_doc(rec["text"], rec.get("meta", {}))
                except Exception:
                    continue
        except Exception:
            pass

    _rebuild_matrix()

def append_jsonl_line(obj: dict):
    """
    Download rag_store.jsonl if it exists, append a JSON line locally,
    then upload the full updated file.
    """
    # Fetch current content if exists
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    except Exception:
        files = []
    existing_bytes = b""
    if RAG_STORE_FILE in files:
        try:
            resp = requests.get(to_public_url(RAG_STORE_FILE), timeout=10)
            if resp.ok:
                existing_bytes = resp.content
        except Exception:
            # If fetch fails, proceed with empty existing content
            existing_bytes = b""

    # Ensure newline at end if existing non-empty
    if existing_bytes and not existing_bytes.endswith(b"\n"):
        existing_bytes += b"\n"

    new_line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    updated = existing_bytes + new_line

    # Upload full file
    api.upload_file(
        path_or_fileobj=io.BytesIO(updated),
        path_in_repo=RAG_STORE_FILE,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="update rag_store.jsonl",
    )

def build_rag_prefix(user_query: str, image_path: Optional[str]) -> str:
    # Retrieve by query scoped to image
    hits_img = retrieve(user_query, k=3, scope_image=image_path)
    # Also retrieve globally to catch general notes
    hits_global = retrieve(user_query, k=2, scope_image=None) if image_path else []
    seen = set()
    snippets = []
    for d in hits_img + hits_global:
        key = d["text"][:64]
        if key in seen:
            continue
        seen.add(key)
        tag = d["meta"].get("type", "ctx")
        snippets.append(f"[{tag}] {d['text']}")
    if not snippets:
        return ""
    joined = "\n".join(snippets[:TOP_K])
    return (
        "Use the following retrieved context if helpful. If irrelevant, ignore it.\n"
        + joined
        + "\n---\n"
    )

def run_vqa_with_rag(image_url: str, image_path: str, prompt: str) -> Optional[str]:
    prefix = build_rag_prefix(prompt, image_path)
    full_prompt = (prefix + prompt).strip()
    return run_vqa(image_url, full_prompt)

# --- Page config ---
st.set_page_config(page_title="HF Gallery + VQA (+ RAG)", layout="wide")
st.title("🧠 HF Dataset Gallery + Visual QA (+ RAG)")

# --- Load RAG corpus once on startup ---
load_rag_corpus()

# --- Sidebar: add images (upload + SDXL) + RAG notes ---
with st.sidebar:
    st.header("Add Images")
    up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    if st.button("Upload to Dataset", use_container_width=True, disabled=up is None):
        with st.spinner("Uploading…"):
            try:
                url = upload_image_file(up)
                st.success("Uploaded!")
                st.code(url)
                st.session_state["gallery_cache_version"] = (
                    st.session_state.get("gallery_cache_version", 0) + 1
                )
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Generate with SDXL")
    sd_prompt = st.text_area("Text prompt", "Astronaut riding a horse")
    out_name = st.text_input("File name (without ext)", "generated_image")
    if st.button("Generate + Upload", use_container_width=True):
        if not sd_prompt.strip():
            st.warning("Enter a prompt")
        else:
            with st.spinner("Generating…"):
                try:
                    img = generate_image(sd_prompt.strip())
                    st.image(img, caption="Preview", width="content")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    png_bytes = buf.getvalue()
                    safe_name = (out_name.strip() or "generated_image") + ".png"
                    url = upload_bytes_as_file(png_bytes, safe_name)
                    st.success("Generated and uploaded!")
                    st.code(url)
                    st.session_state["gallery_cache_version"] = (
                        st.session_state.get("gallery_cache_version", 0) + 1
                    )
                    # Create sidecar .txt with generation prompt and index it
                    sidecar_name = os.path.splitext(safe_name)[0] + ".txt"
                    sidecar_text = f"gen_prompt: {sd_prompt.strip()}"
                    api.upload_file(
                        path_or_fileobj=io.BytesIO(sidecar_text.encode("utf-8")),
                        path_in_repo=sidecar_name,
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        commit_message="add sidecar prompt",
                    )
                    # Update local RAG immediately
                    add_doc(
                        sidecar_text,
                        {"type": "sidecar", "image_path": safe_name},
                    )
                    _rebuild_matrix()
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    st.divider()
    st.subheader("Image notes (RAG)")
    note_txt = st.text_area("Add note for selected image (indexed)", height=80)
    save_note_disabled = "selected_path" not in st.session_state or not st.session_state.get(
        "selected_path"
    )
    if st.button("Save note", use_container_width=True, disabled=save_note_disabled):
        if not st.session_state.get("selected_path"):
            st.warning("Select an image first.")
        elif not note_txt.strip():
            st.warning("Note is empty.")
        else:
            # Add to session index
            add_doc(
                note_txt.strip(),
                {"type": "note", "image_path": st.session_state["selected_path"]},
            )
            _rebuild_matrix()
            # Persist to dataset JSONL
            rec = {
                "text": note_txt.strip(),
                "meta": {
                    "type": "note",
                    "image_path": st.session_state["selected_path"],
                },
            }
            try:
                append_jsonl_line(rec)
                st.success("Note saved and indexed.")
            except Exception as e:
                st.error(f"Failed to persist note: {e}")

# --- Session state ---
if "selected_path" not in st.session_state:
    st.session_state.selected_path = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# --- Top: prompt + Ask button + immediate output, preview on the right ---
colL, colR = st.columns([3, 2])
with colL:
    vqa_prompt = st.text_area(
        "Prompt", "Describe this image in one sentence.", height=80
    )
    ask = st.button("Ask question", type="primary")

    # Handle Ask immediately and render output right here
    if ask:
        if not st.session_state.selected_path:
            st.warning("Please select an image from the gallery below.")
        elif not vqa_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Running VQA…"):
                try:
                    img_path = st.session_state.selected_path
                    img_url = to_public_url(img_path)
                    result = run_vqa_with_rag(img_url, img_path, vqa_prompt.strip())
                    st.session_state.last_answer = result or "(empty)"

                    # Append compact Q&A to RAG and persist
                    if result:
                        qa_text = f"Q: {vqa_prompt.strip()}\nA: {result.strip()}"
                        add_doc(qa_text, {"type": "qa", "image_path": img_path})
                        _rebuild_matrix()
                        rec = {
                            "text": qa_text,
                            "meta": {"type": "qa", "image_path": img_path},
                        }
                        try:
                            append_jsonl_line(rec)
                        except Exception as e:
                            st.warning(f"Could not persist QA: {e}")
                except Exception as e:
                    st.session_state.last_answer = None
                    st.error(f"VQA failed: {e}")

    if st.session_state.last_answer is not None:
        st.markdown("Model Output:")
        st.write(st.session_state.last_answer)

with colR:
    if st.session_state.selected_path:
        st.markdown("Selected image")
        st.image(to_public_url(st.session_state.selected_path), width="content")
    else:
        st.info("Click an image in the gallery to select it.")

# --- Gallery controls ---
st.subheader("Gallery")
gc1, gc2, gc3 = st.columns([1, 1, 2])
with gc1:
    cols = st.slider("Columns", 2, 8, 4)
with gc2:
    refresh = st.button("Refresh")

paths = list_dataset_image_paths()

if not paths:
    st.info("No images in the dataset yet.")
else:
    rows = (len(paths) + cols - 1) // cols
    for r in range(rows):
        row_paths = paths[r * cols : (r + 1) * cols]
        cols_row = st.columns(len(row_paths), gap="small")
        for i, p in enumerate(row_paths):
            with cols_row[i]:
                url = to_public_url(p)
                is_selected = st.session_state.selected_path == p
                st.image(url, width="content")
                if st.button(
                    "Selected" if is_selected else "Select",
                    key=f"sel_{p}",
                    disabled=is_selected,
                    use_container_width=True,
                ):
                    st.session_state.selected_path = p
                    st.rerun()
