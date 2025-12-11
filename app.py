import os
import io
from typing import List
import streamlit as st
from huggingface_hub import InferenceClient, HfApi
from PIL import Image

# --- Setup API clients ---
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ky3ow/lab7-images"

client = InferenceClient(api_key=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# --- Helpers ---
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

def run_vqa(image_url: str, prompt: str) -> str | None:
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

# --- Page config ---
st.set_page_config(page_title="HF Gallery + VQA", layout="wide")
st.title("🧠 HF Dataset Gallery + Visual QA")

# --- Sidebar: add images (upload + SDXL) ---
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
                except Exception as e:
                    st.error(f"Generation failed: {e}")

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
                    img_url = to_public_url(st.session_state.selected_path)
                    result = run_vqa(img_url, vqa_prompt.strip())
                    st.session_state.last_answer = result or "(empty)"
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

# @st.cache_data(show_spinner=False)
# def get_gallery_paths(cache_version: int):
#     _ = cache_version  # cache key
#     return list_dataset_image_paths()

cache_version = st.session_state.get("gallery_cache_version", 0)
paths = list_dataset_image_paths()# if refresh else get_gallery_paths(cache_version)

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
