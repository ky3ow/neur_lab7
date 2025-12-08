import os
import streamlit as st
from huggingface_hub import InferenceClient, HfApi
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- Setup API clients ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_WRITE = os.getenv("HF_WRITE")
REPO_ID = "ky3ow/lab7-images"

client = InferenceClient(api_key=HF_TOKEN)
api = HfApi(token=HF_WRITE)

# --- Helper function ---
def upload_image(file: UploadedFile) -> str:
    """Upload image to HF dataset repo and return its public URL."""
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file.name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file.name}"

def run_completion(image_url: str, prompt: str):
    """Call the model with a prompt and an image URL."""
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
        return

    return completion.choices[0].message.content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="HF Image Completion App", layout="centered")
st.title("🧠 Hugging Face Image Completion App")

tab1, tab2 = st.tabs(["Placeholder", "Image + Prompt"])

# --- Placeholder Tab ---
with tab1:
    st.write("This is a placeholder tab — future features can go here!")

# --- Image Completion Tab ---
with tab2:
    st.subheader("Upload an image and generate a description")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    prompt = st.text_area("Enter your prompt", "Describe this image in one sentence.")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width='content')

    if st.button("Run Completion", type="primary", disabled=uploaded_file is None):
        with st.spinner("Uploading and generating response..."):
            if uploaded_file:
                image_url = upload_image(uploaded_file)
                result = run_completion(image_url, prompt)

                st.success("Response received!")
                st.markdown("**Model Output:**")
                st.write(result)
