import os
import io
import streamlit as st
from huggingface_hub import InferenceClient, HfApi
from streamlit.runtime.uploaded_file_manager import UploadedFile
from PIL import Image

# --- Setup API clients ---
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "ky3ow/lab7-images"

client = InferenceClient(api_key=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# --- Helper functions ---
def upload_image(file: UploadedFile) -> str:
    """Upload image to HF dataset repo and return its public URL."""
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file.name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file.name}"

def upload_bytes_as_file(data: bytes, file_name: str) -> str:
    """Upload raw bytes as a file to the HF dataset and return the public URL."""
    api.upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=file_name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{file_name}"

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

def generate_image(prompt: str) -> Image.Image:
    """Generate an image using SDXL base via Hugging Face InferenceClient."""
    # Output is a PIL.Image object
    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
    return image

# --- Streamlit UI ---
st.set_page_config(page_title="HF Image Completion App", layout="centered")
st.title("🧠 Hugging Face Image Completion App")

tab_sd, tab_placeholder, tab_completion = st.tabs(
    ["Stable Diffusion (Text → Image)", "Placeholder", "Image + Prompt"]
)

# --- Stable Diffusion Tab ---
with tab_sd:
    st.subheader("Generate an image from a text prompt (SDXL) and upload to dataset")
    file_name = st.text_input(
        "Output file name (without extension)", value="generated_image"
    )
    sd_prompt = st.text_area("Enter your prompt", "Astronaut riding a horse")

    gen_btn = st.button("Generate and Upload", type="primary")
    if gen_btn:
        if not sd_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating image with SDXL..."):
                try:
                    img = generate_image(sd_prompt.strip())
                    st.image(img, caption="Generated Image", width="content")

                    # Convert to PNG bytes
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    png_bytes = buf.getvalue()

                    safe_name = (file_name.strip() or "generated_image") + ".png"

                    # Upload to HF dataset repo
                    with st.spinner("Uploading image to dataset…"):
                        public_url = upload_bytes_as_file(png_bytes, safe_name)

                    st.success("Image generated and uploaded successfully!")
                    st.markdown("Public URL:")
                    st.code(public_url)

                    # Optional: provide a download button too
                    st.download_button(
                        label="Download PNG",
                        data=png_bytes,
                        file_name=safe_name,
                        mime="image/png",
                    )
                except Exception as e:
                    st.error(f"Image generation or upload failed: {e}")

# --- Placeholder Tab ---
with tab_placeholder:
    st.write("This is a placeholder tab — future features can go here!")

# --- Image Completion Tab ---
with tab_completion:
    st.subheader("Upload an image and generate a description")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    prompt = st.text_area("Enter your prompt", "Describe this image in one sentence.")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width="content")

    if st.button("Run Completion", type="primary", disabled=uploaded_file is None):
        with st.spinner("Uploading and generating response..."):
            if uploaded_file:
                image_url = upload_image(uploaded_file)
                result = run_completion(image_url, prompt)

                st.success("Response received!")
                st.markdown("**Model Output:**")
                st.write(result)
