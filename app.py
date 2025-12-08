import os
from huggingface_hub import InferenceClient, HfApi

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

api = HfApi(token=os.environ["HF_WRITE"])

repo_id = "ky3ow/lab7-images"

def upload_image(path: str) -> str:
    filename = os.path.basename(path)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset")

    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

url = upload_image(os.path.expanduser("~/Documents/ai/cat"))

completion = client.chat.completions.create(
    model="zai-org/GLM-4.1V-9B-Thinking",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    },
                }
            ]
        }
    ],
)

print(completion.choices[0].message.content)
