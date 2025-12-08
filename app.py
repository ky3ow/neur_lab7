import os
import base64
from huggingface_hub import InferenceClient, HfApi

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

api = HfApi(token=os.environ["HF_WRITE"])

repo_id = "ky3ow/lab7-images"

info = api.upload_file(
    path_or_fileobj=os.path.expanduser("~/Documents/ai/cat"),
    path_in_repo="cat.avif",
    repo_id=repo_id,
    repo_type="dataset",
)

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
                        "url": "https://huggingface.co/datasets/ky3ow/lab7-images/resolve/main/cat.avif"
                    },
                }
            ]
        }
    ],
)

print(completion.choices[0].message.content)
