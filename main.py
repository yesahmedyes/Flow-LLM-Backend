import os
from fastapi import FastAPI
from handlers import handle_image, handle_pdf, handle_text, upload_vectors
from dotenv import load_dotenv
from s3_handler import download_file_from_s3
from pydantic import BaseModel

load_dotenv()

app = FastAPI()


class ParseRequest(BaseModel):
    object_name: str
    user_id: str


@app.post("/api/parse-and-embed")
async def parse_and_embed(request: ParseRequest) -> None:
    object_name = request.object_name
    user_id = request.user_id

    download_file_from_s3("flowllm-bucket", object_name)

    if object_name.endswith(".pdf"):
        vector_embeddings = await handle_pdf(object_name)
    elif object_name.endswith((".png", ".jpg", ".jpeg", ".gif", ".tiff", ".webp")):
        vector_embeddings = await handle_image(object_name)
    else:
        vector_embeddings = await handle_text(object_name)

    if vector_embeddings:
        await upload_vectors(user_id, vector_embeddings)

    os.remove(object_name)
